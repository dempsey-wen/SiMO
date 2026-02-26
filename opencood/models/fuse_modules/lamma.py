


import math
import random
from numpy import record
import torch
from torch import nn
from torchvision.models.resnet import resnet18
from torch.nn import functional as F
from collections import OrderedDict

from einops import rearrange, repeat


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, qkv_bias, norm=nn.LayerNorm):
        super().__init__()

        self.scale = dim_head ** -0.5

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head,
                                                       bias=qkv_bias))
        self.to_k = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head,
                                                       bias=qkv_bias))
        self.to_v = nn.Sequential(norm(dim), nn.Linear(dim, heads * dim_head,
                                                       bias=qkv_bias))

        self.proj = nn.Linear(heads * dim_head, dim)
        self.prenorm = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim))
        self.postnorm = norm(dim)

    def forward(self, q, k, v, H, W, skip=None):
        """
        q: (b l d n)
        k: (b l d N)
        v: (b l d N)
        skip: (b d H W), optional

        return: (b l d H W)
        """
        B, L, D, _ = q.shape

        # Move feature dim to last for multi-head proj
        q = rearrange(q, 'b l d n -> (b l) n d')
        k = rearrange(k, 'b l d N -> (b l) N d')
        v = rearrange(v, 'b l d N -> (b l) N d')

        # Project with multiple heads
        q = self.to_q(q)                                                    # b n (heads dim_head)
        k = self.to_k(k)                                                    # b N (heads dim_head)
        v = self.to_v(v)                                                    # b N (heads dim_head)
        # print(q.shape, k.shape, v.shape)

        # Group the head dim with batch dim
        q = rearrange(q, 'b ... (m d) -> (b m) ... d', m=self.heads,
                      d=self.dim_head)                                      # (b m) n d, b=b*l
        k = rearrange(k, 'b ... (m d) -> (b m) ... d', m=self.heads,
                      d=self.dim_head)                                      # (b m) N d
        v = rearrange(v, 'b ... (m d) -> (b m) ... d', m=self.heads,
                      d=self.dim_head)                                      # (b m) N d

        # Dot product attention along cameras
        dot = self.scale * torch.einsum('b Q d, b K d -> b Q K', q, k)      # (b m) n d, (b m) N d -> (b m) n N
        att = dot.softmax(dim=-1)

        # Combine values (image level features).
        a = torch.einsum('b Q K, b K d -> b Q d', att, v)                   # (b m) n N, (b m) N d -> (b m) n d
        a = rearrange(a, '(b m) ... d -> b ... (m d)', m=self.heads,
                      d=self.dim_head)                                      # (b m) n d -> b n (m d)

        # Combine multiple heads
        z = self.proj(a)                                                    # b n (m d) -> b n d
        # print(z.shape, skip.shape)

        # Optional skip connection
        if skip is not None:
            z = z + rearrange(skip, 'b d H W -> b (H W) d')

        z = self.prenorm(z)
        z = z + self.mlp(z)
        z = self.postnorm(z)
        z = rearrange(z, '(b l) (H W) d -> b l d H W', b=B, l=L, H=H, W=W)
        # print(z.shape)

        return z
    




class LAMMA3(nn.Module):
    """
    LAMMA module.
    ----
    Parameters:
        args: dictionary, configurations for LAMMA
        H: int, height of input feature map
        W: int, width of input feature map

    ----
    Input:
        cam_feat: (B, N, C, H, W)
        lidar_feat: (B, N, C, H, W)
    Output:
        fused_feat: (B, N, C, H, W)
    """
    def __init__(self, args, H, W):
        super().__init__()
        self.args = args

        feat_dim = args['feat_dim']
        dim = args['dim']
        heads = args['heads']
        self.single_mode = args.get('single_mode', False)
        self.random_drop = args.get('random_drop', False)
        self.lidar_drop_ratio = args.get('lidar_drop_ratio', False)
        print(f"Single mode: {self.single_mode}")
        print(f"Random drop: {self.random_drop}, lidar drop ratio: {self.lidar_drop_ratio}")

        self.pos_embed = torch.nn.Parameter(torch.randn(feat_dim, H, W))
        # self.query = torch.nn.Parameter(torch.randn(feat_dim, H, W))
        
        self.feature_proj = nn.Sequential(
                nn.InstanceNorm2d(feat_dim),
                nn.ReLU(),
                nn.Conv2d(feat_dim, dim, 2, stride=2, bias=False))
        
        self.cross_att = CrossAttention(dim=dim, heads=heads, dim_head=dim, qkv_bias=False)

        self.feature_proj_inv = nn.Sequential(
                nn.InstanceNorm2d(dim),
                nn.ReLU(),
                nn.ConvTranspose2d(dim, feat_dim, 2, stride=2, bias=False))

    def forward(self, cam_feat, lidar_feat):
        # feature shape: torch.Size([1, 3, 64, 64, 64])
        B, N, C, Y, X = lidar_feat.shape 
        assert cam_feat.shape == lidar_feat.shape, "cam_feat and lidar_feat should have the same shape"

        # query = repeat(self.query, 'c h w -> b l c h w', b=B, l=N)                      # b l c h w
        bev_plane = repeat(self.pos_embed, 'c h w -> b l c h w', b=B, l=N)              # b l c h w
        cam_feat = cam_feat + bev_plane                                                 # b l c h w
        lidar_feat = lidar_feat + bev_plane                                             # b l c h w

        # query = rearrange(query, 'b l c h w -> (b l) c h w')                            # (b l) c h w
        cam_feat = rearrange(cam_feat, 'b l c h w -> (b l) c h w')                      # (b l) c h w
        lidar_feat = rearrange(lidar_feat, 'b l c h w -> (b l) c h w')                  # (b l) c h w

        # query = self.feature_proj(query)                                                # (b l) d h w
        cam_embed = self.feature_proj(cam_feat)                                         # (b l) d h w
        lidar_embed = self.feature_proj(lidar_feat)                                     # (b l) d h w
        # print(cam_embed.shape, lidar_embed.shape)

        mask = torch.zeros_like(lidar_embed)
        if self.single_mode == 'camera':
            # print('lidar drop')
            lidar_embed *= mask
        elif self.single_mode == 'lidar':
            cam_embed *= mask
        elif self.random_drop:
            # in training, to adapt to modality drop out, 
            # with 0.5 to drop lidar or camera (drop lidar features with lidar_drop_ratio)
            # with 0.5 applying two modalities
            if random.random() >= 0.5:
                if self.lidar_drop_ratio >= random.random():
                    lidar_embed *= mask
                else:
                    cam_embed *= mask

        # query = rearrange(query, '(b l) d h w -> b l d (h w)', b=B, l=N)                # b l d (h w)
        cam_flat = rearrange(cam_embed, '(b l) d h w -> b l d (h w)', b=B, l=N)         # b l d (h w)
        lidar_flat = rearrange(lidar_embed, '(b l) d h w -> b l d (h w)', b=B, l=N)     # b l d (h w)

        query = torch.cat([cam_flat, lidar_flat], dim=-1)                               # b l d (2h w)
        # key = torch.cat([cam_flat, lidar_flat], dim=-1)                                 # b l d (2h w)
        # value = torch.cat([cam_flat, lidar_flat], dim=-1)                               # b l d (2h w)
        cam_lidar_embed = torch.cat([cam_embed, lidar_embed], dim=-1)                   # b l d (2h w)
        # fused_feat = self.cross_att(query, key, value, 2*Y, X, skip=cam_lidar_embed)    # b l d 2h w
        # fused_feat = rearrange(fused_feat, 'b l d (m h)  w -> b l d h w m', m=2)        # b l d h w 2
        # cam_fused, lidar_fused = fused_feat[:, :, :, :, :, 0], fused_feat[:, :, :, :, :, 1]
        # print(fused_feat.shape, cam_fused.shape, lidar_fused.shape)


        cam_fused = self.cross_att(query, cam_flat, cam_flat, Y, int(X/2), skip=cam_lidar_embed)     # b l d 2h w
        lidar_fused = self.cross_att(query, lidar_flat, lidar_flat, Y, int(X/2), skip=cam_lidar_embed) # b l d 2h w

        fused_feat = cam_fused + lidar_fused
        fused_feat = rearrange(fused_feat, 'b l d (m h)  w -> b l d h w m', m=2)        # b l d h w 2


        fused_feat = self.feature_proj_inv(
                        rearrange(fused_feat.sum(dim=-1), 'b l d h w -> (b l) d h w'))  # (b l) c h w
        cam_fused = self.feature_proj_inv(
                        rearrange(cam_fused, 'b l d h w -> (b l) d h w'))               # (b l) c h w
        lidar_fused = self.feature_proj_inv(
                        rearrange(lidar_fused, 'b l d h w -> (b l) d h w'))             # (b l) c h w
        
        return fused_feat, cam_fused, lidar_fused    
    
# if __name__ == '__main__':
#     test_1 = torch.randn(1, 2, 64, 20, 20)
#     test_2 = torch.randn(1, 2, 64, 20, 20)
#     # test_2 = torch.zeros_like(test_1)
    
#     args = {
#         'feat_dim': 64, 
#         'dim': 72, 
#         'heads':2,
#         'single_mode': 'camera',
#         'lidar_drop_ratio': 0.9}

#     lamma = LAMMA3(args, H=20, W=20)

#     print(lamma(test_1, test_2)[0].shape)
