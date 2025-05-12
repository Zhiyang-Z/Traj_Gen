import torch
import torch.nn as nn
from einops import rearrange
from timm.layers.helpers import to_2tuple

class PatchEmbed1D(nn.Module):
    """1D Traj to Patch Embedding"""
    def __init__(
        self,
        traj_length=200,
        patch_size=1,
        in_chans=2,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        assert traj_length % patch_size == 0, "traj_length must be divisible by patch_size"
        self.out_length = traj_length // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)
        assert x.shape[2] == self.out_length, f"Input length ({L}) doesn't match model ({self.out_length})."
        x = rearrange(x, "B C L -> B L C")
        x = self.norm(x)
        return x

class XPatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    def __init__(
        self,
        img_height=2,
        img_width=200,
        patch_size=1,
        in_chans=1,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        img_size = (img_height, img_width)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, random_sample=False):
        B, C, H, W = x.shape
        assert random_sample or (H == self.img_size[0] and W == self.img_size[1]), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            # print(x[0,:,:,0:4])
            x = rearrange(x, "B C H W -> B W H C")
            x = rearrange(x, "B W H C -> B (W H) C")
            # print(x[0,0:4,:])
        else:
            x = rearrange(x, "B C H W -> B H W C")
        x = self.norm(x)
        return x