import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from sam3.model.vitdet import ViT


class LightBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4, 1),
            nn.GELU(),
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, 1),
            nn.BatchNorm2d(out_channels, 1),
            nn.GELU(),
        )
        self.dw1 = nn.Sequential(
            nn.Conv2d(
                in_channels // 8,
                in_channels // 8,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels // 8,
            ),
            nn.BatchNorm2d(in_channels // 8),
            nn.GELU(),
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(
                in_channels // 8,
                in_channels // 8,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_channels // 8,
            ),
            nn.BatchNorm2d(in_channels // 8),
            nn.GELU(),
        )

    def forward(self, x):
        x = self.conv_in(x)
        x1, x2 = torch.split(x, x.shape[1] // 2, 1)
        x3 = self.dw1(x2)
        x4 = self.dw2(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.conv_out(x)
        return x


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        self.conv = LightBlock(in_channels, out_channels)

    def forward(self, x1, x2=None):
        if x2 is not None:
            diffY = x1.size()[2] - x2.size()[2]
            diffX = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim), nn.GELU()
        )
        self.init_weights()

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.prompt_learn.apply(_init_weights)


def _create_vit_backbone(img_size):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        #   img_size=1008,
        img_size=img_size,
        pretrain_img_size=1008,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        # compile_mode=compile_mode,
        compile_mode=None,
    )


class SAM3UNet(nn.Module):
    def __init__(self, checkpoint_path=None, img_size=1008) -> None:
        super(SAM3UNet, self).__init__()
        self.sam3_vit = _create_vit_backbone(img_size)
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path)
            new_ckpt = dict()
            for k, v in ckpt.items():
                if "detector.backbone.vision_backbone.trunk" in k and "freqs_cis" not in k:
                    new_ckpt[k[len("detector.backbone.vision_backbone.trunk.") :]] = v

            if "pos_embed" in new_ckpt:
                pos_embed_checkpoint = new_ckpt["pos_embed"]
                embedding_size = pos_embed_checkpoint.shape[-1]  # 1024
                num_extra_tokens = 1  # Usually 1 for the class token

                # Calculate grid sizes
                # Old grid: sqrt(577 - 1) = 24
                # New grid: sqrt(2305 - 1) = 48
                old_grid_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5)
                new_grid_size = int((self.sam3_vit.pos_embed.shape[1] - num_extra_tokens) ** 0.5)

                if old_grid_size != new_grid_size:
                    print(f"Interpolating pos_embed from {old_grid_size} to {new_grid_size}")

                    # Separate tokens and grid
                    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    grid_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

                    # Reshape to (1, C, H, W) for interpolation
                    grid_tokens = grid_tokens.reshape(
                        1, old_grid_size, old_grid_size, embedding_size
                    ).permute(0, 3, 1, 2)

                    # Interpolate
                    grid_tokens = F.interpolate(
                        grid_tokens,
                        size=(new_grid_size, new_grid_size),
                        mode="bicubic",
                        align_corners=False,
                    )

                    # Reshape back to (1, L, C)
                    grid_tokens = grid_tokens.permute(0, 2, 3, 1).reshape(1, -1, embedding_size)

                    # Concatenate back
                    new_ckpt["pos_embed"] = torch.cat((extra_tokens, grid_tokens), dim=1)

            self.sam3_vit.load_state_dict(new_ckpt, strict=False)
        for param in self.sam3_vit.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.sam3_vit.blocks:
            blocks.append(Adapter(block))
        self.sam3_vit.blocks = nn.Sequential(*blocks)
        self.reduce1 = nn.Conv2d(1024, 128, 1)
        self.reduce2 = nn.Conv2d(1024, 128, 1)
        self.reduce3 = nn.Conv2d(1024, 128, 1)
        self.reduce4 = nn.Conv2d(1024, 128, 1)
        self.up1 = Up(256, 128)
        self.up2 = Up(256, 128)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 128)
        self.head = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.sam3_vit(x)[-1]
        x1 = F.interpolate(self.reduce1(x), size=(H // 4, W // 4), mode="bilinear")
        x2 = F.interpolate(self.reduce2(x), size=(H // 8, W // 8), mode="bilinear")
        x3 = F.interpolate(self.reduce3(x), size=(H // 16, W // 16), mode="bilinear")
        x4 = F.interpolate(self.reduce4(x), size=(H // 32, W // 32), mode="bilinear")
        x = self.up4(x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        out = self.head(x)
        out = F.interpolate(out, size=(H, W), mode="bilinear")
        return out


if __name__ == "__main__":
    model = SAM3UNet().cuda().eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 1008, 1008).cuda()
        out = model(x)
        print(out.shape)
