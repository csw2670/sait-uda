import torch
import torch.nn as nn

def _convTranspose2dOutput(
    input_size: int,
    stride: int,
    padding: int,
    dilation: int,
    kernel_size: int,
    output_padding: int,
):
    """
    Calculate the output size of a ConvTranspose2d.
    Taken from: https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
    """
    return (
        (input_size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )

class Norm2d(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class FCNDecoder(nn.Module):
    """
    Neck that transforms the token-based output of transformer into a single embedding suitable for processing with standard layers.
    Performs 4 ConvTranspose2d operations on the rearranged input with kernel_size=2 and stride=2
    embed_dim (int): Input embedding dimension
        output_embed_dim (int): Output embedding dimension
        Hp (int, optional): Height (in patches) of embedding to be upscaled. Defaults to 14.
        Wp (int, optional): Width (in patches) of embedding to be upscaled. Defaults to 14.
        drop_cls_token (bool, optional): Whether there is a cls_token, which should be dropped. This assumes the cls token is the first token. Defaults to True.
    """

    def __init__(self, embed_dim: int = 1024, output_embed_dim: int = 64, # num_frames: int = 1,
        Hp: int = 7, Wp: int = 7, drop_cls_token: bool = True,):
        super().__init__()
        
        self.drop_cls_token = drop_cls_token
        self.Hp = Hp
        self.Wp = Wp
        self.H_out = 224
        self.W_out = 224
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.num_frames = num_frames
        
        self.inter_channels = output_embed_dim // 4
        self.out_channels = 13
        self.layers = nn.Sequential(
            nn.Conv2d(output_embed_dim, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(self.inter_channels, self.out_channels, 1),
        )

        kernel_size = 2
        stride = 2
        dilation = 1
        padding = 0
        output_padding = 0
        for _ in range(4):
            self.H_out = _convTranspose2dOutput(
                self.H_out, stride, padding, dilation, kernel_size, output_padding
            )
            self.W_out = _convTranspose2dOutput(
                self.W_out, stride, padding, dilation, kernel_size, output_padding
            )

        self.embed_dim = embed_dim
        self.output_embed_dim = output_embed_dim
        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(
                self.embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
            Norm2d(self.output_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(
                self.output_embed_dim,
                self.output_embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=output_padding,
            ),
        )

    def forward(self, x):
        if self.drop_cls_token:
            x = x[:, 1:, :] #(B, 49, 1024)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, self.Hp, self.Wp) #(B, 1024, 7, 7)
        
        x = self.fpn1(x) #(B, 64, 28, 28)
        x = self.fpn2(x) #(B, 64, 112, 112)
        x = self.upsample(x) #(B, 64, 224, 224)
        out = self.layers(x)
        
        return out