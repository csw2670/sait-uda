import torch
import torch.nn as nn

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )

class UNet(nn.Module):
    def __init__(self, mae_model):
        super(UNet, self).__init__()
        self.mae_model = mae_model
        
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fc = nn.Linear(128, 196)

        self.dconv_up4 = double_conv(400, 400)
        self.dconv_up3 = double_conv(400, 320)
        self.dconv_up2 = double_conv(320, 160)
        self.dconv_up1 = double_conv(160, 64)

        self.conv_last = nn.Conv2d(64, 13, 1)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, imgs, mask_ratio=0.75):
        encoder_outputs, mask, ids_restore = self.mae_model.forward_encoder(imgs, mask_ratio)
        encoder_outputs = encoder_outputs.reshape(-1, 400, 128)  
        encoder_outputs = self.fc(encoder_outputs) # (16,400,196)
        encoder_outputs = encoder_outputs.reshape(-1, 400, 14, 14)
    
        x = self.dconv_up4(encoder_outputs) # (16,400,14,14)
        x = self.upsample(x) # (16,400,28,28)
        x = self.dropout(x)

        x = self.dconv_up3(x) # (16,320,28,28)
        x = self.upsample(x)  # (16,320,56,56)      
        x = self.dropout(x)
        
        x = self.dconv_up2(x) # (16,160,56,56)
        x = self.upsample(x) # (16,160,112,112)
        x = self.dropout(x)
        
        x = self.dconv_up1(x) # (16,64,112,112)
        x = self.upsample(x) # (16,64,224,224)

        x = self.conv_last(x) # (16,13,224,224)
        return x