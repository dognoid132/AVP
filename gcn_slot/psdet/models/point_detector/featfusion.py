# 在文件顶部或单独文件中定义
import torch.nn as nn
import torch
import torch.nn.functional as F

class MultiscaleFusionWithAttention(nn.Module):
    def __init__(self, out_ch=512):
        super().__init__()
        self.proj_c4 = nn.Conv2d(512, out_ch, 1)
        self.proj_c3 = nn.Conv2d(256, out_ch, 1)
        self.down_c4 = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)  # /16 → /32
        self.down_c3 = nn.Conv2d(out_ch, out_ch, 3, stride=4, padding=1)  # /8 → /32
        self.attention = nn.Sequential(
            nn.Conv2d(out_ch * 3, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid()
        )
        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, c3, c4, c5):
        f3 = self.proj_c3(c3)
        f4 = self.proj_c4(c4)
        f5 = c5  # assume already 512

        f3_d = F.interpolate(f3, size=c5.shape[2:], mode='bilinear', align_corners=False)
        f4_d = F.interpolate(f4, size=c5.shape[2:], mode='bilinear', align_corners=False)

        w = self.attention(torch.cat([f3_d, f4_d, f5], dim=1))
        w3, w4, w5 = w.chunk(3, dim=1)
        fused = w3*f3_d + w4*f4_d + w5*f5
        return self.smooth(fused)