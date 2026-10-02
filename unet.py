import math
import torch
from torch import nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_cha = (256, 128, 64)
        self.down_cha = (64, 128, 256)
        self.conv0 = nn.Conv2d(1, self.down_cha[0], kernel_size=3, stride=1, padding=1)
        self.convtranspose0 = nn.Conv2d(self.down_cha[0], 1, kernel_size=3, stride=1, padding=1)
        self.downs = nn.ModuleList(Block(self.down_cha[i], self.down_cha[i + 1], up=False) for i in range(len(self.down_cha) - 1))
        self.ups = nn.ModuleList(Block(self.up_cha[i], self.up_cha[i + 1], up=True) for i in range(len(self.up_cha) - 1))
        self.relu = nn.ReLU(inplace=True)
        self.bnorm1 = nn.BatchNorm2d(self.up_cha[0])
        self.convv1 = nn.Conv2d(self.up_cha[0], self.up_cha[0], kernel_size=3, stride=1, padding=1)
        self.convv2 = nn.Conv2d(self.up_cha[0], self.up_cha[0], kernel_size=3, stride=1, padding=1)

    def forward(self, x): ## no residual
        x = self.conv0(x)
        residual = []
        for i in self.downs:
            x = i(x)
            residual.append(x)
        x = self.relu(self.convv1(x))
        x = self.convv2(x)
        for i in self.ups:
            x = torch.cat((residual.pop(), x), dim=1)
            x = i(x)
        return self.convtranspose0(x)




class Block(nn.Module):
    def __init__(self, in_cha, out_cha, up):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bnorm1 = nn.BatchNorm2d(out_cha)
        self.bnorm2 = nn.BatchNorm2d(out_cha)
        ch = 2 if up else 1
        self.q_proj = nn.Conv2d(ch*in_cha, ch*in_cha, kernel_size=1, stride=1, padding=0)
        self.k_proj = nn.Conv2d(ch*in_cha, ch*in_cha, kernel_size=1, stride=1, padding=0)
        self.v_proj = nn.Conv2d(ch*in_cha, ch*in_cha, kernel_size=1, stride=1, padding=0)
        self.proj = nn.Conv2d(ch*in_cha, ch*in_cha, kernel_size=1, stride=1, padding=0)
        for module in [self.q_proj, self.k_proj, self.v_proj, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)
        if up:
            self.conv1 = nn.Conv2d(in_cha*2, out_cha, kernel_size=3, stride=1, padding=1)
            self.transform = nn.ConvTranspose2d(out_cha, out_cha, kernel_size=3, stride=2, padding=1, output_padding=1)
            self.group_norm = nn.GroupNorm(2*in_cha, 2*in_cha)
        else:
            self.conv1 = nn.Conv2d(in_cha, out_cha, kernel_size=3, stride=1, padding=1)
            self.transform = nn.Conv2d(out_cha, out_cha, kernel_size=3, stride=2, padding=1)
            self.group_norm = nn.GroupNorm(in_cha, in_cha)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,HW,C]
        k = k.reshape(B, C, H * W)  # [B,C,HW]
        w = torch.bmm(q, k) * (int(C) ** (-0.5)) #[B,HW,HW]
        w.reshape(B, -1)
        w = F.softmax(w, dim=-1)
        w = w.reshape(B, H*W, H*W)

        v = v.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B,HW,C]
        h = torch.bmm(w, v)  # [B,HW,C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)  # [B,C,H,W]
        h = self.proj(h)  # [B,C,H,W]
        x = x+h
        x = self.bnorm1(self.relu(self.conv1(x)))
        x = self.bnorm2(self.relu(self.transform(x)))
        return x
