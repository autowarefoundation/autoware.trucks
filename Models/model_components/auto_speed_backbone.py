import torch
import torch.nn as nn
from Models.model_components.auto_speed_layers import (
    Conv, C3K2, SPPF, C2PSA, PSA, CSP, SPP, ASC
)


class AutoSpeedBackbone(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()

        # p1/2
        self.p1 = Conv(width[0], width[1], activation=torch.nn.SiLU(), k=3, s=2, p=1)

        # p2/4
        self.p2 = nn.Sequential(
            Conv(width[1], width[2], activation=torch.nn.SiLU(), k=3, s=2, p=1),
            # C3K2(width[2], width[3], depth[0], csp[0], r=4)
            ASC(width[2], width[3], depth[0], csp[0], r=2, h=160, w=160)
        )
        # p3/8
        self.p3 = nn.Sequential(
            Conv(width[3], width[3], activation=torch.nn.SiLU(), k=3, s=2, p=1),
            # C3K2(width[3], width[4], depth[1], csp[0], r=4)
            ASC(width[3], width[4], depth[1], csp[0], r=2, h=80, w=80)
        )
        # p4/16
        self.p4 = nn.Sequential(
            Conv(width[4], width[4], activation=torch.nn.SiLU(), k=3, s=2, p=1),
            # C3K2(width[4], width[4], depth[2], csp[1], r=2)
            ASC(width[4], width[4], depth[1], csp[0], r=2, h=40, w=40)
        )
        # p5/32
        self.p5 = nn.Sequential(
            Conv(width[4], width[5], activation=torch.nn.SiLU(), k=3, s=2, p=1),
            # C3K2(width[5], width[5], depth[3], csp[1], r=2),
            ASC(width[5], width[5], depth[1], csp[0], r=2, h=20, w=20),
            SPPF(width[5], width[5]),
            C2PSA(width[5], width[5])
        )

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5
