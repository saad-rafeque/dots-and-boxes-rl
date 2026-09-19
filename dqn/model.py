"""Dueling Double-DQN network for Dots & Boxes.

Q(s, a) is the expected REMAINING margin for the player to move in s, divided by
the number of boxes, so it lives in [-1, 1] and is directly comparable across
board sizes.  Dueling split: Q = V(s) + A(s,a) - mean_legal A(s, .)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from game import C_IN


class GlobalPoolBias(nn.Module):
    """Lets every cell see board-wide facts (parity, score) that a 3x3 kernel
    can never reach on its own."""

    def __init__(self, ch):
        super().__init__()
        self.fc = nn.Linear(2 * ch, ch)

    def forward(self, x):
        g = torch.cat([x.mean(dim=(2, 3)), x.amax(dim=(2, 3))], dim=1)
        return self.fc(g)[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, ch, gpool):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)
        self.gp = GlobalPoolBias(ch) if gpool else None

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        if self.gp is not None:
            y = y + self.gp(y)
        return F.relu(x + self.b2(self.c2(y)))


class DuelingQNet(nn.Module):
    def __init__(self, channels=64, blocks=6):
        super().__init__()
        self.channels, self.blocks = channels, blocks
        self.stem = nn.Sequential(nn.Conv2d(C_IN, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU())
        self.trunk = nn.Sequential(*[ResBlock(channels, gpool=(i % 3 == 2))
                                     for i in range(blocks)])
        self.adv = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False),
                                 nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.val_conv = nn.Sequential(nn.Conv2d(channels, 32, 1, bias=False),
                                      nn.BatchNorm2d(32), nn.ReLU())
        self.val_fc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, line_flat):
        h = self.trunk(self.stem(x))
        n = x.shape[0]
        a = self.adv(h).reshape(n, -1).index_select(1, line_flat)
        vh = self.val_conv(h)
        vg = torch.cat([vh.mean(dim=(2, 3)), vh.amax(dim=(2, 3))], dim=1)
        v = torch.tanh(self.val_fc(vg)).squeeze(1)
        return a, v


NEG = -1e9


def q_from(adv, v, legal):
    """Combine the dueling heads, with illegal moves removed from the mean and
    pushed to -inf so they can never be selected."""
    legal_f = legal.float()
    n_legal = legal_f.sum(1).clamp(min=1.0)
    adv_z = adv * legal_f
    mean_a = adv_z.sum(1) / n_legal
    q = v[:, None] + adv_z - mean_a[:, None]
    return q.masked_fill(~legal, NEG)
