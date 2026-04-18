#!/usr/bin/env python

import torch
import torch.nn.functional as F


def FunctionCorrelation(tenFirst, tenSecond):
    assert tenFirst.shape == tenSecond.shape
    pad = 4
    second = F.pad(tenSecond, (pad, pad, pad, pad))
    volumes = []
    for dy in range(-pad, pad + 1):
        y0 = dy + pad
        y1 = y0 + tenFirst.shape[2]
        for dx in range(-pad, pad + 1):
            x0 = dx + pad
            x1 = x0 + tenFirst.shape[3]
            shifted = second[:, :, y0:y1, x0:x1]
            volumes.append((tenFirst * shifted).mean(1, keepdim=True))
    return torch.cat(volumes, dim=1)


class ModuleCorrelation(torch.nn.Module):
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, tenFirst, tenSecond):
        return FunctionCorrelation(tenFirst, tenSecond)

