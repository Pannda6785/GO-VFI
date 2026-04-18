#!/usr/bin/env python

import torch


def _flatten_scatter_add(output, values, linear_idx):
    channels = values.shape[1]
    flat_output = output.view(output.shape[0], output.shape[1], -1)
    flat_values = values.reshape(values.shape[0], channels, -1)
    flat_idx = linear_idx.reshape(linear_idx.shape[0], -1).unsqueeze(1).expand(-1, channels, -1)
    flat_output.scatter_add_(2, flat_idx, flat_values)
    return output


def _softsplat_forward(tenInput, tenFlow):
    n, c, h, w = tenInput.shape
    device = tenInput.device
    dtype = tenInput.dtype

    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(n, -1, -1) + tenFlow[:, 0]
    yy = yy.unsqueeze(0).expand(n, -1, -1) + tenFlow[:, 1]

    x0 = torch.floor(xx)
    y0 = torch.floor(yy)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1 - xx) * (y1 - yy)
    wb = (xx - x0) * (y1 - yy)
    wc = (x1 - xx) * (yy - y0)
    wd = (xx - x0) * (yy - y0)

    output = tenInput.new_zeros((n, c, h, w))

    for x_idx, y_idx, weight in (
        (x0, y0, wa),
        (x1, y0, wb),
        (x0, y1, wc),
        (x1, y1, wd),
    ):
        x_long = x_idx.long()
        y_long = y_idx.long()
        valid = (x_long >= 0) & (x_long < w) & (y_long >= 0) & (y_long < h)
        if not torch.any(valid):
            continue
        linear_idx = (y_long.clamp(0, h - 1) * w + x_long.clamp(0, w - 1)).long()
        weighted_values = tenInput * (weight * valid.to(dtype)).unsqueeze(1)
        _flatten_scatter_add(output, weighted_values, linear_idx)

    return output


def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ["summation", "average", "linear", "softmax"]

    if strType == "average":
        tenInput = torch.cat(
            [tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3])],
            1,
        )
    elif strType == "linear":
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)
    elif strType == "softmax":
        metric = tenMetric.exp()
        tenInput = torch.cat([tenInput * metric, metric], 1)

    tenOutput = _softsplat_forward(tenInput, tenFlow)

    if strType != "summation":
        tenNormalize = tenOutput[:, -1:, :, :]
        tenNormalize = torch.where(tenNormalize == 0.0, torch.ones_like(tenNormalize), tenNormalize)
        tenOutput = tenOutput[:, :-1, :, :] / tenNormalize

    return tenOutput


class ModuleSoftsplat(torch.nn.Module):
    def __init__(self, strType):
        super(ModuleSoftsplat, self).__init__()
        self.strType = strType

    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)

