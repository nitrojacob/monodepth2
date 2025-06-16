# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn


class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = num_input_frames

        convs = []
        convs.append(nn.Conv2d(3 * num_input_frames, 16, 7, 2, 3))
        convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(16, 32, 5, 2, 2))
        convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(32, 64, 3, 2, 1))
        convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(64, 128, 3, 2, 1))
        convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(128, 256, 3, 2, 1))
        convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(256, 256, 3, 2, 1))
        convs.append(nn.ReLU(True))
        convs.append(nn.Conv2d(256, 256, 3, 2, 1))
        convs.append(nn.ReLU(True))

        pose_pred = nn.Conv2d(256, 6 * (num_input_frames - 1), 1)

        convs.append(pose_pred)
        self.net = nn.Sequential(*convs)

    def forward(self, out):
        out = self.net(out)
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
