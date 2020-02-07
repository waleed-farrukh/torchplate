import math

import torch
import torch.nn as nn


class YourBasicModel(nn.Module):
    def __init__(self, input_height, input_width):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                kernel_size=(3, 3),
                in_channels=3,
                out_channels=24,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(24)
        )

        # for calculating spatial dimensions of feature layer. This enables input image of arbitrary size
        rand_inp = torch.rand(1, 3, input_height, input_width)
        feature_dim = self.features(rand_inp).shape

        self.classifier = nn.Linear(24 * feature_dim[2] * feature_dim[3], 4)

    def forward(self, input: torch.Tensor):
        conv_out = self.conv(input)
        flattened = conv_out.view(conv_out.size(0), self.classifier[1].in_features)
        class_logits = self.classifier(flattened)

        return class_logits

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.uniform_(-stdv, stdv)
            if m.bias is not None:
                m.bias.data.uniform_(-stdv, stdv)
