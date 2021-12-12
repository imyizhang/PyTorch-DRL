#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class MLPApproximator(torch.nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        activation=torch.nn.ReLU(),
        out_activation=torch.nn.Identity(),
    ):
        super().__init__()
        features = (in_features,) + hidden_features + (out_features,)
        num_affine_maps = len(features) - 1
        layers = []
        for i in range(num_affine_maps):
            nonlinearity = activation if i < num_affine_maps - 1 else out_activation
            layers += [
                torch.nn.Linear(features[i], features[i + 1], bias=True),
                nonlinearity,
             ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
