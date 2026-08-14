"""Depth-heterogeneous ResNet-18 with early exits for DepthFL.

Ported from the Flower DepthFL baseline (Kim et al., ICLR 2023). The backbone
is a CIFAR ResNet-18 (3x3 stride-1 stem, no maxpool) with an auxiliary
classifier ("exit") after each of the four stages. A depth-``n_blocks`` client
holds the stem plus the first ``n_blocks`` stages and their exits, and its
forward returns one logit tensor per held exit.

Registration order of the submodules is preserved exactly so that a shallow
model's ``state_dict()`` is a strict prefix of the full depth-4 model's
``state_dict()``. That prefix property is what lets the server distribute
depth-``d`` sub-models and aggregate heterogeneous depths by parameter position.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class MyGroupNorm(nn.Module):
    """Group Normalization layer (16 groups)."""

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=16, num_channels=num_channels, eps=1e-5, affine=True)

    def forward(self, x):
        return self.norm(x)


class MyBatchNorm(nn.Module):
    """Batch Normalization layer (tracks running stats)."""

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_channels, track_running_stats=True)

    def forward(self, x):
        return self.norm(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, planes, stride=1):
    return nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)


class SepConv(nn.Module):
    """Depthwise-separable bottleneck used in the exit branches."""

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1,
                 norm_layer=MyGroupNorm):
        super().__init__()
        self.operations = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_in),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1,
                      padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            norm_layer(channel_out),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.operations(x)


class BasicBlock(nn.Module):
    """ResNet-18 basic block."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        if self.downsample is not None:
            residual = self.downsample(x)
        output += residual
        output = self.relu(output)
        return output


class MultiResnet(nn.Module):
    """ResNet-18 with early-exit classifiers at each of four stages."""

    def __init__(self, block, layers, n_blocks, num_classes=100, norm_layer=MyBatchNorm):
        super().__init__()
        self.n_blocks = n_blocks
        self.inplanes = 64
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.middle_fc1 = nn.Linear(512 * block.expansion, num_classes)
        self.scala1 = nn.Sequential(
            SepConv(64 * block.expansion, 128 * block.expansion, norm_layer=norm_layer),
            SepConv(128 * block.expansion, 256 * block.expansion, norm_layer=norm_layer),
            SepConv(256 * block.expansion, 512 * block.expansion, norm_layer=norm_layer),
            nn.AdaptiveAvgPool2d(1),
        )
        self.attention1 = nn.Sequential(
            SepConv(64 * block.expansion, 64 * block.expansion, norm_layer=norm_layer),
            norm_layer(64 * block.expansion),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Sigmoid(),
        )

        if n_blocks > 1:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.middle_fc2 = nn.Linear(512 * block.expansion, num_classes)
            self.scala2 = nn.Sequential(
                SepConv(128 * block.expansion, 256 * block.expansion, norm_layer=norm_layer),
                SepConv(256 * block.expansion, 512 * block.expansion, norm_layer=norm_layer),
                nn.AdaptiveAvgPool2d(1),
            )
            self.attention2 = nn.Sequential(
                SepConv(128 * block.expansion, 128 * block.expansion, norm_layer=norm_layer),
                norm_layer(128 * block.expansion),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Sigmoid(),
            )

        if n_blocks > 2:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.middle_fc3 = nn.Linear(512 * block.expansion, num_classes)
            self.scala3 = nn.Sequential(
                SepConv(256 * block.expansion, 512 * block.expansion, norm_layer=norm_layer),
                nn.AdaptiveAvgPool2d(1),
            )
            self.attention3 = nn.Sequential(
                SepConv(256 * block.expansion, 256 * block.expansion, norm_layer=norm_layer),
                norm_layer(256 * block.expansion),
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Sigmoid(),
            )

        if n_blocks > 3:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.fc_layer = nn.Linear(512 * block.expansion, num_classes)
            self.scala4 = nn.AdaptiveAvgPool2d(1)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, block, planes, layers, stride=1, norm_layer=None):
        norm_layer = self.norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layer = [block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, layers):
            layer.append(block(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        fea1 = self.attention1(x)
        fea1 = fea1 * x
        out1_feature = self.scala1(fea1).view(x.size(0), -1)
        middle_output1 = self.middle_fc1(out1_feature)
        if self.n_blocks == 1:
            return [middle_output1]

        x = self.layer2(x)
        fea2 = self.attention2(x)
        fea2 = fea2 * x
        out2_feature = self.scala2(fea2).view(x.size(0), -1)
        middle_output2 = self.middle_fc2(out2_feature)
        if self.n_blocks == 2:
            return [middle_output1, middle_output2]

        x = self.layer3(x)
        fea3 = self.attention3(x)
        fea3 = fea3 * x
        out3_feature = self.scala3(fea3).view(x.size(0), -1)
        middle_output3 = self.middle_fc3(out3_feature)
        if self.n_blocks == 3:
            return [middle_output1, middle_output2, middle_output3]

        x = self.layer4(x)
        out4_feature = self.scala4(x).view(x.size(0), -1)
        output4 = self.fc_layer(out4_feature)
        return [middle_output1, middle_output2, middle_output3, output4]


class KLLoss(nn.Module):
    """KL divergence loss for self-distillation (temperature = 1)."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, pred, label):
        predict = F.log_softmax(pred / self.temperature, dim=1)
        target_data = F.softmax(label / self.temperature, dim=1)
        target_data = target_data + 1e-7
        target = target_data.detach().clone()
        loss = (
            self.temperature * self.temperature
            * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        return loss


def create_depthfl_model(n_blocks: int, num_classes: int = 100, norm_type: str = "bn") -> MultiResnet:
    """Create a depth-``n_blocks`` MultiResnet-18 (n_blocks in {1,2,3,4})."""
    norm = str(norm_type).lower()
    norm_layer = MyGroupNorm if norm == "gn" else MyBatchNorm
    return MultiResnet(BasicBlock, [2, 2, 2, 2], int(n_blocks),
                       num_classes=num_classes, norm_layer=norm_layer)


def num_state_entries(n_blocks: int, num_classes: int = 100, norm_type: str = "bn") -> int:
    """Number of state_dict tensors held by a depth-``n_blocks`` model."""
    return len(create_depthfl_model(n_blocks, num_classes, norm_type).state_dict())


@torch.no_grad()
def test_depthfl(net: nn.Module, testloader: DataLoader, device: torch.device
                 ) -> Tuple[float, float, List[float]]:
    """Evaluate a full-depth model: ensemble accuracy plus per-exit accuracy.

    Returns (loss, ensemble_accuracy, [per-exit accuracy for exits 1..4]).
    The ensemble is the argmax of the mean of the exit logits.
    """
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    correct_single = [0, 0, 0, 0]
    net.eval()
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        output_lst = net(images)

        ensemble = torch.stack(output_lst, dim=2)
        ensemble = torch.sum(ensemble, dim=2) / len(output_lst)

        loss += criterion(ensemble, labels).item()
        _, predicted = torch.max(ensemble, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for i, single in enumerate(output_lst):
            _, predicted = torch.max(single, 1)
            correct_single[i] += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    accuracy_single = [c / total for c in correct_single]
    return loss, accuracy, accuracy_single
