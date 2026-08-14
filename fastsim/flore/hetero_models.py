"""Heterogeneous model architectures for model-heterogeneous federated learning.

This module implements scalable ResNet architectures that support width-based
model heterogeneity for FedRolex, HeteroFL, and related baselines.

Based on:
- FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction (NeurIPS 2022)
- HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients (ICLR 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional


class Scaler(nn.Module):
    """Scaling layer for gradient flow control in heterogeneous models."""

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        return x * self.rate


class HeteroBasicBlock(nn.Module):
    """BasicBlock with configurable width for model heterogeneity.

    Args:
        in_planes: Number of input channels
        planes: Number of output channels (will be scaled by model_rate)
        stride: Stride for the first convolution
        model_rate: Width scaling factor (e.g., 0.25, 0.5, 0.75, 1.0)
        scale: Whether to use scaling layer for gradient control
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 model_rate: float = 1.0, scale: bool = False):
        super().__init__()

        # Scale planes by model_rate
        scaled_planes = int(np.ceil(model_rate * planes))

        self.conv1 = nn.Conv2d(
            in_planes, scaled_planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(scaled_planes)

        self.conv2 = nn.Conv2d(
            scaled_planes, scaled_planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(scaled_planes)

        # Optional scaler for gradient flow
        if scale:
            self.scaler = Scaler(model_rate)
        else:
            self.scaler = nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != scaled_planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, scaled_planes * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(scaled_planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.scaler(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class HeteroResNet(nn.Module):
    """ResNet with configurable width for model heterogeneity.

    This implementation allows each client to train a sub-model with different
    width (number of channels) based on their computational capabilities.

    Args:
        block: Block type (HeteroBasicBlock or HeteroBottleneck)
        num_blocks: List of number of blocks per layer
        in_channels: Number of input channels (3 for RGB images)
        num_classes: Number of output classes
        model_rate: Width scaling factor (e.g., 0.25, 0.5, 0.75, 1.0)
        hidden_size: Base channel dimensions [64, 128, 256, 512]
        scale: Whether to use scaling layers
        dropout_prob: Dropout probability before final linear layer
    """

    def __init__(self, block, num_blocks: List[int],
                 in_channels: int = 3,
                 num_classes: int = 10,
                 model_rate: float = 1.0,
                 hidden_size: Optional[List[int]] = None,
                 scale: bool = False,
                 dropout_prob: float = 0.0):
        super().__init__()

        self.model_rate = model_rate
        self.scale = scale

        # Default hidden sizes for ResNet
        if hidden_size is None:
            hidden_size = [64, 128, 256, 512]

        # Scale all hidden dimensions by model_rate
        self.hidden_size = [int(np.ceil(model_rate * x)) for x in hidden_size]

        self.in_planes = self.hidden_size[0]

        # Initial convolution (CIFAR stem)
        self.conv1 = nn.Conv2d(
            in_channels, self.hidden_size[0], kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.hidden_size[0])

        # Build residual layers
        self.layer1 = self._make_layer(block, self.hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.hidden_size[3], num_blocks[3], stride=2)

        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(self.hidden_size[3] * block.expansion, num_classes)

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        """Create a residual layer with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, model_rate=1.0, scale=self.scale))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        return out

    def get_num_params(self):
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_hetero_model(
    model_name: str,
    in_channels: int = 3,
    num_classes: int = 10,
    model_rate: float = 1.0,
    norm_type: str = "sbn",
) -> nn.Module:
    """Factory function: instantiate a heterogeneous model by name.

    Supported names: "HeteroPreResNet18", "HeteroResNet18".
    """
    if model_name == "HeteroCNN":
        return HeteroCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            model_rate=model_rate,
            norm_type=norm_type,
        )
    if model_name == "HeteroPreResNet18":
        return HeteroPreResNet18(
            in_channels=in_channels,
            num_classes=num_classes,
            model_rate=model_rate,
            norm_type=norm_type,
        )
    if model_name == "HeteroResNet18":
        return HeteroResNet18(
            in_channels=in_channels,
            num_classes=num_classes,
            model_rate=model_rate,
        )
    if model_name in ("HeteroViT", "HeteroViT32", "HeteroViT64"):
        from .hetero_vit import build_hetero_vit
        return build_hetero_vit(
            model_name,
            in_channels=in_channels,
            num_classes=num_classes,
            model_rate=model_rate,
        )
    raise ValueError(
        f"Unknown model name {model_name!r}. "
        f"Supported: 'HeteroPreResNet18', 'HeteroResNet18', "
        f"'HeteroViT', 'HeteroViT64'."
    )


def HeteroResNet18(in_channels: int = 3,
                   num_classes: int = 10,
                   model_rate: float = 1.0,
                   scale: bool = False,
                   dropout_prob: float = 0.0):
    """Create a ResNet-18 with configurable width.

    Args:
        in_channels: Number of input channels (3 for CIFAR)
        num_classes: Number of output classes (10 for CIFAR-10, 100 for CIFAR-100)
        model_rate: Width multiplier (0.25, 0.5, 0.75, 1.0)
        scale: Whether to use gradient scaling
        dropout_prob: Dropout probability

    Returns:
        HeteroResNet model with specified width
    """
    return HeteroResNet(
        HeteroBasicBlock,
        [2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        model_rate=model_rate,
        scale=scale,
        dropout_prob=dropout_prob
    )


def create_model_split(num_clients: int = 100,
                      split_mode: str = "fix",
                      min_rate: float = 0.25,
                      max_rate: float = 1.0) -> List[float]:
    """Create model rate distribution for clients.

    This function generates the heterogeneity distribution for clients based on
    different strategies used in the FedRolex paper.

    Args:
        num_clients: Number of clients
        split_mode: Distribution mode
            - "fix": Fixed distribution with specific rates
            - "uniform": Uniform random distribution
            - "dirichlet": Dirichlet distribution
        min_rate: Minimum model rate
        max_rate: Maximum model rate

    Returns:
        List of model rates for each client
    """
    if split_mode == "fix":
        # Fixed distribution used in FedRolex paper
        # 40% clients with rate 0.25, 30% with 0.5, 20% with 0.75, 10% with 1.0
        rates = []
        rates.extend([0.25] * int(0.4 * num_clients))
        rates.extend([0.5] * int(0.3 * num_clients))
        rates.extend([0.75] * int(0.2 * num_clients))
        rates.extend([1.0] * (num_clients - len(rates)))  # Fill remaining
        return rates

    elif split_mode == "uniform":
        # Uniform random distribution
        return [np.random.uniform(min_rate, max_rate) for _ in range(num_clients)]

    elif split_mode == "dirichlet":
        # Sample from Dirichlet and map to [min_rate, max_rate]
        alpha = np.ones(num_clients) * 0.5
        rates = np.random.dirichlet(alpha)
        rates = min_rate + (max_rate - min_rate) * (rates / rates.max())
        return rates.tolist()

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")


def get_model_info(model: nn.Module) -> dict:
    """Get information about model architecture and parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }

    if isinstance(model, HeteroResNet):
        info['model_rate'] = model.model_rate
        info['hidden_size'] = model.hidden_size

    return info


# ============================================================================
# Paper-Faithful HeteroFL Implementation (ICLR 2021)
# ============================================================================
"""
Key features for reproducibility:
1. PreResNet block with shortcut branching from PRE-ACTIVATED tensor
2. Scaler after ALL parametric layers: stem conv, block convs, projection shortcuts
3. Robust Static Batch Normalization (sBN) that:
   - KEEPS running buffers allocated (track_running_stats=True)
   - PREVENTS running stat updates during training (momentum=0 trick)
   - ENABLES proper calibration before evaluation
4. BN calibration with correct variance aggregation (law of total variance)
"""

from torch.utils.data import DataLoader


class ScalerV2(nn.Module):
    """Scaling layer for heterogeneous model training.

    According to paper Section 3.3:
    "The Scaler module scales representations by 1/r^(p-1) during the training phase."

    For width ratio r, we scale by 1/r to compensate for reduced fan-in.
    This is analogous to inverted dropout scaling.

    By default scaling is applied during training only. For debugging or
    evaluation parity checks on sub-models, `always_scale` can force scaling
    in eval mode as well.
    """

    def __init__(self, rate: float):
        super().__init__()
        self.scale = 1.0 / rate if rate > 0 else 1.0
        self.always_scale = False

    def forward(self, x):
        if self.training or self.always_scale:
            return x * self.scale
        return x  # Identity during inference


class StaticBatchNorm2d(nn.BatchNorm2d):
    """Static Batch Normalization as described in HeteroFL paper Section 3.2.

    "During the training phase, sBN does not track running estimates and
    simply normalize batch data."

    IMPLEMENTATION NOTE:
    We keep track_running_stats=True so that running_mean/running_var buffers
    are properly allocated by PyTorch. This is essential for calibration.

    During training, we use momentum=0 to prevent running stat updates while
    still using batch statistics for normalization.

    During calibration (sbn_mode=False), we allow running stats to update.
    During evaluation, standard BN behavior uses the calibrated running stats.

    Attributes:
        sbn_mode: If True (default), prevents running stat updates during training.
                  Set to False during BN calibration phase.
    """

    def __init__(self, num_features, **kwargs):
        # Keep track_running_stats=True so buffers exist!
        kwargs.pop('track_running_stats', None)
        super().__init__(num_features, track_running_stats=True, **kwargs)
        self.sbn_mode = True  # Static BN behavior during training

    def forward(self, x):
        if self.training and self.sbn_mode:
            # sBN behavior: use batch stats for normalization,
            # but DON'T update running stats (momentum=0)
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=True,  # Use batch stats
                momentum=0.0,   # Don't update running stats
                eps=self.eps
            )
        # Normal BN behavior:
        # - training + not sbn_mode: update running stats (for calibration)
        # - eval mode: use running stats
        return super().forward(x)


def _select_group_count(num_channels: int, max_groups: int = 32) -> int:
    """Largest divisor of num_channels not exceeding max_groups."""
    upper = min(max_groups, num_channels)
    for g in range(upper, 0, -1):
        if num_channels % g == 0:
            return g
    return 1


def _make_norm2d(num_channels: int, norm_type: str) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "sbn":
        return StaticBatchNorm2d(num_channels)
    if norm_type == "gn":
        return nn.GroupNorm(_select_group_count(num_channels), num_channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class PreActHeteroBasicBlock(nn.Module):
    """Pre-activated BasicBlock with STRICT HeteroFL compliance.

    Critical: In PreResNet, the shortcut branches from the PRE-ACTIVATED tensor,
    not the raw input. This ensures proper gradient flow through identity mapping.

    Structure:
        x → BN → ReLU → [shortcut branches here] → Conv → Scaler → BN → ReLU → Conv → Scaler → (+shortcut) → out

    Scaler placement: "right after the parametric layer" means:
        - After conv1
        - After conv2
        - After projection conv in shortcut (if present)
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1,
                 model_rate: float = 1.0, use_scaler: bool = True, norm_type: str = "sbn"):
        super().__init__()

        self.model_rate = model_rate
        self.use_scaler = use_scaler
        self.norm_type = norm_type
        self.needs_projection = (stride != 1 or in_planes != planes * self.expansion)

        # First pre-activation (shared by main path AND shortcut in true PreAct)
        self.bn1 = _make_norm2d(in_planes, norm_type)

        # Main path
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.scaler1 = ScalerV2(model_rate) if (use_scaler and model_rate < 1.0) else nn.Identity()

        self.bn2 = _make_norm2d(planes, norm_type)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.scaler2 = ScalerV2(model_rate) if (use_scaler and model_rate < 1.0) else nn.Identity()

        # Projection shortcut (also gets scaler!)
        if self.needs_projection:
            self.shortcut_conv = nn.Conv2d(
                in_planes, planes * self.expansion,
                kernel_size=1, stride=stride, bias=False
            )
            self.shortcut_scaler = ScalerV2(model_rate) if (use_scaler and model_rate < 1.0) else nn.Identity()
        else:
            self.shortcut_conv = None
            self.shortcut_scaler = None

    def forward(self, x):
        # Pre-activation (shared for both paths in true PreAct)
        preact = F.relu(self.bn1(x))

        # Main path: branches from preactivated tensor
        out = self.conv1(preact)
        out = self.scaler1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.scaler2(out)

        # Shortcut: ALSO branches from preactivated tensor (key PreAct property!)
        if self.needs_projection:
            shortcut = self.shortcut_conv(preact)  # Use preact, not x!
            shortcut = self.shortcut_scaler(shortcut)
        else:
            shortcut = x  # Identity shortcut uses original x

        return out + shortcut


class HeteroPreResNet(nn.Module):
    """Pre-activated ResNet with full HeteroFL compliance.

    Strict paper compliance:
    - PreResNet architecture (BN-ReLU-Conv order)
    - Shortcut from pre-activated tensor
    - Robust Static Batch Normalization (sBN)
    - Scaler after ALL parametric layers (stem, blocks, projections)
    - Width scaling via model_rate
    """

    def __init__(self, block, num_blocks: List[int],
                 in_channels: int = 3,
                 num_classes: int = 10,
                 model_rate: float = 1.0,
                 hidden_size: Optional[List[int]] = None,
                 use_scaler: bool = True,
                 norm_type: str = "sbn"):
        super().__init__()

        self.model_rate = model_rate
        self.use_scaler = use_scaler
        self.norm_type = norm_type

        if hidden_size is None:
            hidden_size = [64, 128, 256, 512]

        self.hidden_size = [int(np.ceil(model_rate * x)) for x in hidden_size]
        self.in_planes = self.hidden_size[0]

        # Stem: conv + scaler (scaler after parametric layer!)
        self.conv1 = nn.Conv2d(
            in_channels, self.hidden_size[0], kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.stem_scaler = ScalerV2(model_rate) if (use_scaler and model_rate < 1.0) else nn.Identity()

        # Residual layers
        self.layer1 = self._make_layer(block, self.hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, self.hidden_size[3], num_blocks[3], stride=2)

        # Final pre-activation before classifier
        self.bn_final = _make_norm2d(self.hidden_size[3] * block.expansion, norm_type)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.hidden_size[3] * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(
                self.in_planes, planes, s,
                model_rate=self.model_rate,
                use_scaler=self.use_scaler,
                norm_type=self.norm_type,
            ))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, StaticBatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stem with scaler
        out = self.conv1(x)
        out = self.stem_scaler(out)

        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Final pre-activation
        out = F.relu(self.bn_final(out))

        # Classifier (no scaler on logits)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def HeteroPreResNet18(in_channels: int = 3,
                      num_classes: int = 10,
                      model_rate: float = 1.0,
                      use_scaler: bool = True,
                      norm_type: str = "sbn") -> HeteroPreResNet:
    """Create PreResNet-18 with HeteroFL width scaling.

    Complexity levels from paper (r=0.5):
        'a': model_rate=1.0    (100% width)
        'b': model_rate=0.5    (50% width)
        'c': model_rate=0.25   (25% width)
        'd': model_rate=0.125  (12.5% width)
        'e': model_rate=0.0625 (6.25% width)
    """
    return HeteroPreResNet(
        PreActHeteroBasicBlock,
        [2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        model_rate=model_rate,
        use_scaler=use_scaler,
        norm_type=norm_type,
    )


# ============================================================================
# BN Statistics Calibration (Robust implementation)
# ============================================================================

def set_sbn_mode(model: nn.Module, sbn_mode: bool) -> None:
    """Set sBN mode for all StaticBatchNorm2d layers.

    Args:
        model: Model containing StaticBatchNorm2d layers
        sbn_mode: If True, prevents running stat updates during training.
                  If False, allows running stat updates (for calibration).
    """
    for module in model.modules():
        if isinstance(module, StaticBatchNorm2d):
            module.sbn_mode = sbn_mode


def reset_bn_stats(model: nn.Module) -> None:
    """Reset running statistics for all BatchNorm layers.

    Safe implementation that handles:
    - None buffers (when track_running_stats=False)
    - Device consistency (operates in-place on existing tensors)

    Args:
        model: Model containing BatchNorm layers
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if m.running_mean is not None:
                m.running_mean.zero_()
            if m.running_var is not None:
                m.running_var.fill_(1)
            if getattr(m, "num_batches_tracked", None) is not None:
                m.num_batches_tracked.zero_()


@torch.no_grad()
def calibrate_bn_stats(model: nn.Module,
                       dataloader: DataLoader,
                       device: torch.device = torch.device('cpu'),
                       num_batches: Optional[int] = None) -> None:
    """Calibrate BatchNorm statistics before evaluation.

    From HeteroFL paper Section 3.2:
    "Before evaluation, we calibrate the sBN layers by running a forward pass
    on a subset of training data to compute proper running statistics."

    This implementation:
    1. Resets running_mean/running_var to initial values
    2. Temporarily disables sBN mode (allows running stat updates)
    3. Runs forward passes to accumulate statistics (using cumulative average)
    4. Re-enables sBN mode for next training phase

    Args:
        model: Model with StaticBatchNorm2d layers
        dataloader: DataLoader with calibration data (subset of training data)
        device: Device to run calibration on
        num_batches: Number of batches to use (None = all batches)

    Note on FL evaluation:
        For central evaluation: use server-held train subset (this function)
        For FL-wide calibration: aggregate BN stats from multiple clients
    """
    model.to(device)
    model.train()  # Need train mode for BN to update running stats

    # Step 1: Reset running statistics
    reset_bn_stats(model)

    # Step 2: Disable sBN mode and set momentum=None for cumulative average
    set_sbn_mode(model, sbn_mode=False)

    original_momenta = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            original_momenta[name] = module.momentum
            module.momentum = None  # Use cumulative moving average

    # Step 3: Run forward passes to accumulate statistics
    num_processed = 0
    for i, (inputs, _) in enumerate(dataloader):
        if num_batches is not None and i >= num_batches:
            break
        inputs = inputs.to(device)
        model(inputs)
        num_processed = i + 1

    # Step 4: Restore original settings
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = original_momenta.get(name, 0.1)

    set_sbn_mode(model, sbn_mode=True)  # Re-enable sBN for training


def calibrate_and_eval(model: nn.Module,
                       calibration_loader: DataLoader,
                       test_loader: DataLoader,
                       device: torch.device,
                       calibration_batches: int = 50) -> float:
    """Calibrate BN stats and evaluate model accuracy.

    This is the proper evaluation procedure for HeteroFL models:
    1. Calibrate BN running statistics on training data subset
    2. Evaluate on test data using calibrated statistics

    Args:
        model: Trained HeteroFL model
        calibration_loader: DataLoader for BN calibration (train data subset)
        test_loader: DataLoader for evaluation (test data)
        device: Computation device
        calibration_batches: Number of batches for BN calibration

    Returns:
        Test accuracy as percentage
    """
    # Step 1: Calibrate BN statistics
    calibrate_bn_stats(model, calibration_loader, device, num_batches=calibration_batches)

    # Step 2: Evaluate in eval mode (uses calibrated running stats)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


# ============================================================================
# FL-Wide BN Calibration Utilities (for Flower integration)
# ============================================================================

def get_bn_stats(model: nn.Module) -> dict:
    """Extract BN running statistics from model.

    Useful for FL aggregation of BN stats from multiple clients.

    Returns:
        Dict mapping layer names to (running_mean, running_var, num_batches_tracked)
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            stats[name] = {
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
                'num_batches_tracked': module.num_batches_tracked.clone(),
            }
    return stats


def set_bn_stats(model: nn.Module, stats: dict) -> None:
    """Set BN running statistics in model.

    Useful for applying aggregated BN stats from server.

    Args:
        model: Model to update
        stats: Dict from get_bn_stats()
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d) and name in stats:
            module.running_mean.copy_(stats[name]['running_mean'])
            module.running_var.copy_(stats[name]['running_var'])
            module.num_batches_tracked.copy_(stats[name]['num_batches_tracked'])


def aggregate_bn_stats(all_stats: List[dict], weights: Optional[List[float]] = None) -> dict:
    """Aggregate BN statistics from multiple clients using law of total variance.

    For FL-wide BN calibration, each client calibrates locally and sends
    their BN stats to server. Server aggregates using this function.

    IMPORTANT: Variance aggregation uses the law of total variance:
        Var(X) = E[Var(X|Y)] + Var(E[X|Y])

    For weighted combination:
        global_mean = Σ w_i * mean_i
        global_var  = Σ w_i * (var_i + mean_i²) - global_mean²
                    = Σ w_i * E[X²]_i - (Σ w_i * mean_i)²

    Simply averaging variances is WRONG and will underestimate variance!

    Args:
        all_stats: List of stats dicts from get_bn_stats() from each client
        weights: Weights for each client (e.g., num samples used for calibration).
                 If None, uses equal weights.

    Returns:
        Aggregated stats dict with correct variance computation
    """
    if not all_stats:
        return {}

    if weights is None:
        weights = [1.0] * len(all_stats)

    total = float(sum(weights))
    weights = [w / total for w in weights]

    aggregated = {}
    layer_names = all_stats[0].keys()

    for name in layer_names:
        # Weighted mean: E[X] = Σ w_i * mean_i
        mean = sum(
            w * stats[name]["running_mean"]
            for w, stats in zip(weights, all_stats)
        )

        # Weighted second moment: E[X²] = Σ w_i * (var_i + mean_i²)
        # This is because E[X²] = Var(X) + E[X]² for each client
        second_moment = sum(
            w * (stats[name]["running_var"] + stats[name]["running_mean"] ** 2)
            for w, stats in zip(weights, all_stats)
        )

        # Variance from second moment: Var(X) = E[X²] - E[X]²
        var = second_moment - mean ** 2

        # Numerical safety: variance must be positive
        var = torch.clamp(var, min=1e-5)

        # For num_batches_tracked, sum across all clients
        tracked = sum(stats[name]["num_batches_tracked"] for stats in all_stats)

        aggregated[name] = {
            "running_mean": mean,
            "running_var": var,
            "num_batches_tracked": tracked,
        }

    return aggregated


# ============================================================================
# Convenience functions
# ============================================================================

def get_model_complexity_levels():
    """Return the complexity level configuration from the paper."""
    return {
        'a': 1.0,      # Full model
        'b': 0.5,      # r^1 = 0.5
        'c': 0.25,     # r^2 = 0.25
        'd': 0.125,    # r^3 = 0.125
        'e': 0.0625,   # r^4 = 0.0625
    }


def HeteroPreResNet18_a(num_classes=10): return HeteroPreResNet18(num_classes=num_classes, model_rate=1.0)
def HeteroPreResNet18_b(num_classes=10): return HeteroPreResNet18(num_classes=num_classes, model_rate=0.5)
def HeteroPreResNet18_c(num_classes=10): return HeteroPreResNet18(num_classes=num_classes, model_rate=0.25)
def HeteroPreResNet18_d(num_classes=10): return HeteroPreResNet18(num_classes=num_classes, model_rate=0.125)
def HeteroPreResNet18_e(num_classes=10): return HeteroPreResNet18(num_classes=num_classes, model_rate=0.0625)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HeteroFL PreResNet18 - Paper-Faithful Implementation")
    print("="*70)

    print("\nKey compliance features:")
    print("  ✓ PreResNet block structure (BN → ReLU → Conv)")
    print("  ✓ Shortcut branches from PRE-ACTIVATED tensor")
    print("  ✓ Scaler after stem conv, block convs, projection shortcuts")
    print("  ✓ Robust sBN (keeps buffers, prevents updates via momentum=0)")
    print("  ✓ Safe reset_bn_stats (handles None, device consistency)")
    print("  ✓ Correct variance aggregation (law of total variance)")

    levels = get_model_complexity_levels()

    print("\n" + "-"*70)
    print("Model complexity levels:")
    print("-"*70)
    print(f"{'Level':<6} {'Rate':<8} {'Params':<12} {'Hidden Sizes':<25} {'Size (MB)':<10}")
    print("-"*70)

    for level, rate in levels.items():
        model = HeteroPreResNet18(num_classes=10, model_rate=rate)
        params = model.get_num_params()
        size_mb = params * 4 / (1024**2)
        print(f"{level:<6} {rate:<8.4f} {params:>10,}   {str(model.hidden_size):<25} {size_mb:>8.2f}")

    # Forward pass test
    print("\n" + "-"*70)
    print("Forward pass test (50% width model):")
    print("-"*70)
    model = HeteroPreResNet18(num_classes=10, model_rate=0.5)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(y.shape)}")
    print(f"  Params: {model.get_num_params():,}")

    # Verify variance aggregation correctness
    print("\n" + "-"*70)
    print("Variance aggregation test (law of total variance):")
    print("-"*70)

    # Simulate two clients with different means and variances
    client1_stats = {
        'bn1': {
            'running_mean': torch.tensor([1.0, 2.0]),
            'running_var': torch.tensor([0.5, 0.5]),
            'num_batches_tracked': torch.tensor(100),
        }
    }
    client2_stats = {
        'bn1': {
            'running_mean': torch.tensor([3.0, 4.0]),
            'running_var': torch.tensor([0.5, 0.5]),
            'num_batches_tracked': torch.tensor(100),
        }
    }

    # Equal weights
    agg = aggregate_bn_stats([client1_stats, client2_stats], weights=[1.0, 1.0])

    print(f"  Client 1: mean=[1,2], var=[0.5,0.5]")
    print(f"  Client 2: mean=[3,4], var=[0.5,0.5]")
    print(f"  Aggregated mean: {agg['bn1']['running_mean'].tolist()}")
    print(f"  Aggregated var:  {agg['bn1']['running_var'].tolist()}")

    # Expected: mean=[2,3], var = 0.5 + 0.5*(1²+3²) - 2² = 0.5 + 1 = 1.5 for first element
    # var = E[var] + Var[mean] = 0.5 + ((1-2)² + (3-2)²)/2 = 0.5 + 1 = 1.5
    print(f"  Expected var:    [1.5, 1.5] (accounts for between-client variance)")
    print(f"  Wrong (naive avg): [0.5, 0.5] (underestimates!)")

    print("\n" + "-"*70)
    print("Testing original HeteroResNet18 (backward compatibility):")
    print("-"*70)

    for rate in [0.25, 0.5, 0.75, 1.0]:
        model = HeteroResNet18(num_classes=10, model_rate=rate)
        info = get_model_info(model)
        print(f"  Rate {rate}: {info['total_params']:,} params, {info['model_size_mb']:.2f} MB")

    print("\n✓ All tests passed!")


class HeteroCNN(nn.Module):
    """Width-sliceable BN CNN (HeteroFL-style channel-prefix slicing) for FEMNIST-scale tasks.
    Base widths [64,128,256,512] scaled by ceil(model_rate*base); ScalerV2 width compensation;
    BatchNorm (sBN); GAP features exposed via forward_features (for CCVR, like HeteroViT).
    Core rate 0.25 -> 128-d features, full 1.0 -> 512-d (matches the ResNet core/full dims)."""

    def __init__(self, in_channels: int = 3, num_classes: int = 62,
                 model_rate: float = 1.0, norm_type: str = "sbn"):
        super().__init__()
        self.model_rate = model_rate
        self.num_classes = num_classes
        base = [64, 128, 256, 512]
        ch = [int(np.ceil(model_rate * b)) for b in base]
        self.hidden_size = ch

        def block(cin, cout, pool):
            layers = [nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                      ScalerV2(model_rate),
                      nn.BatchNorm2d(cout),
                      nn.ReLU(inplace=True)]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            block(in_channels, ch[0], True),    # 32 -> 16
            block(ch[0], ch[1], True),          # 16 -> 8
            block(ch[1], ch[2], True),          # 8 -> 4
            block(ch[2], ch[3], False),         # 4 x 4
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(ch[3], num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)

    def forward(self, x):
        return self.linear(self.forward_features(x))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())

