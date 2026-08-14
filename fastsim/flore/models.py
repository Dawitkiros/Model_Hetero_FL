
"""Implement the neural network models and training functions."""
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
from easydict import EasyDict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim.sgd import SGD
from torch.optim.optimizer import Optimizer
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
import torchvision.models as M


def make_resnet(model_name="resnet34", num_classes=200, pretrained=True):
    if model_name == "resnet34":
        model = M.resnet34(weights=M.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_name == "resnet50":
        model = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    else:
        raise ValueError

    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

#ResNet9
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(
            self, 
            input_dim,
            hidden_dims, 
            num_classes):
        super(ResNet9, self).__init__()
        
        self.conv1 = conv_block(input_dim, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d((1,1)), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

import torch.nn as nn

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, 
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, planes * self.expansion, 
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, num_classes=100, dropout_prob=0.0):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # "CIFAR stem": 3×3 conv, stride=1
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optional dropout before final FC
        self.dropout = nn.Dropout(p=dropout_prob)  
        
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
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
        out = self.dropout(out)         # only here if desired
        out = self.linear(out)
        return out

def ResNet18(in_channels=3, num_classes=100, dropout_prob=0.0):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  in_channels=in_channels,
                  num_classes=num_classes,
                  dropout_prob=dropout_prob)

class CNN(nn.Module):
    """Implement a CNN model for CIFAR-10.

    Parameters
    ----------
    input_dim : int
        The input dimension for classifier.
    hidden_dims : List[int]
        The hidden dimensions for classifier.
    num_classes : int
        The number of classes in the dataset.
    """

    def __init__(self, input_dim, hidden_dims, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        """Implement forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNN500k(nn.Module):
    def __init__(self, num_channels, image_size, num_classes):
        super().__init__()
        
        self.layer_stack = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        
            nn.Flatten(),
            nn.Linear(32 * int(image_size/8) * int(image_size/8), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)


class CNNMnist(nn.Module):
    """Implement a CNN model for MNIST and Fashion-MNIST.

    Parameters
    ----------
    input_dim : int
        The input dimension for classifier.
    hidden_dims : List[int]
        The hidden dimensions for classifier.
    num_classes : int
        The number of classes in the dataset.
    """

    def __init__(self, input_dim, hidden_dims, num_classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)

    def forward(self, x):
        """Implement forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 4 * 4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"""SCAFFOLD training loop - Fixed and improved."""

import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from typing import List


class ScaffoldOptimizer(SGD):
    """SGD optimizer with SCAFFOLD control variate correction."""

    def __init__(self, params, lr, momentum=0, weight_decay=0):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        
    def step_scaffold(self, server_cv: List[torch.Tensor], client_cv: List[torch.Tensor], closure=None):
        """Perform SCAFFOLD optimization step.
        
        Args:
            server_cv: Server control variate (must match parameters order)
            client_cv: Client control variate (must match parameters order)
            closure: Optional closure
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Step 1: Standard SGD step
        super().step()
        
        # Step 2: Apply SCAFFOLD correction
        # Flatten all parameters across all groups into a single list
        all_params = []
        for group in self.param_groups:
            all_params.extend(group['params'])
        
        lr = self.param_groups[0]['lr']  # Assume same LR for all groups
        
        # Apply correction to each parameter
        for param_idx, param in enumerate(all_params):
            if param.grad is None:
                continue
            
            # Get corresponding control variates
            s_cv = server_cv[param_idx]
            c_cv = client_cv[param_idx]
            
            # Move to parameter's device if needed
            if s_cv.device != param.device:
                s_cv = s_cv.to(param.device)
            if c_cv.device != param.device:
                c_cv = c_cv.to(param.device)
            
            # Verify shapes match
            if s_cv.shape != param.shape:
                raise RuntimeError(
                    f"Control variate shape mismatch at index {param_idx}: "
                    f"param shape={param.shape}, s_cv shape={s_cv.shape}, c_cv shape={c_cv.shape}"
                )
            
            # Apply correction: y = y - η*(c - c_i)
            param.data.add_(s_cv - c_cv, alpha=-lr)
        
        return loss


def train_scaffold(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    server_cv: List[torch.Tensor],
    client_cv: List[torch.Tensor],
) -> dict:
    """Train the network using SCAFFOLD algorithm.
    
    Args:
        net: Neural network to train
        trainloader: Training data loader
        device: Device to train on
        epochs: Number of training epochs
        learning_rate: Learning rate (η_l)
        momentum: SGD momentum parameter
        weight_decay: Weight decay for regularization
        server_cv: Server control variate c^t (list of tensors)
        client_cv: Client control variate c_i^t (list of tensors)
        
    Returns:
        Dictionary with training metrics
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = ScaffoldOptimizer(
        net.parameters(), 
        lr=learning_rate, 
        momentum=momentum,   # type: ignore
        weight_decay=weight_decay # type: ignore
    )
    
    net.train()
    
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss, epoch_samples = _train_one_epoch_scaffold(
            net=net,
            trainloader=trainloader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            server_cv=server_cv,
            client_cv=client_cv,
        )
        total_loss += epoch_loss
        total_samples += epoch_samples
    
    avg_loss = total_loss / epochs
    
    return {
        "loss": avg_loss,
        "num_samples": total_samples,
        "num_batches": len(trainloader),
    }


def _train_one_epoch_scaffold(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: ScaffoldOptimizer,
    server_cv: List[torch.Tensor],
    client_cv: List[torch.Tensor],
) -> tuple:
    """Train the network for one epoch with SCAFFOLD.
    
    Args:
        net: Neural network
        trainloader: Training data loader
        device: Device to train on
        criterion: Loss function
        optimizer: SCAFFOLD optimizer
        server_cv: Server control variate
        client_cv: Client control variate
        
    Returns:
        Tuple of (epoch_loss, num_samples)
    """
    epoch_loss = 0.0
    num_samples = 0
    
    for batch_idx, (data, target) in enumerate(trainloader):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = net(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # SCAFFOLD optimizer step with control variates
        optimizer.step_scaffold(server_cv, client_cv)
        
        # Track metrics
        epoch_loss += loss.item() * data.size(0)
        num_samples += data.size(0)
    
    return epoch_loss / num_samples, num_samples


# Alternative: Cleaner functional implementation
def train_scaffold_functional(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    server_cv: List[torch.Tensor],
    client_cv: List[torch.Tensor],
) -> dict:
    """Functional implementation without custom optimizer class.
    
    This is simpler and more transparent about what's happening.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    net.train()
    total_loss = 0.0
    total_samples = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0
        
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Standard SGD step: y = y - η*∇f(y)
            optimizer.step()
            
            # SCAFFOLD correction: y = y - η*(c - c_i)
            with torch.no_grad():
                param_idx = 0
                for param in net.parameters():
                    if param.grad is None:
                        continue
                    
                    s_cv = server_cv[param_idx].to(param.device)
                    c_cv = client_cv[param_idx].to(param.device)
                    
                    param.data.add_(s_cv - c_cv, alpha=-learning_rate)
                    param_idx += 1
            
            epoch_loss += loss.item() * data.size(0)
            epoch_samples += data.size(0)
        
        total_loss += epoch_loss / epoch_samples
        total_samples = epoch_samples
    
    return {
        "loss": total_loss / epochs,
        "num_samples": total_samples,
        "num_batches": len(trainloader),
    }

from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

import torch
import torch.nn as nn
from typing import Tuple

def _bn_train_dropout_eval(net: nn.Module):
    # Use batch stats for BN (train), but disable dropout noise.
    for m in net.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.eval()
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

def train_fedavg_plain(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> tuple[int, int, float]:
    """FedAvg-style local training without AMP, for debugging/comparison."""
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    net.train()

    total_steps = 0
    total_samples = 0
    running_loss = 0.0

    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = net(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            bs = data.size(0)
            total_steps += 1
            total_samples += bs
            running_loss += loss.item() * bs

    avg_loss = running_loss / max(total_samples, 1)
    return total_steps, total_samples, avg_loss

def train_fedavg_num_steps(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    num_steps: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    proximal_mu: float = 0.0,
) -> None:
    """Train the network using FedAvg/FedProx for a fixed number of SGD steps."""
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()

    # For FedProx: snapshot of global params
    global_params = [p.detach().clone() for p in net.parameters()]

    scaler = GradScaler(device="cuda" if device.type == "cuda" else "cpu")

    data_iter = iter(trainloader)
    steps_done = 0

    while steps_done < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # recycle the dataloader if we ran out of batches
            data_iter = iter(trainloader)
            batch = next(data_iter)

        # Support both dict-style and tuple-style batches
        if isinstance(batch, dict):
            x = batch.get("image") or batch.get("x") or batch[0]
            y = batch.get("label") or batch.get("y") or batch[1]
        else:
            x, y = batch

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = net(x)
            loss = criterion(outputs, y)

            if proximal_mu > 0.0:
                prox_term = 0.0
                for p, p0 in zip(net.parameters(), global_params):
                    prox_term = prox_term + ((p - p0) ** 2).sum()
                loss = loss + 0.5 * proximal_mu * prox_term

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        steps_done += 1


# def train_fedavg(
#     net: nn.Module,
#     trainloader: DataLoader,
#     device: torch.device,
#     epochs: int,
#     learning_rate: float,
#     momentum: float,
#     weight_decay: float,
#     proximal_mu: float = 0.0,
# ) -> None:
#     """Backward-compatible wrapper: epochs × len(trainloader) steps."""
#     num_steps = epochs * len(trainloader)
#     train_fedavg_num_steps(
#         net=net,
#         trainloader=trainloader,
#         device=device,
#         num_steps=num_steps,
#         learning_rate=learning_rate,
#         momentum=momentum,
#         weight_decay=weight_decay,
#         proximal_mu=proximal_mu,
#     )

def train_fedavg(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    shell_mask: Optional[Dict[str, torch.Tensor]] = None,
) -> None:
    # pylint: disable=too-many-arguments
    """Train the network on the training set using FedAvg.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        The number of epochs to train the network.
    learning_rate : float
        The learning rate.
    momentum : float
        The momentum for SGD optimizer.
    weight_decay : float
        The weight decay for SGD optimizer.

    Returns
    -------
    None
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    scaler = GradScaler(device='cuda')
    for _ in range(epochs):
        net = _train_one_epoch(net, trainloader, device, criterion, optimizer, scaler,
                               shell_mask=shell_mask)

def _train_one_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    shell_mask: Optional[Dict[str, torch.Tensor]] = None,
) -> nn.Module:
    """Train the network on the training set for one epoch with AMP."""
    net.train()
    running_loss = 0.0

    for data, target in trainloader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        with autocast(device_type='cuda'):  # AMP context
            output = net(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        if shell_mask is not None:
            for name, p in net.named_parameters():
                if p.grad is not None and name in shell_mask:
                    p.grad.mul_(shell_mask[name])
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return net


def set_lr_by_round(optimizer, base_lr, round_num, total_rounds, min_lr=1e-6):
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * round_num / total_rounds))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_fedavg_adam(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    global_round: int = 0,
    lr: float   = 3e-4,
    weight_decay: float = 0.05,
    total_rounds: int = 150,
) -> Dict[str, float]:
    """
    Local training with AdamW + warm-up + cosine.
    Called once per client before each global round.
    """

    # ------ optimiser -------------------------------------------------------
    # (optionally exclude biases & LayerNorm from weight decay)
    decay, no_decay = [], []
    for n, p in net.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or n.endswith('.bias') or n.endswith('.weight_g') or \
   'pos_embed' in n or 'cls_token' in n else decay).append(p)

    param_groups = [
        {"params": decay,     "weight_decay": weight_decay},
        {"params": no_decay,  "weight_decay": 0.0},
    ]

    import math
    warmup = max(1, int(0.05 * total_rounds))
    if global_round < warmup:
        cur_lr = lr * float(global_round + 1) / warmup
    else:
        prog = (global_round - warmup) / max(1, total_rounds - warmup)
        cur_lr = 0.05 * lr + 0.5 * (lr - 0.05 * lr) * (1 + math.cos(math.pi * prog))
    optimizer = AdamW(param_groups, lr=cur_lr, betas=(0.9, 0.999))

    # ------ loss + AMP scaffold ---------------------------------------------
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(device='cuda')

    net.train()
    for _ in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                logits = net(data)
                loss   = criterion(logits, target)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
    return net.state_dict()

# class ScaffoldOptimizer(SGD):
#     """Implements SGD optimizer step function as defined in the SCAFFOLD paper."""

#     def __init__(self, params, lr, momentum, weight_decay):
#         super().__init__(
#             params, lr=lr, momentum=momentum, weight_decay=weight_decay
#         )

#     def step_custom(self, server_cv: List[torch.Tensor], client_cv: List[torch.Tensor]):
#         """Implement the custom step function for SCAFFOLD.
        
#         Applies: y_i = y_i - η*(g_i + μ*v_t) - η*(c_global - c_local)
#         where the first part is done by step() and second part here.
#         """
#         # First apply standard SGD step (gradient + momentum)
#         self.step()
        
#         # Then apply control variate correction
#         for group in self.param_groups:
#             for par, s_cv, c_cv in zip(group["params"], server_cv, client_cv):
#                 if par.requires_grad:
#                     # Apply: param = param - lr * (c_global - c_local)
#                     par.data.add_(s_cv - c_cv, alpha=-group["lr"])


# def train_scaffold(
#     net: nn.Module,
#     trainloader: DataLoader,
#     device: torch.device,
#     epochs: int,
#     learning_rate: float,
#     momentum: float,
#     weight_decay: float,
#     c_local: torch.Tensor,
#     c_global: torch.Tensor,
# ) -> Tuple[int, int]:
#     """Train the network using SCAFFOLD with control variates.
    
#     Parameters
#     ----------
#     net : nn.Module
#         The neural network to train.
#     trainloader : DataLoader
#         The training set dataloader object.
#     device : torch.device
#         The device on which to train.
#     epochs : int
#         The number of epochs to train.
#     learning_rate : float
#         The learning rate.
#     momentum : float
#         The momentum for SGD optimizer.
#     weight_decay : float
#         The weight decay for SGD optimizer.
#     c_local : torch.Tensor
#         Local control variate (flat vector).
#     c_global : torch.Tensor
#         Global control variate (flat vector).
    
#     Returns
#     -------
#     Tuple[int, int]
#         (total_steps, total_samples_seen)
#     """
#     criterion = nn.CrossEntropyLoss()
#     optimizer = ScaffoldOptimizer(
#         net.parameters(), 
#         lr=learning_rate, 
#         momentum=momentum, 
#         weight_decay=weight_decay
#     )
#     net.train()
    
#     # Convert flat control variates to per-parameter lists
#     server_cv_list = _flat_to_param_list(net, c_global)
#     client_cv_list = _flat_to_param_list(net, c_local)
    
#     total_steps = 0
#     total_samples = 0
    
#     for epoch in range(epochs):
#         epoch_steps, epoch_samples = _train_one_epoch_scaffold(
#             net, trainloader, device, criterion, optimizer, 
#             server_cv_list, client_cv_list
#         )
#         total_steps += epoch_steps
#         total_samples += epoch_samples
    
#     return total_steps, total_samples


# def _train_one_epoch_scaffold(
#     net: nn.Module,
#     trainloader: DataLoader,
#     device: torch.device,
#     criterion: nn.Module,
#     optimizer: ScaffoldOptimizer,
#     server_cv: List[torch.Tensor],
#     client_cv: List[torch.Tensor],
# ) -> Tuple[int, int]:
#     """Train one epoch with SCAFFOLD control variate corrections."""
#     net.train()
#     steps = 0
#     samples = 0
    
#     for data, target in trainloader:
#         data, target = data.to(device), target.to(device)
        
#         optimizer.zero_grad(set_to_none=True)
        
#         # Forward pass
#         output = net(data)
#         loss = criterion(output, target)
        
#         # Backward pass
#         loss.backward()
        
#         # Apply custom step: SGD with momentum + control variate correction
#         optimizer.step_custom(server_cv, client_cv)
        
#         steps += 1
#         samples += data.size(0)
    
#     return steps, samples


def _flat_to_param_list(net: nn.Module, flat_tensor: torch.Tensor) -> List[torch.Tensor]:
    """Convert flat control variate tensor to list of per-parameter tensors.
    
    Parameters
    ----------
    net : nn.Module
        The neural network.
    flat_tensor : torch.Tensor
        Flat control variate vector.
    
    Returns
    -------
    List[torch.Tensor]
        List of control variate tensors, one per parameter.
    """
    param_list = []
    offset = 0
    
    for param in net.parameters():
        if not param.requires_grad:
            # Add zero tensor for non-trainable params (won't be used)
            param_list.append(torch.zeros_like(param))
            continue
        
        numel = param.numel()
        # Extract slice and reshape to match parameter shape
        param_cv = flat_tensor[offset:offset+numel].view_as(param).clone()
        param_list.append(param_cv)
        offset += numel
    
    return param_list

def train_scaffold(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    local_steps: int,         
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    c_local: torch.Tensor,
    c_global: torch.Tensor,
) -> Tuple[int, int, float]:
    """Train one client with SCAFFOLD correction for a fixed number of steps."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    net.train()

    # Pre-split control variates into parameter-shaped tensors
    server_cv_list = _flat_to_param_list(net, c_global)
    client_cv_list = _flat_to_param_list(net, c_local)

    total_steps = 0
    total_samples = 0
    running_loss = 0.0

    # We'll iterate over the dataloader as many times as needed
    data_iter = iter(trainloader)

    while total_steps < local_steps:
        try:
            data, target = next(data_iter)
        except StopIteration:
            # Restart the iterator when we run out of data
            data_iter = iter(trainloader)
            data, target = next(data_iter)

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()

        # SCAFFOLD control variate correction: grad += c_global - c_local
        for param, s_cv, c_cv in zip(net.parameters(), server_cv_list, client_cv_list):
            if param.requires_grad and param.grad is not None:
                param.grad.data.add_(s_cv - c_cv)

        optimizer.step()

        bs = data.size(0)
        total_steps += 1
        total_samples += bs
        running_loss += loss.item() * bs

    avg_loss = running_loss / max(total_samples, 1)
    return total_steps, total_samples, avg_loss

# def train_scaffold(
#     net: nn.Module,
#     trainloader: DataLoader,
#     device: torch.device,
#     epochs: int,
#     learning_rate: float,
#     momentum: float,
#     weight_decay: float,
#     c_local: torch.Tensor,
#     c_global: torch.Tensor,
# ) -> Tuple[int, int]:
#     criterion = nn.CrossEntropyLoss()
#     # ✅ Use regular SGD optimizer
#     optimizer = torch.optim.SGD(
#         net.parameters(), 
#         lr=learning_rate, 
#         momentum=momentum, 
#         weight_decay=weight_decay
#     )
#     net.train()
    
#     # Convert flat control variates to per-parameter lists
#     server_cv_list = _flat_to_param_list(net, c_global)
#     client_cv_list = _flat_to_param_list(net, c_local)
    
#     total_steps = 0
#     total_samples = 0
    
#     for epoch in range(epochs):
#         for data, target in trainloader:
#             data, target = data.to(device), target.to(device)
            
#             optimizer.zero_grad()
            
#             # Forward + backward
#             output = net(data)
#             loss = criterion(output, target)
#             loss.backward()
            
#             # ✅ Apply control variate correction to gradients BEFORE optimizer step
#             for param, s_cv, c_cv in zip(net.parameters(), server_cv_list, client_cv_list):
#                 if param.requires_grad and param.grad is not None:
#                     # Modify gradient: grad = grad - c_local + c_global
#                     param.grad.data.add_(s_cv - c_cv)
            
#             # Now optimizer applies momentum to CORRECTED gradients
#             optimizer.step()
            
#             total_steps += 1
#             total_samples += data.size(0)
    
#     return total_steps, total_samples



# def train_scaffold(
#     net: nn.Module,
#     trainloader: DataLoader,
#     device: torch.device,
#     epochs: int,
#     learning_rate: float,
#     momentum: float,
#     weight_decay: float,
#     c_local: torch.Tensor,
#     c_global: torch.Tensor,
# ) -> Tuple[int, int]:
#     """Train the network using SCAFFOLD with control variates.
    
#     Parameters
#     ----------
#     net : nn.Module
#         The neural network to train.
#     trainloader : DataLoader
#         The training set dataloader object.
#     device : torch.device
#         The device on which to train.
#     epochs : int
#         The number of epochs to train.
#     learning_rate : float
#         The learning rate.
#     momentum : float
#         The momentum for SGD optimizer.
#     weight_decay : float
#         The weight decay for SGD optimizer.
#     c_local : torch.Tensor
#         Local control variate (flat vector).
#     c_global : torch.Tensor
#         Global control variate (flat vector).
    
#     Returns
#     -------
#     Tuple[int, int]
#         (total_steps, total_samples_seen)
#     """
#     criterion = nn.CrossEntropyLoss()
#     optimizer = SGD(
#         net.parameters(), 
#         lr=learning_rate, 
#         momentum=momentum, 
#         weight_decay=weight_decay
#     )
#     net.train()
    
#     # Compute control variate correction: c_global - c_local
#     c_diff = c_global - c_local
    
#     total_steps = 0
#     total_samples = 0
    
#     for epoch in range(epochs):
#         epoch_steps, epoch_samples = _train_one_epoch_scaffold(
#             net, trainloader, device, criterion, optimizer, c_diff
#         )
#         total_steps += epoch_steps
#         total_samples += epoch_samples
    
#     return total_steps, total_samples


# def _train_one_epoch_scaffold(
#     net: nn.Module,
#     trainloader: DataLoader,
#     device: torch.device,
#     criterion: nn.Module,
#     optimizer: Optimizer,
#     c_diff: torch.Tensor,
# ) -> Tuple[int, int]:
#     """Train one epoch with SCAFFOLD control variate corrections."""
#     net.train()
#     steps = 0
#     samples = 0
    
#     for data, target in trainloader:
#         data, target = data.to(device), target.to(device)
        
#         optimizer.zero_grad(set_to_none=True)
        
#         # Forward pass
#         output = net(data)
#         loss = criterion(output, target)
        
#         # Backward pass
#         loss.backward()
        
#         # Add SCAFFOLD correction: gradient += (c_global - c_local)
#         offset = 0
#         for param in net.parameters():
#             if not param.requires_grad:
#                 continue
#             numel = param.numel()
#             param.grad.add_(c_diff[offset:offset+numel].view_as(param))
#             offset += numel
        
#         optimizer.step()
        
#         steps += 1
#         samples += data.size(0)
    
#     return steps, samples

# def train_scaffold(
#     net: nn.Module,
#     trainloader: DataLoader,
#     device: torch.device,
#     epochs: int,  # Ignored
#     learning_rate: float,
#     momentum: float,
#     weight_decay: float,
#     c_local: torch.Tensor,
#     c_global: torch.Tensor,
#     fixed_steps: int = 390,  # ← Fixed K for all clients
# ) -> Tuple[int, int]:
#     """Train with fixed number of steps (SCAFFOLD assumption)."""
#     criterion = nn.CrossEntropyLoss()
#     optimizer = SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
#     net.train()
    
#     c_diff = c_global - c_local
#     total_samples = 0
    
#     # Create repeating iterator for clients with less data
#     data_iter = iter(trainloader)
    
#     for step in range(fixed_steps):
#         try:
#             data, target = next(data_iter)
#         except StopIteration:
#             # Restart iterator if we run out of data
#             data_iter = iter(trainloader)
#             data, target = next(data_iter)
        
#         data, target = data.to(device), target.to(device)
        
#         optimizer.zero_grad(set_to_none=True)
#         output = net(data)
#         loss = criterion(output, target)
#         loss.backward()
        
#         # Add SCAFFOLD correction
#         offset = 0
#         for param in net.parameters():
#             if not param.requires_grad:
#                 continue
#             numel = param.numel()
#             param.grad.add_(c_diff[offset:offset+numel].view_as(param))
#             offset += numel
        
#         optimizer.step()
#         total_samples += data.size(0)
    
#     return fixed_steps, total_samples
# models.py - Add this model

class TwoLayerMLP(nn.Module):
    """Two-layer fully connected network for MNIST/Fashion-MNIST"""
    def __init__(self, input_dim=784, hidden1=256, hidden2=128, num_classes=10):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Flatten the input
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to evaluate.
    testloader : DataLoader
        The test set dataloader object.
    device : torch.device
        The device on which to evaluate the network.

    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the network on the test set.
    """
    # print("Testing the model...")
    criterion = nn.CrossEntropyLoss(reduction="sum")
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss = loss / total
    acc = correct / total
    return loss, acc


class LogisticRegression(nn.Module):
    """A network for logistic regression using a single fully connected layer.

    As described in the Li et al., 2020 paper :

    [Federated Optimization in Heterogeneous Networks] (

    https://arxiv.org/pdf/1812.06127.pdf)
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(28 * 28, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = self.linear(torch.flatten(input_tensor, 1))
        return output_tensor


def train_fedavg_mnist(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    proximal_mu: float,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    proximal_mu : float
        Parameter for the weight of the proximal term.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        print("Training epoch... with device:", device)
        net = _train_fedavg_mnist_one_epoch(
            net, global_params, trainloader, device, criterion, optimizer, proximal_mu # type: ignore
        )


def _train_fedavg_mnist_one_epoch(
    net: nn.Module,
    global_params: list[Parameter],
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.Adam,
    proximal_mu: float,
) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    global_params : List[Parameter]
        The parameters of the global model (from the server).
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training
    proximal_mu : float
        Parameter for the weight of the proximal term.

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for batch in trainloader:
        label_key = (
            "character" if "character" in batch else "label"
        )  # FEMNIST's label is called "character"
        images, labels = batch["image"].to(device), batch[label_key].to(device)
        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(
            net.parameters(), global_params, strict=True
        ):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    return net

def test_mnist(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            label_key = (
                "character" if "character" in batch else "label"
            )  # FEMNIST's label is called "character"
            images, labels = batch["image"].to(device), batch[label_key].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

def test_mnist_scaffold(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate the network on the entire test set (MNIST/F-MNIST/FEMNIST-style)."""

    criterion = torch.nn.CrossEntropyLoss()
    net.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")

    with torch.no_grad():
        for batch in testloader:
            # Support both dict-batches (FEMNIST) and tuple-batches (torchvision MNIST)
            if isinstance(batch, dict):
                label_key = "character" if "character" in batch else "label"
                images = batch["image"].to(device)
                labels = batch[label_key].to(device)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)

            outputs = net(images)
            batch_loss = criterion(outputs, labels)

            # Accumulate loss properly
            bs = labels.size(0)
            total_loss += batch_loss.item() * bs

            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += bs
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def get_weights(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    """Load parameters into model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict()
    
    for k, v in params_dict:
        if k.endswith('num_batches_tracked'):
            tensor = torch.tensor(v, dtype=torch.long)
        else:
            tensor = torch.Tensor(v)
        state_dict[k] = tensor
    
    net.load_state_dict(state_dict, strict=False)

def set_weights_with_dtype_handling(model: torch.nn.Module, ndarrays: List[np.ndarray]) -> None:
    """
    Robustly load weights coming from Flower:
      - If ndarrays is empty/None: no-op (keeps current model weights).
      - If length matches trainable params: copy directly into .parameters().
      - If length matches full state_dict (params + buffers): load_state_dict.
      - Else: treat as flat chunk(s) that concatenate to total trainable size.
    Dtypes are cast to match target tensors (handles BN counters/FP16/BF16/FP32).
    """
    if ndarrays is None or len(ndarrays) == 0:
        # Nothing to set — caller likely invoked with no parameters this round.
        return

    device = next(model.parameters()).device
    target_dtype = next(model.parameters()).dtype

    state = model.state_dict()
    state_items = list(state.items())
    num_state = len(state_items)

    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_params = [p for _, p in trainable_named]
    num_trainable = len(trainable_params)

    def _to_like(arr, like: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(arr, device=device)
        if t.dtype != like.dtype:
            t = t.to(dtype=like.dtype)
        if t.numel() != like.numel():
            raise ValueError(
                f"Numel mismatch: got {t.numel()} vs expected {like.numel()} for shape {tuple(like.shape)}"
            )
        return t.view_as(like)

    # Case 1: trainable-only (Flower's common format)
    if len(ndarrays) == num_trainable:
        with torch.no_grad():
            for p, arr in zip(trainable_params, ndarrays):
                p.copy_(_to_like(arr, p))
        return

    # Case 2: full state_dict length (params + buffers)
    if len(ndarrays) == num_state:
        new_state = OrderedDict()
        for (k, ref), arr in zip(state_items, ndarrays):
            # Ensure BN tracking counters are long
            if k.endswith("num_batches_tracked"):
                new_state[k] = torch.as_tensor(arr, device=device, dtype=torch.long)
            else:
                new_state[k] = _to_like(arr, ref)
        # strict=False allows harmless missing buffers if any
        model.load_state_dict(new_state, strict=False)
        return

    # Case 3: flat vector(s) – concatenate and vector_to_parameters
    flat_src = np.concatenate([np.asarray(a).ravel() for a in ndarrays])  # will fail if empty
    total_trainable = sum(p.numel() for p in trainable_params)
    if flat_src.size != total_trainable:
        raise ValueError(
            f"Length mismatch: flat={flat_src.size}, trainable_total={total_trainable}, "
            f"in_len={len(ndarrays)}, state_len={num_state}, trainable_len={num_trainable}"
        )

    vec = torch.from_numpy(flat_src).to(device=device, dtype=target_dtype)
    torch.nn.utils.vector_to_parameters(vec, trainable_params)




def load_from_checkpoint_or_parameters(net, checkpoint_path, parameters=None):
    """Load model weights from checkpoint if exists, otherwise from parameters.
    
    Args:
        net: The model to load weights into
        checkpoint_path: Path to checkpoint file
        parameters: Optional list of ndarrays to load if checkpoint doesn't exist
        
    Returns:
        bool: True if loaded from checkpoint, False if loaded from parameters
    """
    if os.path.exists(checkpoint_path):
        net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        return True
    elif parameters is not None:
        set_weights_with_dtype_handling(net, parameters)
        return False
    else:
        # Model remains with random initialization
        return False

def instantiate_model(config: EasyDict):
    """Instantiate the model necessary for the experiment.

    Args:
        config (dict): The config used to determine the model type.

    Raises
    ------
        ValueError: The model type specified by the config is currently not supported

    Returns
    -------
        nn.Module: Instantiated model for experimentation.
    """

    # Set seeds for reproducibility
    # if "seed" in config:
    #     seed = config["seed"]
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)  # if using GPU
    #     np.random.seed(seed)
    #     random.seed(seed)

    model_name = config["model"]["name"]
    num_classes = config["model"]["num_classes"]

    if model_name == "ResNet9":
        net = ResNet9(input_dim=3, hidden_dims=[], num_classes=num_classes)
    elif model_name == "ResNet18":
        net = ResNet18(in_channels=3, num_classes=num_classes)
    elif model_name == "HeteroResNet18" or model_name == "HeteroPreResNet18":
        # Import HeteroPreResNet18 for model-heterogeneous FL (paper-faithful implementation)
        from .hetero_models import HeteroPreResNet18
        model_rate = config.get("global_model_rate", 1.0)
        net = HeteroPreResNet18(
            in_channels=3,
            num_classes=num_classes,
            model_rate=model_rate
        )
    elif model_name.startswith("resnet34"):
        net = make_resnet(model_name=model_name, num_classes=num_classes)
    elif model_name == "LogisticRegression":
        net = LogisticRegression(num_classes=num_classes)
    elif model_name == "TwoLayerMLP":
        net = TwoLayerMLP(input_dim=28*28, hidden1=256, hidden2=128, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}, This model type is currently not supported.")

    return net