"""Train a rate-0.25 core (use_scaler=False, sBN) centrally on CIFAR-100, save the
checkpoint nearest acc=0.466 (matching the homo_small FL core) for the skyline oracles."""
import sys, time
import torch, torch.nn as nn
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from flore.hetero_models import HeteroPreResNet18, calibrate_bn_stats
from probe.skyline_common import (DEV, TF_TRAIN, TF_TEST, ROOT, test_loader, calib_loader)
import torchvision as tv
from torch.utils.data import DataLoader

TARGET = 0.466
CKPT = str(_P(__file__).resolve().parent / "core025_sbn.pt")

train = tv.datasets.CIFAR100(ROOT, train=True, download=True, transform=TF_TRAIN)
trainld = DataLoader(train, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
testld = test_loader(); calibld = calib_loader()

torch.manual_seed(42)
net = HeteroPreResNet18(in_channels=3, num_classes=100, model_rate=0.25,
                        use_scaler=False, norm_type="sbn").to(DEV)
print(f"rate-0.25 core params={sum(p.numel() for p in net.parameters())/1e6:.3f}M")
opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
EPOCHS = 30
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate():
    calibrate_bn_stats(net, calibld, DEV, num_batches=20)
    net.eval(); c = t = 0
    for x, y in testld:
        x, y = x.to(DEV), y.to(DEV)
        c += (net(x).argmax(1) == y).sum().item(); t += y.size(0)
    return c / t

best = None
print("epoch test_acc lr")
for ep in range(1, EPOCHS + 1):
    net.train()
    for x, y in trainld:
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad(); crit(net(x), y).backward(); opt.step()
    sched.step()
    acc = evaluate()
    print(f"{ep:3d} {acc:.4f} {opt.param_groups[0]['lr']:.4f}", flush=True)
    if best is None or abs(acc - TARGET) < abs(best[0] - TARGET):
        best = (acc, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
        torch.save({"acc": acc, "state": best[1]}, CKPT)
    if acc >= TARGET + 0.02 and ep >= 6:
        print(f"reached target+; stopping. best saved acc={best[0]:.4f}")
        break
print(f"SAVED core nearest {TARGET}: acc={best[0]:.4f} -> {CKPT}")
