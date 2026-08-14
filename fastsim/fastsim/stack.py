import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap

from .data import apply_augment, draw_augment


def predraw_client_steps(sim, idx, epochs):
    """Pre-draw one client's local-training randomness in the exact order the
    sequential ClientLoader would consume it: per epoch a permutation, then per
    batch the augmentation draws. Returns a flat list of
    (index_tensor, oy, ox, flip) across all epochs."""
    cfg = sim.cfg
    n = len(idx)
    steps = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=sim.train_gen, device=sim.device)
        for start in range(0, n, cfg.batch_size):
            batch = idx[perm[start:start + cfg.batch_size]]
            oy, ox, flip = draw_augment(len(batch), sim.train_gen, sim.device, hflip=cfg.hflip)
            steps.append((batch, oy, ox, flip))
    return steps


def _materialize(sim, step):
    batch, oy, ox, flip = step
    x = sim.data.train_x[batch]
    x = apply_augment(x, oy, ox, flip)
    x = sim.data.normalize(x)
    return x, sim.data.train_y[batch]


def _loss_kd(template, params, buffers, x, y):
    logits_t = functional_call(template, (params, buffers), (x,))
    loss = F.cross_entropy(logits_t, y)
    soft = F.softmax(logits_t.detach(), dim=1)
    logits_s = functional_call(template, (params, buffers), (x,))
    log_prob = F.log_softmax(logits_s, dim=1)
    return loss + (-(soft * log_prob).sum(dim=1).mean())


def _loss_plain(template, params, buffers, x, y):
    logits = functional_call(template, (params, buffers), (x,))
    return F.cross_entropy(logits, y)


class FjordNarrowStack:
    """Lockstep executor for a group of same-rate FjORD clients whose ordered
    dropout candidate set is the single width they own (no sub-model path).

    Each client's own batch order, sBN batch statistics, KD math, and
    SGD-with-momentum update sequence are preserved exactly; only the execution
    is stacked across clients with torch.func.vmap. Batches whose size differs
    from the modal size run as singleton groups so batch statistics stay exact.
    """

    def __init__(self, template, device, compile_mode="off"):
        self.template = template
        self.device = device
        self.compile_mode = compile_mode
        self._grad_fns = {}

    def _grads_fn(self, kd):
        if kd not in self._grad_fns:
            loss_fn = _loss_kd if kd else _loss_plain
            fn = lambda p, b, x, y: loss_fn(self.template, p, b, x, y)
            gfn = vmap(grad(fn), in_dims=(0, 0, 0, 0))
            if self.compile_mode != "off" and self.device.type == "cuda":
                gfn = torch.compile(gfn, mode=self.compile_mode)
            self._grad_fns[kd] = gfn
        return self._grad_fns[kd]

    def run(self, init_state, client_steps, sim, lr, momentum, weight_decay, kd):
        g = len(client_steps)
        pnames = set(n for n, _ in self.template.named_parameters())
        params = {k: torch.stack([init_state[k].detach().clone().float() for _ in range(g)])
                  for k in init_state if k in pnames}
        buffers = {k: torch.stack([init_state[k].detach().clone() for _ in range(g)])
                   for k in init_state if k not in pnames}
        momenta = {k: torch.zeros_like(v) for k, v in params.items()}
        stacked_grads = self._grads_fn(kd)

        max_steps = max(len(s) for s in client_steps)
        for s in range(max_steps):
            active = [i for i in range(g) if s < len(client_steps[i])]
            sizes = {}
            for i in active:
                sizes.setdefault(len(client_steps[i][s][0]), []).append(i)
            for size, members in sizes.items():
                xs, ys = [], []
                for i in members:
                    x, y = _materialize(sim, client_steps[i][s])
                    xs.append(x)
                    ys.append(y)
                xs = torch.stack(xs)
                ys = torch.stack(ys)
                sel = torch.tensor(members, device=self.device)
                sel_params = {k: v[sel] for k, v in params.items()}
                sel_buffers = {k: v[sel] for k, v in buffers.items()}
                grads = stacked_grads(sel_params, sel_buffers, xs, ys)
                for k in params:
                    gk = grads[k]
                    if weight_decay:
                        gk = gk + weight_decay * sel_params[k]
                    if momentum:
                        buf = momentum * momenta[k][sel] + gk
                        momenta[k][sel] = buf
                        gk = buf
                    params[k][sel] = sel_params[k] - lr * gk
        out = []
        for i in range(g):
            state = {}
            for k, v in init_state.items():
                if k in params:
                    state[k] = params[k][i].to(v.dtype)
                else:
                    state[k] = v.detach().clone()
            out.append(state)
        return out
