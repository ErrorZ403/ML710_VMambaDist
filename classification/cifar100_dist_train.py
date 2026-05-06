import argparse
import math
import os
import random
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

# Keep import style aligned with classification/main.py execution pattern.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from models import vmamba as vmamba_mod  # noqa: E402


VSSM = vmamba_mod.VSSM
SS2D = vmamba_mod.SS2D


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


@dataclass
class DistEnv:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


@dataclass
class TrainEpochResult:
    loss: float
    acc: float
    samples: float
    steps: int
    compute_time_s: float
    first_step_loss: float
    last_step_loss: float


def parse_int_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Could not parse integer list from: {text}")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("VMamba CIFAR-100 distributed trainer")
    parser.add_argument("--parallel-mode", type=str, default="ddp", choices=["ddp", "scan_tp", "pipeline"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./output/cifar100_vmamba")
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=10)

    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", type=str, default="")
    parser.add_argument("--sample-milestones", type=str, default="100000,500000")
    parser.add_argument("--iter-milestone", type=int, default=100000)
    parser.add_argument("--walltime-milestone-hours", type=float, default=10.0)

    # VMamba hyper-parameters tuned for CIFAR-sized inputs.
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--depths", type=str, default="2,2,5,2")
    parser.add_argument("--dims", type=str, default="96")
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--ssm-d-state", type=int, default=1)
    parser.add_argument("--ssm-ratio", type=float, default=1.0)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--forward-type", type=str, default="v05_noz")
    parser.add_argument("--ssm-conv", type=int, default=3)

    # Pipeline-specific.
    parser.add_argument("--pipeline-devices", type=str, default="0,1")

    return parser.parse_args()


def init_distributed(args: argparse.Namespace) -> DistEnv:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
    use_cuda = torch.cuda.is_available()

    needs_dist = args.parallel_mode in {"ddp", "scan_tp"} and world_size > 1
    if needs_dist:
        if not use_cuda:
            raise RuntimeError("DDP/scan_tp mode requires CUDA for this script.")
        torch.cuda.set_device(local_rank)
        # Newer PyTorch versions accept `device_id` and avoid NCCL rank->device guessing warnings.
        try:
            dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device("cuda", local_rank))
        except TypeError:
            dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
        return DistEnv(distributed=True, rank=rank, world_size=world_size, local_rank=local_rank, device=device)

    if args.parallel_mode == "pipeline" and world_size > 1:
        raise RuntimeError("Pipeline mode is single-process in this script. Use plain `python`, not multi-rank torchrun.")

    if use_cuda:
        if args.parallel_mode != "pipeline":
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    return DistEnv(distributed=False, rank=rank, world_size=world_size, local_rank=local_rank, device=device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int, rank: int, same_across_ranks: bool) -> None:
    full_seed = seed if same_across_ranks else seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(full_seed)


def is_main_process(env: DistEnv) -> bool:
    return env.rank == 0


def rank0_print(env: DistEnv, text: str) -> None:
    if is_main_process(env):
        print(text, flush=True)


def resolve_cifar_root(data_dir: str) -> str:
    """
    torchvision.datasets.CIFAR100 expects:
      root/
        cifar-100-python/
          train, test, meta
    Users often pass `.../cifar-100-python` directly; map it to parent.
    """
    p = Path(data_dir).expanduser().resolve()
    if p.name == "cifar-100-python" and (p / "train").exists() and (p / "test").exists() and (p / "meta").exists():
        return str(p.parent)
    return str(p)


def maybe_prepare_cifar_download(args: argparse.Namespace, env: DistEnv) -> None:
    root = resolve_cifar_root(args.data_dir)
    if args.parallel_mode == "scan_tp":
        if is_main_process(env) and args.download:
            datasets.CIFAR100(root=root, train=True, download=True)
            datasets.CIFAR100(root=root, train=False, download=True)
        if env.distributed:
            dist.barrier()
        return

    if env.distributed:
        if is_main_process(env) and args.download:
            datasets.CIFAR100(root=root, train=True, download=True)
            datasets.CIFAR100(root=root, train=False, download=True)
        dist.barrier()
    elif args.download:
        datasets.CIFAR100(root=root, train=True, download=True)
        datasets.CIFAR100(root=root, train=False, download=True)


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, val_transform


def build_cifar100_loaders(
    args: argparse.Namespace, env: DistEnv
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DistributedSampler]]:
    train_transform, val_transform = build_transforms()
    cifar_root = resolve_cifar_root(args.data_dir)

    if args.parallel_mode == "scan_tp":
        if not is_main_process(env):
            return None, None, None
        train_set = datasets.CIFAR100(root=cifar_root, train=True, transform=train_transform, download=False)
        val_set = datasets.CIFAR100(root=cifar_root, train=False, transform=val_transform, download=False)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
        return train_loader, val_loader, None

    train_set = datasets.CIFAR100(root=cifar_root, train=True, transform=train_transform, download=False)
    val_set = datasets.CIFAR100(root=cifar_root, train=False, transform=val_transform, download=False)

    train_sampler: Optional[DistributedSampler] = None
    val_sampler = None
    if env.distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=env.world_size, rank=env.rank, shuffle=True)
        val_sampler = DistributedSampler(val_set, num_replicas=env.world_size, rank=env.rank, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def build_vmamba(args: argparse.Namespace) -> VSSM:
    dims_list = parse_int_list(args.dims)
    dims = dims_list[0] if len(dims_list) == 1 else dims_list
    depths = parse_int_list(args.depths)
    model = VSSM(
        patch_size=args.patch_size,
        in_chans=3,
        num_classes=100,
        depths=depths,
        dims=dims,
        ssm_d_state=args.ssm_d_state,
        ssm_ratio=args.ssm_ratio,
        ssm_dt_rank="auto",
        ssm_conv=args.ssm_conv,
        ssm_conv_bias=False,
        forward_type=args.forward_type,
        mlp_ratio=args.mlp_ratio,
        drop_path_rate=args.drop_path,
        norm_layer="ln2d",
        downsample_version="v3",
        patchembed_version="v2",
        imgsize=32,
    )
    return model


class PipelineVSSM(nn.Module):
    """
    Simple pipeline-style model parallelism:
    move whole VSS blocks (and downsample units) onto different devices.
    """

    def __init__(self, base_model: VSSM, device_ids: Sequence[int]):
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("Pipeline mode requires CUDA.")
        if len(device_ids) < 2:
            raise ValueError("Pipeline mode needs at least 2 device ids, e.g. --pipeline-devices 0,1")

        self.device_ids = [int(d) for d in device_ids]
        self.devices = [torch.device("cuda", d) for d in self.device_ids]
        self.input_device = self.devices[0]
        self.output_device = self.devices[-1]
        self.channel_first = base_model.channel_first

        self.patch_embed = base_model.patch_embed.to(self.input_device)
        self.pos_embed = base_model.pos_embed
        if self.pos_embed is not None:
            self.pos_embed = nn.Parameter(self.pos_embed.to(self.input_device))

        units: List[nn.Module] = []
        for layer in base_model.layers:
            for block in layer.blocks:
                units.append(block)
            if not isinstance(layer.downsample, nn.Identity):
                units.append(layer.downsample)
        self.units = nn.ModuleList(units)
        self.unit_devices = self._build_contiguous_assignment(len(self.units), self.devices)

        for module, dev in zip(self.units, self.unit_devices):
            module.to(dev)

        self.classifier = base_model.classifier.to(self.output_device)

    @staticmethod
    def _build_contiguous_assignment(num_units: int, devices: Sequence[torch.device]) -> List[torch.device]:
        if num_units == 0:
            return []
        n_dev = len(devices)
        boundaries = [round(i * num_units / n_dev) for i in range(n_dev + 1)]
        assignment: List[torch.device] = []
        for i in range(n_dev):
            assignment.extend([devices[i]] * (boundaries[i + 1] - boundaries[i]))
        return assignment

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.input_device, non_blocking=True)
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed if self.channel_first else self.pos_embed.permute(0, 2, 3, 1)
            x = x + pos_embed

        for module, dev in zip(self.units, self.unit_devices):
            x = x.to(dev, non_blocking=True)
            x = module(x)

        x = x.to(self.output_device, non_blocking=True)
        x = self.classifier(x)
        return x


def _scan_tp_forward_core(self: SS2D, x: torch.Tensor, **runtime_kwargs) -> torch.Tensor:
    cfg = dict(getattr(self, "_scan_tp_cfg", {}))
    cfg.update(runtime_kwargs)

    force_fp32 = bool(cfg.get("force_fp32", False))
    ssoflex = bool(cfg.get("ssoflex", True))
    no_einsum = bool(cfg.get("no_einsum", False))
    selective_scan_backend = cfg.get("selective_scan_backend", None)
    scan_mode = cfg.get("scan_mode", "cross2d")
    scan_force_torch = bool(cfg.get("scan_force_torch", False))

    scan_map = {"cross2d": 0, "unidi": 1, "bidi": 2, "cascade2d": -1}
    scan_mode_id = scan_map.get(scan_mode, scan_mode if isinstance(scan_mode, int) else None)

    # Fallback for modes this branch-sharding implementation does not support.
    if scan_mode_id != 0 or (not dist.is_available()) or (not dist.is_initialized()):
        return self.forward_corev2(x, **cfg)

    group = getattr(self, "_scan_tp_group", None)
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    B, _, H, W = x.shape
    K = self.k_group
    D = self.d_inner
    N = self.d_state
    R = self.dt_rank
    L = H * W
    local_ids = [branch for branch in range(K) if branch % world_size == rank]

    def selective_scan(u, delta, A, Bv, Cv, Dv=None, delta_bias=None, delta_softplus=True):
        return vmamba_mod.selective_scan_fn(
            u,
            delta,
            A,
            Bv,
            Cv,
            Dv,
            delta_bias,
            delta_softplus,
            ssoflex,
            backend=selective_scan_backend,
        )

    x_proj_bias = getattr(self, "x_proj_bias", None)
    xs = vmamba_mod.cross_scan_fn(
        x, in_channel_first=True, out_channel_first=True, scans=scan_mode_id, force_torch=scan_force_torch
    )
    local_y_flat = torch.zeros((B, D, L), device=x.device, dtype=torch.float32)

    if local_ids:
        xs_local = xs[:, local_ids].contiguous()  # (B, K_local, D, L)
        k_local = len(local_ids)

        if no_einsum:
            x_dbl = F.conv1d(
                xs_local.view(B, -1, L),
                self.x_proj_weight[local_ids].contiguous().view(-1, D, 1),
                bias=(x_proj_bias[local_ids].contiguous().view(-1) if x_proj_bias is not None else None),
                groups=k_local,
            )
            dts, Bs, Cs = torch.split(x_dbl.view(B, k_local, -1, L), [R, N, N], dim=2)
            dts = F.conv1d(
                dts.contiguous().view(B, -1, L),
                self.dt_projs_weight[local_ids].contiguous().view(k_local * D, -1, 1),
                groups=k_local,
            )
        else:
            x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_local, self.x_proj_weight[local_ids])
            if x_proj_bias is not None:
                x_dbl = x_dbl + x_proj_bias[local_ids].view(1, k_local, -1, 1)
            dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
            dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight[local_ids])

        xs_local = xs_local.view(B, -1, L)
        dts = dts.contiguous().view(B, -1, L)
        Bs = Bs.contiguous().view(B, k_local, N, L)
        Cs = Cs.contiguous().view(B, k_local, N, L)
        As = -self.A_logs.to(torch.float32).exp().view(K, D, N)[local_ids].contiguous().view(-1, N)
        Ds = self.Ds.to(torch.float32).view(K, D)[local_ids].contiguous().view(-1)
        delta_bias = self.dt_projs_bias[local_ids].contiguous().view(-1).to(torch.float32)

        if force_fp32:
            xs_local = xs_local.to(torch.float32)
            dts = dts.to(torch.float32)
            Bs = Bs.to(torch.float32)
            Cs = Cs.to(torch.float32)
        else:
            dts = dts.to(xs_local.dtype)
            Bs = Bs.to(xs_local.dtype)
            Cs = Cs.to(xs_local.dtype)

        ys_local = selective_scan(xs_local, dts, As, Bs, Cs, Ds, delta_bias, True).view(B, k_local, -1, L)

        for local_idx, branch_id in enumerate(local_ids):
            branch = ys_local[:, local_idx]  # (B, D, L)
            if branch_id == 0:
                contrib = branch
            elif branch_id == 1:
                contrib = branch.view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
            elif branch_id == 2:
                contrib = torch.flip(branch, dims=[-1])
            elif branch_id == 3:
                contrib = torch.flip(branch, dims=[-1]).view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
            else:
                raise RuntimeError(f"Unexpected branch id {branch_id} for K={K}")
            local_y_flat = local_y_flat + contrib.to(torch.float32)

    dist.all_reduce(local_y_flat, op=dist.ReduceOp.SUM, group=group)

    y = local_y_flat.view(B, -1, H, W)
    if not self.channel_first:
        y = y.view(B, -1, H * W).transpose(1, 2).contiguous().view(B, H, W, -1)
    y = self.out_norm(y)
    return y.to(x.dtype)


def enable_scan_branch_parallel(model: nn.Module, group=None) -> int:
    patched = 0
    for module in model.modules():
        if not isinstance(module, SS2D):
            continue
        forward_core = getattr(module, "forward_core", None)
        if not isinstance(forward_core, partial):
            continue
        if getattr(forward_core.func, "__name__", "") != "forward_corev2":
            continue

        cfg = dict(forward_core.keywords or {})
        scan_mode = cfg.get("scan_mode", "cross2d")
        scan_map = {"cross2d": 0, "unidi": 1, "bidi": 2, "cascade2d": -1}
        scan_mode_id = scan_map.get(scan_mode, scan_mode if isinstance(scan_mode, int) else None)
        if scan_mode_id != 0:
            continue

        module._scan_tp_enabled = True
        module._scan_tp_group = group
        module._scan_tp_cfg = cfg
        module._scan_tp_original_forward_core = forward_core
        module.forward_core = types.MethodType(_scan_tp_forward_core, module)
        patched += 1
    return patched


@torch.no_grad()
def sync_scan_branch_parameters(model: nn.Module, group=None) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size(group=group)

    for module in model.modules():
        if not getattr(module, "_scan_tp_enabled", False):
            continue
        K = module.k_group
        D = module.d_inner
        N = module.d_state
        a_logs = module.A_logs.view(K, D, N)
        d_vals = module.Ds.view(K, D)
        x_proj_bias = getattr(module, "x_proj_bias", None)
        for branch in range(K):
            src = branch % world_size
            dist.broadcast(module.x_proj_weight[branch], src=src, group=group)
            dist.broadcast(module.dt_projs_weight[branch], src=src, group=group)
            dist.broadcast(module.dt_projs_bias[branch], src=src, group=group)
            dist.broadcast(a_logs[branch], src=src, group=group)
            dist.broadcast(d_vals[branch], src=src, group=group)
            if x_proj_bias is not None:
                dist.broadcast(x_proj_bias[branch], src=src, group=group)


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


def reduce_stats_if_needed(stats: torch.Tensor, env: DistEnv) -> torch.Tensor:
    if env.distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return stats


def reduce_scalar_max(value: float, env: DistEnv, device: torch.device) -> float:
    if not env.distributed:
        return float(value)
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def reduce_scalar_mean(value: float, env: DistEnv, device: torch.device) -> float:
    if not env.distributed:
        return float(value)
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= env.world_size
    return float(t.item())


def maybe_cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def count_model_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def query_single_gpu_metrics(device_idx: int) -> dict:
    metrics = {}
    if not torch.cuda.is_available():
        return metrics

    total_mem_bytes = float(torch.cuda.get_device_properties(device_idx).total_memory)
    alloc_bytes = float(torch.cuda.memory_allocated(device_idx))
    reserve_bytes = float(torch.cuda.memory_reserved(device_idx))
    peak_alloc_bytes = float(torch.cuda.max_memory_allocated(device_idx))
    peak_reserve_bytes = float(torch.cuda.max_memory_reserved(device_idx))

    metrics.update(
        {
            "gpu_mem_allocated_gb": alloc_bytes / (1024 ** 3),
            "gpu_mem_reserved_gb": reserve_bytes / (1024 ** 3),
            "gpu_mem_peak_allocated_gb": peak_alloc_bytes / (1024 ** 3),
            "gpu_mem_peak_reserved_gb": peak_reserve_bytes / (1024 ** 3),
            "gpu_mem_peak_util_pct": 100.0 * peak_reserve_bytes / max(total_mem_bytes, 1.0),
        }
    )

    # nvidia-smi live telemetry (if available).
    try:
        cmd = [
            "nvidia-smi",
            f"--id={device_idx}",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        vals = [x.strip() for x in out.split(",")]
        if len(vals) >= 6:
            metrics.update(
                {
                    "gpu_utilization_pct": float(vals[0]),
                    "gpu_memory_utilization_pct": float(vals[1]),
                    "gpu_memory_used_gb": float(vals[2]) / 1024.0,
                    "gpu_memory_total_gb": float(vals[3]) / 1024.0,
                    "gpu_power_w": float(vals[4]),
                    "gpu_temperature_c": float(vals[5]),
                }
            )
    except Exception:
        # Keep memory metrics even if nvidia-smi is unavailable.
        pass

    return metrics


def aggregate_local_gpu_metrics(device_indices: Sequence[int]) -> dict:
    if not device_indices:
        return {}
    per_gpu = [query_single_gpu_metrics(i) for i in device_indices]
    keys = sorted({k for d in per_gpu for k in d.keys()})
    out = {}
    for k in keys:
        vals = [d[k] for d in per_gpu if k in d]
        if vals:
            out[k] = float(sum(vals) / len(vals))
    return out


def compute_training_kpis(
    samples: float,
    epoch_wall_s: float,
    compute_s: float,
    first_step_loss: float,
    last_step_loss: float,
) -> dict:
    eps = 1e-12
    throughput = samples / max(epoch_wall_s, eps)
    compute_util = compute_s / max(epoch_wall_s, eps)
    compute_util = max(0.0, min(compute_util, 1.0))
    system_goodput = throughput * compute_util
    stat_eff_loss_per_sample = (first_step_loss - last_step_loss) / max(samples, eps)
    stat_goodput_loss_per_sec = throughput * stat_eff_loss_per_sample
    return {
        "throughput_samples_per_s": throughput,
        "compute_utilization": compute_util,
        "system_goodput_samples_per_s": system_goodput,
        "stat_eff_loss_gain_per_sample": stat_eff_loss_per_sample,
        "stat_goodput_loss_gain_per_s": stat_goodput_loss_per_sec,
    }


def train_one_epoch_standard(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    env: DistEnv,
    input_device: torch.device,
    print_freq: int,
) -> TrainEpochResult:
    model.train()
    loss_sum = 0.0
    correct_sum = 0.0
    sample_sum = 0.0
    compute_time_s = 0.0
    first_step_loss: Optional[float] = None
    last_step_loss: float = 0.0
    num_steps = 0

    for step, (images, targets) in enumerate(loader):
        compute_t0 = time.perf_counter()
        images = images.to(input_device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            targets_dev = targets.to(outputs.device, non_blocking=True)
            loss = criterion(outputs, targets_dev)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        maybe_cuda_sync()
        compute_time_s += time.perf_counter() - compute_t0
        num_steps = step + 1

        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            batch = targets_dev.numel()
            correct = (preds == targets_dev).sum().item()
            step_loss = float(loss.item())
            if first_step_loss is None:
                first_step_loss = step_loss
            last_step_loss = step_loss
            loss_sum += step_loss * batch
            correct_sum += float(correct)
            sample_sum += float(batch)

        if is_main_process(env) and (step + 1) % print_freq == 0:
            avg_loss = loss_sum / max(sample_sum, 1.0)
            avg_acc = 100.0 * correct_sum / max(sample_sum, 1.0)
            print(f"  step {step + 1}/{len(loader)}  loss={avg_loss:.4f}  acc@1={avg_acc:.2f}%", flush=True)

    stats = torch.tensor([loss_sum, correct_sum, sample_sum], device=input_device, dtype=torch.float64)
    stats = reduce_stats_if_needed(stats, env)
    train_loss = float(stats[0].item() / max(stats[2].item(), 1.0))
    train_acc = float(100.0 * stats[1].item() / max(stats[2].item(), 1.0))
    if first_step_loss is None:
        first_step_loss = 0.0
    return TrainEpochResult(
        loss=train_loss,
        acc=train_acc,
        samples=float(stats[2].item()),
        steps=num_steps,
        compute_time_s=compute_time_s,
        first_step_loss=first_step_loss,
        last_step_loss=last_step_loss,
    )


@torch.no_grad()
def evaluate_standard(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    amp_enabled: bool,
    env: DistEnv,
    input_device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    sample_sum = 0.0

    for images, targets in loader:
        images = images.to(input_device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            targets_dev = targets.to(outputs.device, non_blocking=True)
            loss = criterion(outputs, targets_dev)

        preds = outputs.argmax(dim=1)
        batch = targets_dev.numel()
        correct = (preds == targets_dev).sum().item()
        loss_sum += float(loss.item()) * batch
        correct_sum += float(correct)
        sample_sum += float(batch)

    stats = torch.tensor([loss_sum, correct_sum, sample_sum], device=input_device, dtype=torch.float64)
    stats = reduce_stats_if_needed(stats, env)
    val_loss = float(stats[0].item() / max(stats[2].item(), 1.0))
    val_acc = float(100.0 * stats[1].item() / max(stats[2].item(), 1.0))
    return val_loss, val_acc


def iter_scan_tp_batches(
    loader: Optional[DataLoader],
    env: DistEnv,
    img_size: int = 32,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    if not env.distributed:
        raise RuntimeError("scan_tp batch iterator requires distributed initialization.")
    device = env.device
    iterator = iter(loader) if is_main_process(env) else None

    while True:
        if is_main_process(env):
            try:
                images, targets = next(iterator)  # type: ignore[arg-type]
                batch_size = int(images.shape[0])
            except StopIteration:
                images, targets = None, None
                batch_size = 0
        else:
            images, targets = None, None
            batch_size = 0

        batch_size_tensor = torch.tensor([batch_size], device=device, dtype=torch.int64)
        dist.broadcast(batch_size_tensor, src=0)
        batch_size = int(batch_size_tensor.item())
        if batch_size == 0:
            break

        if is_main_process(env):
            images = images.to(device, non_blocking=True)  # type: ignore[union-attr]
            targets = targets.to(device, non_blocking=True)  # type: ignore[union-attr]
        else:
            images = torch.empty((batch_size, 3, img_size, img_size), device=device, dtype=torch.float32)
            targets = torch.empty((batch_size,), device=device, dtype=torch.long)

        dist.broadcast(images, src=0)
        dist.broadcast(targets, src=0)
        yield images, targets


def train_one_epoch_scan_tp(
    model: nn.Module,
    train_loader_rank0: Optional[DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    amp_enabled: bool,
    env: DistEnv,
    print_freq: int,
) -> TrainEpochResult:
    model.train()
    loss_sum = 0.0
    correct_sum = 0.0
    sample_sum = 0.0
    compute_time_s = 0.0
    first_step_loss: Optional[float] = None
    last_step_loss: float = 0.0
    num_steps = 0

    for step, (images, targets) in enumerate(iter_scan_tp_batches(train_loader_rank0, env)):
        compute_t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets.to(outputs.device))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        maybe_cuda_sync()
        compute_time_s += time.perf_counter() - compute_t0
        num_steps = step + 1

        # Keep full-model replicas consistent after each ownership update.
        sync_scan_branch_parameters(model)

        if is_main_process(env):
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                batch = targets.numel()
                correct = (preds == targets.to(preds.device)).sum().item()
                step_loss = float(loss.item())
                if first_step_loss is None:
                    first_step_loss = step_loss
                last_step_loss = step_loss
                loss_sum += step_loss * batch
                correct_sum += float(correct)
                sample_sum += float(batch)
                if (step + 1) % print_freq == 0:
                    avg_loss = loss_sum / max(sample_sum, 1.0)
                    avg_acc = 100.0 * correct_sum / max(sample_sum, 1.0)
                    print(f"  step {step + 1}  loss={avg_loss:.4f}  acc@1={avg_acc:.2f}%", flush=True)

    # Metrics are tracked on rank 0 only (all ranks process the same batches in scan_tp mode).
    if is_main_process(env):
        train_loss = loss_sum / max(sample_sum, 1.0)
        train_acc = 100.0 * correct_sum / max(sample_sum, 1.0)
        if first_step_loss is None:
            first_step_loss = 0.0
    else:
        train_loss = 0.0
        train_acc = 0.0
        first_step_loss = 0.0
    return TrainEpochResult(
        loss=train_loss,
        acc=train_acc,
        samples=float(sample_sum),
        steps=num_steps,
        compute_time_s=compute_time_s,
        first_step_loss=first_step_loss,
        last_step_loss=last_step_loss,
    )


@torch.no_grad()
def evaluate_scan_tp(
    model: nn.Module,
    val_loader_rank0: Optional[DataLoader],
    criterion: nn.Module,
    amp_enabled: bool,
    env: DistEnv,
) -> Tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct_sum = 0.0
    sample_sum = 0.0

    for images, targets in iter_scan_tp_batches(val_loader_rank0, env):
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets.to(outputs.device))

        if is_main_process(env):
            preds = outputs.argmax(dim=1)
            batch = targets.numel()
            correct = (preds == targets.to(preds.device)).sum().item()
            loss_sum += float(loss.item()) * batch
            correct_sum += float(correct)
            sample_sum += float(batch)

    if is_main_process(env):
        val_loss = loss_sum / max(sample_sum, 1.0)
        val_acc = 100.0 * correct_sum / max(sample_sum, 1.0)
    else:
        val_loss = 0.0
        val_acc = 0.0
    return val_loss, val_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_acc: float,
    args: argparse.Namespace,
    env: DistEnv,
) -> None:
    if not is_main_process(env):
        return
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    ckpt = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "args": vars(args),
    }
    path = out_dir / f"ckpt_epoch_{epoch:03d}.pth"
    torch.save(ckpt, path)
    print(f"Saved checkpoint: {path}", flush=True)


def main() -> None:
    args = parse_args()
    env = init_distributed(args)

    seed_everything(args.seed, env.rank, same_across_ranks=(args.parallel_mode == "scan_tp"))
    amp_enabled = torch.cuda.is_available() and (not args.disable_amp)

    rank0_print(env, f"Parallel mode: {args.parallel_mode}")
    rank0_print(env, f"World size: {env.world_size} | Rank: {env.rank} | Device: {env.device}")

    maybe_prepare_cifar_download(args, env)
    train_loader, val_loader, train_sampler = build_cifar100_loaders(args, env)

    base_model = build_vmamba(args)
    input_device = env.device
    loss_device = env.device

    if args.parallel_mode == "pipeline":
        device_ids = parse_int_list(args.pipeline_devices)
        model: nn.Module = PipelineVSSM(base_model, device_ids=device_ids)
        input_device = model.input_device 
        loss_device = model.output_device 
    else:
        model = base_model.to(env.device)

    if args.parallel_mode == "scan_tp":
        if not env.distributed:
            raise RuntimeError("scan_tp mode requires torchrun with nproc_per_node > 1.")
        patched = enable_scan_branch_parallel(model)
        if patched == 0:
            raise RuntimeError("No SS2D modules were patched for scan branch tensor parallelism.")
        rank0_print(env, f"Patched SS2D modules for scan_tp: {patched}")
        sync_scan_branch_parameters(model)

    total_params, trainable_params = count_model_parameters(model)
    model_size_mb = total_params * 4 / (1024 ** 2)  # fp32 footprint estimate
    rank0_print(
        env,
        f"Model params | total={total_params:,} trainable={trainable_params:,} "
        f"(~{model_size_mb:.2f} MB fp32)",
    )

    if args.parallel_mode == "ddp" and env.distributed:
        model = DistributedDataParallel(model, device_ids=[env.local_rank], broadcast_buffers=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(loss_device)
    optimizer = build_optimizer(args, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if is_main_process(env) and args.tensorboard:
        if SummaryWriter is None:
            print("TensorBoard requested but unavailable. Install `tensorboard` package.", flush=True)
        else:
            tb_dir = Path(args.tensorboard_dir).expanduser() if args.tensorboard_dir else (out_dir / "tensorboard")
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard logging enabled: {tb_dir}", flush=True)
            writer.add_scalar("model/total_params", total_params, 0)
            writer.add_scalar("model/trainable_params", trainable_params, 0)
            writer.add_scalar("model/fp32_size_mb", model_size_mb, 0)
            writer.flush()

    # Milestone metrics (aligned with your comparison table style).
    sample_milestones = sorted(parse_int_list(args.sample_milestones))
    sample_m1 = sample_milestones[0] if len(sample_milestones) > 0 else 100000
    sample_m2 = sample_milestones[1] if len(sample_milestones) > 1 else 500000
    walltime_milestone_s = max(args.walltime_milestone_hours, 0.0) * 3600.0
    iter_milestone = max(int(args.iter_milestone), 1)

    cumulative_wall_s = 0.0
    cumulative_samples = 0.0
    cumulative_steps = 0.0
    loss_at_0: Optional[float] = None
    loss_at_m1: Optional[float] = None
    loss_at_m2: Optional[float] = None
    loss_at_wall_milestone: Optional[float] = None
    val_loss_at_wall_milestone: Optional[float] = None
    wall_time_to_iter_milestone_h: Optional[float] = None

    # GPUs to monitor for utilization/memory telemetry.
    metric_device_indices: List[int] = []
    if torch.cuda.is_available():
        if args.parallel_mode == "pipeline":
            metric_device_indices = [int(i) for i in parse_int_list(args.pipeline_devices)]
        else:
            if env.device.index is not None:
                metric_device_indices = [int(env.device.index)]

    if args.eval_only:
        if args.parallel_mode == "scan_tp":
            val_loss, val_acc = evaluate_scan_tp(model, val_loader, criterion, amp_enabled, env)
        else:
            val_loss, val_acc = evaluate_standard(model, val_loader, criterion, amp_enabled, env, input_device)  # type: ignore[arg-type]
        if is_main_process(env):
            print(f"Eval | loss={val_loss:.4f} | acc@1={val_acc:.2f}%")
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, 0)
                writer.add_scalar("val/acc1", val_acc, 0)
                writer.flush()
                writer.close()
        cleanup_distributed()
        return

    best_acc = 0.0
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if torch.cuda.is_available():
            for dev_idx in metric_device_indices:
                torch.cuda.reset_peak_memory_stats(dev_idx)

        t0 = time.time()
        if args.parallel_mode == "scan_tp":
            train_result = train_one_epoch_scan_tp(
                model, train_loader, criterion, optimizer, scaler, amp_enabled, env, args.print_freq
            )
            val_loss, val_acc = evaluate_scan_tp(model, val_loader, criterion, amp_enabled, env)
        else:
            train_result = train_one_epoch_standard(
                model, train_loader, criterion, optimizer, scaler, amp_enabled, env, input_device, args.print_freq
            )  # type: ignore[arg-type]
            val_loss, val_acc = evaluate_standard(model, val_loader, criterion, amp_enabled, env, input_device)  # type: ignore[arg-type]

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_time = reduce_scalar_max(epoch_time, env, env.device)
        compute_time = reduce_scalar_max(train_result.compute_time_s, env, env.device)
        epoch_steps = reduce_scalar_max(float(train_result.steps), env, env.device)
        kpis = compute_training_kpis(
            samples=train_result.samples,
            epoch_wall_s=epoch_time,
            compute_s=compute_time,
            first_step_loss=train_result.first_step_loss,
            last_step_loss=train_result.last_step_loss,
        )
        throughput_iter_per_s = epoch_steps / max(epoch_time, 1e-12)

        if loss_at_0 is None:
            loss_at_0 = train_result.first_step_loss
        cumulative_wall_s += epoch_time
        cumulative_samples += train_result.samples
        cumulative_steps += epoch_steps

        if loss_at_m1 is None and cumulative_samples >= sample_m1:
            loss_at_m1 = train_result.loss
        if loss_at_m2 is None and cumulative_samples >= sample_m2:
            loss_at_m2 = train_result.loss
        if loss_at_wall_milestone is None and cumulative_wall_s >= walltime_milestone_s:
            loss_at_wall_milestone = train_result.loss
            val_loss_at_wall_milestone = val_loss
        if wall_time_to_iter_milestone_h is None and cumulative_steps >= iter_milestone:
            wall_time_to_iter_milestone_h = cumulative_wall_s / 3600.0

        stat_eff_0_m1 = ((loss_at_0 - loss_at_m1) / float(sample_m1)) if (loss_at_0 is not None and loss_at_m1 is not None) else float("nan")
        stat_eff_m1_m2 = ((loss_at_m1 - loss_at_m2) / float(max(sample_m2 - sample_m1, 1))) if (loss_at_m1 is not None and loss_at_m2 is not None) else float("nan")
        goodput_0_m1 = throughput_iter_per_s * stat_eff_0_m1 if not math.isnan(stat_eff_0_m1) else float("nan")
        goodput_m1_m2 = throughput_iter_per_s * stat_eff_m1_m2 if not math.isnan(stat_eff_m1_m2) else float("nan")
        cumulative_throughput_samples_per_s = cumulative_samples / max(cumulative_wall_s, 1e-12)
        cumulative_throughput_iter_per_s = cumulative_steps / max(cumulative_wall_s, 1e-12)
        est_wall_to_iter_milestone_h = (iter_milestone / max(cumulative_throughput_iter_per_s, 1e-12)) / 3600.0

        local_gpu_metrics = aggregate_local_gpu_metrics(metric_device_indices)
        gpu_metrics = {}
        for name, value in local_gpu_metrics.items():
            gpu_metrics[name] = reduce_scalar_mean(value, env, env.device)

        if is_main_process(env):
            best_acc = max(best_acc, val_acc)
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"train_loss={train_result.loss:.4f} train_acc={train_result.acc:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
                f"best={best_acc:.2f}% time={epoch_time:.1f}s "
                f"| throughput={kpis['throughput_samples_per_s']:.2f} samples/s "
                f"| stat_eff={kpis['stat_eff_loss_gain_per_sample']:.6f} loss/sample "
                f"| goodput={kpis['system_goodput_samples_per_s']:.2f} samples/s "
                f"| cum_wall={cumulative_wall_s/3600.0:.2f}h",
                flush=True,
            )
            if writer is not None:
                step = epoch + 1
                writer.add_scalar("train/loss", train_result.loss, step)
                writer.add_scalar("train/acc1", train_result.acc, step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                writer.add_scalar("val/loss", val_loss, step)
                writer.add_scalar("val/acc1", val_acc, step)
                writer.add_scalar("kpi/throughput_samples_per_s", kpis["throughput_samples_per_s"], step)
                writer.add_scalar("kpi/compute_utilization", kpis["compute_utilization"], step)
                writer.add_scalar("kpi/system_goodput_samples_per_s", kpis["system_goodput_samples_per_s"], step)
                writer.add_scalar("kpi/stat_eff_loss_gain_per_sample", kpis["stat_eff_loss_gain_per_sample"], step)
                writer.add_scalar("kpi/stat_goodput_loss_gain_per_s", kpis["stat_goodput_loss_gain_per_s"], step)
                writer.add_scalar("kpi/train_samples_per_epoch", train_result.samples, step)
                writer.add_scalar("kpi/train_steps_per_epoch", train_result.steps, step)
                writer.add_scalar("kpi/epoch_wall_time_s", epoch_time, step)
                writer.add_scalar("kpi/epoch_compute_time_s", compute_time, step)
                writer.add_scalar("kpi/throughput_iter_per_s", throughput_iter_per_s, step)
                writer.add_scalar("kpi/cumulative_wall_time_s", cumulative_wall_s, step)
                writer.add_scalar("kpi/cumulative_wall_time_h", cumulative_wall_s / 3600.0, step)
                writer.add_scalar("kpi/cumulative_samples", cumulative_samples, step)
                writer.add_scalar("kpi/cumulative_steps", cumulative_steps, step)
                writer.add_scalar("kpi/cumulative_throughput_samples_per_s", cumulative_throughput_samples_per_s, step)
                writer.add_scalar("kpi/cumulative_throughput_iter_per_s", cumulative_throughput_iter_per_s, step)
                writer.add_scalar("kpi/est_wall_time_to_iter_milestone_h", est_wall_to_iter_milestone_h, step)
                if wall_time_to_iter_milestone_h is not None:
                    writer.add_scalar("kpi/wall_time_to_iter_milestone_h", wall_time_to_iter_milestone_h, step)
                if loss_at_0 is not None:
                    writer.add_scalar("kpi/loss_at_0", loss_at_0, step)
                if loss_at_m1 is not None:
                    writer.add_scalar(f"kpi/loss_at_{sample_m1}_samples", loss_at_m1, step)
                if loss_at_m2 is not None:
                    writer.add_scalar(f"kpi/loss_at_{sample_m2}_samples", loss_at_m2, step)
                if loss_at_wall_milestone is not None:
                    writer.add_scalar(f"kpi/loss_at_{args.walltime_milestone_hours:g}h_wall_train", loss_at_wall_milestone, step)
                if val_loss_at_wall_milestone is not None:
                    writer.add_scalar(f"kpi/loss_at_{args.walltime_milestone_hours:g}h_wall_val", val_loss_at_wall_milestone, step)
                if not math.isnan(stat_eff_0_m1):
                    writer.add_scalar(f"kpi/stat_eff_0_to_{sample_m1}_samples", stat_eff_0_m1, step)
                    writer.add_scalar(f"kpi/goodput_0_to_{sample_m1}_samples", goodput_0_m1, step)
                if not math.isnan(stat_eff_m1_m2):
                    writer.add_scalar(f"kpi/stat_eff_{sample_m1}_to_{sample_m2}_samples", stat_eff_m1_m2, step)
                    writer.add_scalar(f"kpi/goodput_{sample_m1}_to_{sample_m2}_samples", goodput_m1_m2, step)
                for metric_name, metric_value in gpu_metrics.items():
                    writer.add_scalar(f"gpu/{metric_name}", metric_value, step)
                writer.flush()

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, best_acc, args, env)

    if writer is not None:
        writer.flush()
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
