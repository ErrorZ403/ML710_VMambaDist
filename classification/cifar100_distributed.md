# CIFAR-100 Distributed VMamba Training

This training entrypoint is implemented in:

- `classification/cifar100_dist_train.py`

It supports three execution modes:

1. `ddp`: standard data parallel training with `torchrun`
2. `scan_tp`: scan-branch tensor/model parallelism (SS2D branches split by rank)
3. `pipeline`: pipeline-style model parallelism (whole VSS blocks on different GPUs)

## 1) Simple DDP

```bash
torchrun --nproc_per_node=4 classification/cifar100_dist_train.py \
  --parallel-mode ddp \
  --data-dir ./data \
  --download \
  --epochs 200 \
  --batch-size 128
```

Notes:

- `--batch-size` is per-rank in DDP.
- Use `--dims`, `--depths`, `--drop-path`, etc. to scale model size.

## 2) Scan-Branch Tensor Parallelism

```bash
torchrun --nproc_per_node=4 classification/cifar100_dist_train.py \
  --parallel-mode scan_tp \
  --data-dir ./data \
  --epochs 200 \
  --batch-size 128
```

Notes:

- SS2D scan branches are partitioned by rank (`branch_id % world_size` ownership).
- This mode keeps branch parameters synchronized after every optimizer step.
- This mode expects multi-rank launch (`nproc_per_node > 1`).

## 3) Pipeline Parallelism (Whole VSS Blocks)

```bash
python classification/cifar100_dist_train.py \
  --parallel-mode pipeline \
  --pipeline-devices 0,1,2,3 \
  --data-dir ./data \
  --epochs 200 \
  --batch-size 256
```

Notes:

- Run pipeline mode as a single process.
- `--pipeline-devices` controls the device sequence used for block placement.

## Useful Flags

```bash
--optimizer adamw|sgd
--lr 5e-4
--weight-decay 0.05
--label-smoothing 0.1
--disable-amp
--save-every 10
--eval-only
--tensorboard
--tensorboard-dir /path/to/tb_logs
--sample-milestones 100000,500000
--iter-milestone 100000
--walltime-milestone-hours 10
```

## Training KPIs

Each epoch now reports and logs:

- `throughput_samples_per_s`: processed training samples per second
- `stat_eff_loss_gain_per_sample`: `(first_step_loss - last_step_loss) / samples`
- `system_goodput_samples_per_s`: `throughput * compute_utilization`
- `stat_goodput_loss_gain_per_s`: `throughput * stat_eff_loss_gain_per_sample`
- `epoch_wall_time_s`, `cumulative_wall_time_h`
- `loss_at_100000_samples`, `loss_at_500000_samples`
- `loss_at_10h_wall_train`, `loss_at_10h_wall_val`
- `stat_eff_0_to_100000_samples`, `stat_eff_100000_to_500000_samples`
- `goodput_0_to_100000_samples`, `goodput_100000_to_500000_samples`
- `throughput_iter_per_s`, `cumulative_throughput_iter_per_s`

GPU telemetry (per epoch, averaged across local GPUs and ranks):

- `gpu/gpu_utilization_pct`
- `gpu/gpu_memory_utilization_pct`
- `gpu/gpu_mem_peak_reserved_gb`
- `gpu/gpu_mem_peak_util_pct`
- `gpu/gpu_power_w`
- `gpu/gpu_temperature_c`

TensorBoard tags:

- `train/*`, `val/*`
- `kpi/*`

Example:

```bash
torchrun --nproc_per_node=4 classification/cifar100_dist_train.py \
  --parallel-mode ddp \
  --data-dir ./data \
  --epochs 200 \
  --batch-size 128 \
  --tensorboard \
  --tensorboard-dir ./output/cifar100_vmamba/tensorboard
```

## Sequential Grid Scripts (50 epochs)

Scripts are in `classification/scripts/`:

- `run_ddp_grid_1gpu.sh`
- `run_ddp_grid_2gpu.sh`
- `run_ddp_grid_4gpu.sh`
- `run_ddp_grid_all.sh`
- `run_scan_tp_grid_2gpu.sh`
- `run_scan_tp_grid_4gpu.sh`
- `run_scan_tp_grid_all.sh`
- `run_pipeline_grid_2gpu.sh`
- `run_pipeline_grid_4gpu.sh`
- `run_pipeline_grid_all.sh`
- `run_all_parallel_strategies_grid.sh`

Each script runs batch sizes `64, 128, 256, 512` with `epochs=50`.

GPU-count support by strategy:

- `ddp`: `1, 2, 4`
- `scan_tp`: `2, 4` (requires `nproc_per_node > 1`)
- `pipeline`: `2, 4` (single-process model-parallel)

Example:

```bash
bash classification/scripts/run_ddp_grid_all.sh /path/to/datasets
```

All strategies sequentially:

```bash
bash classification/scripts/run_all_parallel_strategies_grid.sh /path/to/datasets
```
