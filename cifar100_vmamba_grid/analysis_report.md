# CIFAR100 VMamba Grid Analysis

## Scope

- Root: `cifar100_vmamba_grid`
- Parsed runs: **28**
- Settings: `ddp, pipeline, scan_tp`
- Batch sizes: `64, 128, 256, 512`
- GPU counts: `1, 2, 4`
- Baseline for +/- comparison: `ddp_g1` (same batch size)

## Global Best Metrics

| Metric | Run | Value |
|---|---|---:|
| Best validation accuracy | `scan_tp g4 bs64` | 70.29% |
| Best mean throughput | `ddp g4 bs512` | 5586.70 samples/s |
| Best mean goodput | `ddp g4 bs512` | 4795.89 samples/s |
| Fastest mean epoch time | `ddp g4 bs512` | 8.85 s |

## Aggregate by Setting

| Setting | Runs | Mean Best Acc (%) | Max Best Acc (%) | Mean Throughput | Mean Goodput | Mean Epoch Time (s) | Mean Wall Time (h) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 12 | 65.90 | 68.28 | 2788.52 | 2497.91 | 22.65 | 0.31 |
| `pipeline` | 8 | 67.39 | 69.26 | 1358.33 | 1240.46 | 38.01 | 0.53 |
| `scan_tp` | 8 | 68.22 | 70.29 | 816.88 | 676.71 | 64.59 | 0.90 |

## Aggregate by Setting + GPU Count

| Setting | GPUs | Runs | Mean Best Acc (%) | Max Best Acc (%) | Mean Throughput | Mean Goodput | Mean Epoch Time (s) | Mean Wall Time (h) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | 4 | 67.23 | 68.28 | 1434.65 | 1302.15 | 35.09 | 0.49 |
| `ddp` | 2 | 4 | 66.14 | 68.02 | 2464.79 | 2234.02 | 20.96 | 0.29 |
| `ddp` | 4 | 4 | 64.34 | 67.03 | 4466.12 | 3957.55 | 11.89 | 0.16 |
| `pipeline` | 2 | 4 | 67.68 | 69.26 | 1382.35 | 1265.16 | 37.59 | 0.52 |
| `pipeline` | 4 | 4 | 67.09 | 68.25 | 1334.31 | 1215.76 | 38.43 | 0.54 |
| `scan_tp` | 2 | 4 | 68.22 | 70.20 | 939.50 | 782.61 | 55.57 | 0.77 |
| `scan_tp` | 4 | 4 | 68.22 | 70.29 | 694.25 | 570.81 | 73.60 | 1.02 |

## Technique-Level +/- vs Baseline (`ddp_g1`, same batch)

| Technique | Runs | Acc Delta (pp) | Acc Delta (%) | Throughput Delta (%) | Goodput Delta (%) | Epoch Time Delta (%) | Wall Time Delta (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 8 | -1.99 | -2.99 | +138.75 | +135.45 | -53.79 | -54.19 |
| `pipeline` | 8 | +0.16 | +0.23 | -6.01 | -5.39 | +7.38 | +7.50 |
| `scan_tp` | 8 | +0.99 | +1.45 | -43.51 | -48.52 | +82.46 | +82.24 |

## Per-Batch Bests

| Batch | Best Acc Run | Best Acc (%) | Best Throughput Run | Throughput | Fastest Epoch Run | Epoch Time (s) |
|---:|---|---:|---|---:|---|---:|
| 64 | `scan_tp g4` | 70.29 | `ddp g4` | 2879.40 | `ddp g4` | 17.37 |
| 128 | `scan_tp g2` | 69.80 | `ddp g4` | 4273.04 | `ddp g4` | 11.68 |
| 256 | `scan_tp g2` | 67.64 | `ddp g4` | 5125.33 | `ddp g4` | 9.64 |
| 512 | `scan_tp g4` | 65.68 | `ddp g4` | 5586.70 | `ddp g4` | 8.85 |

## All Experiments Grouped by Batch Size (+/- vs Baseline)

### Batch Size 64

Baseline: `ddp g1 bs64` | best_acc=68.28% | mean_throughput=1238.67 | mean_goodput=1135.82 | mean_epoch_time=40.40s | wall=0.56h

| Mode | GPUs | Best Acc (%) | Δ Acc (pp) | Δ Acc (%) | Mean Throughput | Δ Throughput (%) | Mean Goodput | Δ Goodput (%) | Mean Epoch Time (s) | Δ Epoch Time (%) | Wall Time (h) | Δ Wall Time (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | 68.28 | +0.00 | +0.00 | 1238.67 | +0.00 | 1135.82 | +0.00 | 40.40 | +0.00 | 0.56 | +0.00 |
| `ddp` | 2 | 68.02 | -0.26 | -0.38 | 1778.46 | +43.58 | 1646.12 | +44.93 | 28.13 | -30.38 | 0.39 | -30.36 |
| `ddp` | 4 | 67.03 | -1.25 | -1.83 | 2879.40 | +132.46 | 2663.37 | +134.49 | 17.37 | -57.00 | 0.24 | -57.14 |
| `pipeline` | 2 | 69.26 | +0.98 | +1.44 | 983.62 | -20.59 | 908.49 | -20.01 | 50.85 | +25.86 | 0.71 | +26.79 |
| `pipeline` | 4 | 68.25 | -0.03 | -0.04 | 1010.39 | -18.43 | 932.65 | -17.89 | 49.53 | +22.59 | 0.69 | +23.21 |
| `scan_tp` | 2 | 70.20 | +1.92 | +2.81 | 650.38 | -47.49 | 521.78 | -54.06 | 76.88 | +90.28 | 1.07 | +91.07 |
| `scan_tp` | 4 | 70.29 | +2.01 | +2.94 | 534.41 | -56.86 | 428.75 | -62.25 | 93.55 | +131.55 | 1.30 | +132.14 |

### Batch Size 128

Baseline: `ddp g1 bs128` | best_acc=67.99% | mean_throughput=1409.09 | mean_goodput=1285.06 | mean_epoch_time=35.44s | wall=0.49h

| Mode | GPUs | Best Acc (%) | Δ Acc (pp) | Δ Acc (%) | Mean Throughput | Δ Throughput (%) | Mean Goodput | Δ Goodput (%) | Mean Epoch Time (s) | Δ Epoch Time (%) | Wall Time (h) | Δ Wall Time (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | 67.99 | +0.00 | +0.00 | 1409.09 | +0.00 | 1285.06 | +0.00 | 35.44 | +0.00 | 0.49 | +0.00 |
| `ddp` | 2 | 67.34 | -0.65 | -0.96 | 2407.96 | +70.89 | 2196.91 | +70.96 | 20.77 | -41.40 | 0.29 | -40.82 |
| `ddp` | 4 | 66.32 | -1.67 | -2.46 | 4273.04 | +203.25 | 3853.32 | +199.85 | 11.68 | -67.05 | 0.16 | -67.35 |
| `pipeline` | 2 | 68.64 | +0.65 | +0.96 | 1348.46 | -4.30 | 1238.83 | -3.60 | 37.12 | +4.74 | 0.52 | +6.12 |
| `pipeline` | 4 | 67.63 | -0.36 | -0.53 | 1334.20 | -5.31 | 1220.17 | -5.05 | 37.48 | +5.75 | 0.52 | +6.12 |
| `scan_tp` | 2 | 69.80 | +1.81 | +2.66 | 922.89 | -34.50 | 756.47 | -41.13 | 54.12 | +52.71 | 0.75 | +53.06 |
| `scan_tp` | 4 | 69.60 | +1.61 | +2.37 | 684.85 | -51.40 | 556.58 | -56.69 | 72.92 | +105.76 | 1.01 | +106.12 |

### Batch Size 256

Baseline: `ddp g1 bs256` | best_acc=67.03% | mean_throughput=1506.27 | mean_goodput=1362.72 | mean_epoch_time=33.18s | wall=0.46h

| Mode | GPUs | Best Acc (%) | Δ Acc (pp) | Δ Acc (%) | Mean Throughput | Δ Throughput (%) | Mean Goodput | Δ Goodput (%) | Mean Epoch Time (s) | Δ Epoch Time (%) | Wall Time (h) | Δ Wall Time (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | 67.03 | +0.00 | +0.00 | 1506.27 | +0.00 | 1362.72 | +0.00 | 33.18 | +0.00 | 0.46 | +0.00 |
| `ddp` | 2 | 65.95 | -1.08 | -1.61 | 2736.89 | +81.70 | 2468.94 | +81.18 | 18.17 | -45.23 | 0.25 | -45.65 |
| `ddp` | 4 | 63.46 | -3.57 | -5.33 | 5125.33 | +240.27 | 4517.64 | +231.52 | 9.64 | -70.94 | 0.13 | -71.74 |
| `pipeline` | 2 | 67.45 | +0.42 | +0.63 | 1564.39 | +3.86 | 1428.84 | +4.85 | 31.95 | -3.72 | 0.44 | -4.35 |
| `pipeline` | 4 | 67.03 | +0.00 | +0.00 | 1456.64 | -3.29 | 1320.55 | -3.09 | 34.33 | +3.44 | 0.48 | +4.35 |
| `scan_tp` | 2 | 67.64 | +0.61 | +0.91 | 1060.85 | -29.57 | 893.26 | -34.45 | 47.08 | +41.86 | 0.65 | +41.30 |
| `scan_tp` | 4 | 67.29 | +0.26 | +0.39 | 763.92 | -49.28 | 633.46 | -53.52 | 65.37 | +96.99 | 0.91 | +97.83 |

### Batch Size 512

Baseline: `ddp g1 bs512` | best_acc=65.62% | mean_throughput=1584.56 | mean_goodput=1425.00 | mean_epoch_time=31.35s | wall=0.44h

| Mode | GPUs | Best Acc (%) | Δ Acc (pp) | Δ Acc (%) | Mean Throughput | Δ Throughput (%) | Mean Goodput | Δ Goodput (%) | Mean Epoch Time (s) | Δ Epoch Time (%) | Wall Time (h) | Δ Wall Time (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | 65.62 | +0.00 | +0.00 | 1584.56 | +0.00 | 1425.00 | +0.00 | 31.35 | +0.00 | 0.44 | +0.00 |
| `ddp` | 2 | 63.25 | -2.37 | -3.61 | 2935.86 | +85.28 | 2624.11 | +84.15 | 16.77 | -46.51 | 0.23 | -47.73 |
| `ddp` | 4 | 60.56 | -5.06 | -7.71 | 5586.70 | +252.57 | 4795.89 | +236.55 | 8.85 | -71.77 | 0.12 | -72.73 |
| `pipeline` | 2 | 65.38 | -0.24 | -0.37 | 1632.93 | +3.05 | 1484.48 | +4.17 | 30.45 | -2.88 | 0.42 | -4.55 |
| `pipeline` | 4 | 65.45 | -0.17 | -0.26 | 1536.02 | -3.06 | 1389.68 | -2.48 | 32.38 | +3.28 | 0.45 | +2.27 |
| `scan_tp` | 2 | 65.22 | -0.40 | -0.61 | 1123.89 | -29.07 | 958.94 | -32.71 | 44.20 | +40.98 | 0.61 | +38.64 |
| `scan_tp` | 4 | 65.68 | +0.06 | +0.09 | 793.84 | -49.90 | 664.45 | -53.37 | 62.58 | +99.58 | 0.87 | +97.73 |


## TensorBoard Summary

| Setting | Mean GPU Util (%) | Mean GPU Mem Util (%) | Median Epoch to 60% Val Acc | Median Epoch to 65% Val Acc |
|---|---:|---:|---:|---:|
| `ddp` | 65.1 | 24.4 | 16.0 | 29 |
| `pipeline` | 27.7 | 11.0 | 16.0 | 29.0 |
| `scan_tp` | 64.2 | 10.9 | 15.0 | 25.0 |

## Findings

1. `scan_tp` gives the highest accuracy overall (best run: `g4 bs64`, 70.29%), but has the lowest throughput and longest wall-clock time.
2. `ddp` is best for speed/scaling; accuracy degrades as total global batch and GPU count increase, especially at `bs512`.
3. `pipeline` is the most balanced mode: accuracy near baseline with moderate throughput penalty relative to `ddp`.
4. In this grid, increasing GPUs from 2 to 4 helps `ddp`, but does not improve `pipeline` throughput and reduces `scan_tp` throughput.


## Notes

- Throughput/Goodput/Time metrics are epoch means across 50 epochs.
- +/- deltas are always relative to `ddp_g1` at the same batch size.

## Statistical Efficiency Metrics

- `stat_eff` is from logs (`loss/sample`).
- `stat_goodput` is computed as epoch mean of `stat_eff * throughput` (`loss/s`) to combine statistical + system speed.
- Relative `%` for `stat_eff` is not stable (values are near zero and sign-changing), so deltas below are absolute.

### Aggregate by Setting

| Setting | Mean stat_eff (loss/sample) | Mean \|stat_eff\| | Mean stat_goodput (loss/s) |
|---|---:|---:|---:|
| `ddp` | +0.00000043 | 0.00000184 | +0.00116 |
| `pipeline` | +0.00000029 | 0.00000181 | +0.00033 |
| `scan_tp` | +0.00000028 | 0.00000184 | +0.00021 |

### All Experiments by Batch (Absolute Delta vs Baseline `ddp_g1`)

#### Batch Size 64

Baseline: `ddp g1 bs64` | mean_stat_eff=+0.00000008 | mean|stat_eff|=0.00000284 | mean_stat_goodput=+0.00002

| Mode | GPUs | Mean stat_eff | Δ stat_eff | Mean \|stat_eff\| | Δ \|stat_eff\| | Mean stat_goodput | Δ stat_goodput |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | +0.00000008 | +0.00000000 | 0.00000284 | +0.00000000 | +0.00002 | +0.00000 |
| `ddp` | 2 | +0.00000038 | +0.00000030 | 0.00000246 | -0.00000038 | +0.00062 | +0.00060 |
| `ddp` | 4 | +0.00000040 | +0.00000032 | 0.00000232 | -0.00000052 | +0.00095 | +0.00093 |
| `pipeline` | 2 | -0.00000002 | -0.00000010 | 0.00000274 | -0.00000010 | -0.00009 | -0.00011 |
| `pipeline` | 4 | +0.00000040 | +0.00000032 | 0.00000244 | -0.00000040 | +0.00031 | +0.00029 |
| `scan_tp` | 2 | -0.00000014 | -0.00000022 | 0.00000266 | -0.00000018 | -0.00012 | -0.00014 |
| `scan_tp` | 4 | +0.00000042 | +0.00000034 | 0.00000238 | -0.00000046 | +0.00020 | +0.00019 |

#### Batch Size 128

Baseline: `ddp g1 bs128` | mean_stat_eff=-0.00000002 | mean|stat_eff|=0.00000158 | mean_stat_goodput=-0.00007

| Mode | GPUs | Mean stat_eff | Δ stat_eff | Mean \|stat_eff\| | Δ \|stat_eff\| | Mean stat_goodput | Δ stat_goodput |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | -0.00000002 | +0.00000000 | 0.00000158 | +0.00000000 | -0.00007 | +0.00000 |
| `ddp` | 2 | +0.00000084 | +0.00000086 | 0.00000172 | +0.00000014 | +0.00188 | +0.00195 |
| `ddp` | 4 | +0.00000062 | +0.00000064 | 0.00000198 | +0.00000040 | +0.00213 | +0.00220 |
| `pipeline` | 2 | +0.00000038 | +0.00000040 | 0.00000182 | +0.00000024 | +0.00042 | +0.00049 |
| `pipeline` | 4 | +0.00000028 | +0.00000030 | 0.00000160 | +0.00000002 | +0.00027 | +0.00034 |
| `scan_tp` | 2 | +0.00000026 | +0.00000028 | 0.00000170 | +0.00000012 | +0.00019 | +0.00026 |
| `scan_tp` | 4 | +0.00000030 | +0.00000032 | 0.00000186 | +0.00000028 | +0.00018 | +0.00025 |

#### Batch Size 256

Baseline: `ddp g1 bs256` | mean_stat_eff=+0.00000058 | mean|stat_eff|=0.00000158 | mean_stat_goodput=+0.00084

| Mode | GPUs | Mean stat_eff | Δ stat_eff | Mean \|stat_eff\| | Δ \|stat_eff\| | Mean stat_goodput | Δ stat_goodput |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | +0.00000058 | +0.00000000 | 0.00000158 | +0.00000000 | +0.00084 | +0.00000 |
| `ddp` | 2 | +0.00000038 | -0.00000020 | 0.00000146 | -0.00000012 | +0.00086 | +0.00002 |
| `ddp` | 4 | +0.00000056 | -0.00000002 | 0.00000164 | +0.00000006 | +0.00235 | +0.00151 |
| `pipeline` | 2 | +0.00000044 | -0.00000014 | 0.00000160 | +0.00000002 | +0.00059 | -0.00025 |
| `pipeline` | 4 | +0.00000018 | -0.00000040 | 0.00000134 | -0.00000024 | +0.00015 | -0.00069 |
| `scan_tp` | 2 | +0.00000042 | -0.00000016 | 0.00000154 | -0.00000004 | +0.00041 | -0.00044 |
| `scan_tp` | 4 | +0.00000036 | -0.00000022 | 0.00000160 | +0.00000002 | +0.00025 | -0.00060 |

#### Batch Size 512

Baseline: `ddp g1 bs512` | mean_stat_eff=+0.00000036 | mean|stat_eff|=0.00000148 | mean_stat_goodput=+0.00053

| Mode | GPUs | Mean stat_eff | Δ stat_eff | Mean \|stat_eff\| | Δ \|stat_eff\| | Mean stat_goodput | Δ stat_goodput |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ddp` | 1 | +0.00000036 | +0.00000000 | 0.00000148 | +0.00000000 | +0.00053 | +0.00000 |
| `ddp` | 2 | +0.00000042 | +0.00000006 | 0.00000138 | -0.00000010 | +0.00107 | +0.00054 |
| `ddp` | 4 | +0.00000054 | +0.00000018 | 0.00000158 | +0.00000010 | +0.00275 | +0.00221 |
| `pipeline` | 2 | +0.00000028 | -0.00000008 | 0.00000148 | +0.00000000 | +0.00038 | -0.00015 |
| `pipeline` | 4 | +0.00000042 | +0.00000006 | 0.00000146 | -0.00000002 | +0.00057 | +0.00004 |
| `scan_tp` | 2 | +0.00000034 | -0.00000002 | 0.00000150 | +0.00000002 | +0.00035 | -0.00018 |
| `scan_tp` | 4 | +0.00000030 | -0.00000006 | 0.00000146 | -0.00000002 | +0.00022 | -0.00031 |

