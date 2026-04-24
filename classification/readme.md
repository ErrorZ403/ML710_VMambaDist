## origins

based on https://github.com/microsoft/Swin-Transformer#20240103

`main.py` and `utils/utils_ema.py` is modified from https://github.com/microsoft/Swin-Transformer#20240103, based on https://github.com/facebookresearch/ConvNeXt#20240103

## cifar100 distributed training

See [`cifar100_distributed.md`](./cifar100_distributed.md) for:

- VMamba CIFAR-100 image classification training
- Simple DDP
- Scan-branch tensor/model parallelism
- VSS-block pipeline parallelism
