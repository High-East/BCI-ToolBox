# BCI-ToolBox

## 1. Introduction
BCI-ToolBox is deep learning pipeline for motor-imagery classification.  
This repo contains five models: ShallowConvNet, DeepConvNet, EEGNet, FBCNet, BCI2021.  
(BCI2021 is not an official name.)

## 2. Installation

### Environment

- Python == 3.7.10
- PyTorch == 1.9.0
- mne == 0.23.0
- braindecode == 0.5.1
- CUDA == 11.0

### Create conda environment

```shell
conda install pytorch=1.9.0 cudatoolkit=11.1 -c pytorch -c nvidia
conda install numpy pandas matplotlib pyyaml ipywidgets
pip install torchinfo braindecode moabb mne
```

## 3. Directory structure
```shell
.
├── README.md
├── base
│   ├── constructor.py
│   └── layers.py
├── configs
│   ├── BCI2021
│   │   └── default.yaml
│   ├── DeepConvNet
│   │   └── default.yaml
│   ├── EEGNet
│   │   └── default.yaml
│   ├── FBCNet
│   │   └── default.yaml
│   ├── ShallowConvNet
│   │   └── default.yaml
│   └── demo
│       ├── arch.yaml
│       ├── bci2021.yaml
│       ├── test.yaml
│       ├── train.yaml
│       └── training_params.yaml
├── data_loader
│   ├── data_generator.py
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── bnci2014.py
│   │   ├── cho2017.py
│   │   ├── folder_dataset.py
│   │   ├── openbmi.py
│   │   └── tmp_dataset.py
│   └── transforms.py
├── main.py
├── models
│   ├── BCI2021
│   │   ├── BCI2021.py
│   │   └── __init__.py
│   ├── DeepConvNet
│   │   ├── DeepConvNet.py
│   │   └── __init__.py
│   ├── EEGNet
│   │   ├── EEGNet.py
│   │   └── __init__.py
│   ├── FBCNet
│   │   ├── FBCNet.py
│   │   └── __init__.py
│   ├── ShallowConvNet
│   │   ├── ShallowConvNet.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── model_builder.py
├── trainers
│   ├── __init__.py
│   ├── cls_trainer.py
│   └── trainer_maker.py
└── utils
    ├── calculator.py
    ├── painter.py
    └── utils.py

```

## 4. Dataset

- Use [braindecode](https://braindecode.org)

## 5. Get Started
- We use [W&B](https://wandb.ai/home) for experimental management tool.
- [Example for W&B Dashboard](https://wandb.ai/high-east/Demo/table?workspace=user-high-east)
- [Example for W&B Report](https://wandb.ai/high-east/Demo/reports/Report-sample--VmlldzoxNDAxMzY5)

### Create wandb_key.yaml file
- Create wandb_key.yaml file in configs directory.
    ```yaml
    # wandb_key.yaml
    key: WANDB API keys
    ```
- WANDB API keys can be obtained from your W&B account settings.

### train
**Use W&B**
```bash
python main.py --config_file=configs/demo/train.yaml
```

**Not use W&B**
```bash
python main.py --config_file=configs/demo/train.yaml --no_wandb
```

**USE GPU**
```bash
python main.py --config_file=configs/demo/train.yaml --device=0  # Use GPU 0
python main.py --config_file=configs/demo/train.yaml --device=1  # Use GPU 1
python main.py --config_file=configs/demo/train.yaml --device=2  # Use GPU 2
```
- GPU numbers depend on your server.

**USE Sweep**
```yaml
# W&B
sweep_file: configs/demo/training_params.yaml
project: Demo
tags: [train]
```
- Add this block to config file for finding training parameters.

```yaml
# W&B
sweep_file: configs/demo/arch.yaml
sweep_type: arch
project: Demo
tags: [train]
```
- Add this block to config file for finding model architecture.

### test
```bash
python main.py --config_file=configs/demo/test.yaml
```

## 5. References
- ShallowConvNet [[Paper]](https://arxiv.org/pdf/1703.05051.pdf) [[Code]](https://github.com/braindecode/braindecode/blob/master/braindecode/models/shallow_fbcsp.py)
- DeepConvNet [[Paper]](https://arxiv.org/pdf/1703.05051.pdf)  [[Code]](https://github.com/braindecode/braindecode/blob/master/braindecode/models/deep4.py)
- EEGNet [[Paper]](https://arxiv.org/abs/1611.08024) [[Code]](https://github.com/braindecode/braindecode/blob/master/braindecode/models/eegnet.py)
- FBCNet [[Paper]](https://arxiv.org/abs/2104.01233) [[Code]](https://github.com/ravikiran-mane/FBCNet)
- BCI2021 [[Paper]](https://ieeexplore.ieee.org/document/9385293) [[Code]](https://github.com/High-East/Attention-based-spatio-temporal-spectral-feature-learning-for-subject-specific-EEG-classification)
