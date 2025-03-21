# Skeleton-Guided Diffusion Model for Accurate Foot X-ray Synthesis in Hallux Valgus Diagnosis
Skeleton-Constrained Conditional Diffusion Model (SCCDM) in the diff-domain for foot X-ray synthesis, transforming natural foot images into X-ray images while ensuring logical skeletal structures through multi-scale feature extraction and self-attention.

The code in this toolbox implements the ["Skeleton-Guided Diffusion Model for Accurate Foot X-ray Synthesis in Hallux Valgus Diagnosis"]()


The Overview of the Skeletal-Constrained Conditional Diffusion Model is as follows.
![Overview](/static/overview.png)


Installation
---------------------

Ensure you have the following dependencies installed:

- Python 3.8+
- PyTorch
- [Additional dependencies from requirements.txt]

To install, run:

```bash
git clone https://github.com/midisec/SCCDM
pip3 install requirements.txt
```

## Usage

### Prepare Your Dataset

To prepare the dataset, organize your data into multiple subfolders, where each subfolder represents a single sample. Each subfolder should contain **two PNG images**:

- One **original image** (filename starting with `"original"`).
- One **xray image** (filename starting with `"xray"`).

```
dataset/
│── sample1/
│   ├── original_1.png
│   ├── xray_1.png
│
│── sample2/
│   ├── original_2.png
│   ├── xray_2.png
│
│── sample3/
│   ├── original_3.png
│   ├── xray_3.png
│
│── ...
```

### Training the Model
Setting the `config.py `
```bash
python train.py
```

###  Evaluating the Model

```bash
python evaluate.py
```

##  Citation
If you find this work useful, please cite our paper:
```
...
```

Licensing
---------

Copyright (C) 2025 Midi Wan

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.