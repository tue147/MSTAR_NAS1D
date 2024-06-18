# MSTAR_NAS1D

## About

This is Neural architecture search (NAS) for timeseries classification

## Installation

### Prerequisites

#### Hardware
GPU that has CUDA supports
#### Software
Conda

### Environment setup
- Python == 3.11.1
```bash
conda create -n MSTAR python==3.11.1
```
- Packages
```bash
pip install -r requirements.txt
``` 

## Training
```bash
python train_NAS.py
```
The output will be saved in the working directory.

## Results
You can find our pre-trained results on 4 datasets in the Trained folders. \
All the files were saved using *torch.save* function. \
Use *torch.load* function to view.

## Our team

- **[Cao Minh Tue](https://github.com/tue147)** - *Leader*
- **[Nguyen Dang Duong](https://github.com/Mr-Duo)** - *Contributor*
- **[Nghiem Minh Hieu](https://github.com/gitgud8055)** - *Contributor*
