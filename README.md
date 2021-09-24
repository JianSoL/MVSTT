# Dynamic Spatial-Temporal Transformer Fusion Network for Traffic Forecasting
<p align="center">
  <img width="800" height="800" src=./documents/model.png>
</p>


## Requirements
- python 3.7
- pytorch
- numpy
- pandas
## Data Preparation
DSTTFN is implemented on those several public traffic datasets.
- **PEMS03**, **PEMS04**, **PEMS07** and **PEMS08** from [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).
## Model Train
PEMS03, PEMS04, PEMS07, PEMS08:
```
python train.py --dataset PEMS08
```



## Model Test
PEMS03, PEMS04, PEMS07, PEMS08:
```
python test.py --dataset PEMS08
```
