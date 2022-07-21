# MVSTT: A Multi-View Spatial-Temporal Transformer Network for Traffic Forecasting
<p align="center">
  <img width="1000"  src=./model/model.png>
</p>


## Requirements
- python 3.7
- pytorch
- numpy
- pandas
## Data Preparation
MVSTT is implemented on those several public traffic datasets.
- **PEMS03**, **PEMS04**, **PEMS07** and **PEMS08** from [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).
- Due to the size limitation of the dataset, you can access our dataset in Google Cloud Drive.[PEMS](https://drive.google.com/drive/folders/1wxNZtR_a8uYm7E-JT1qIwEWNUehlQ6xM?usp=sharing)
## Model Train
PEMS03, PEMS04, PEMS07, PEMS08:
```
python train.py --dataset PEMS08
```



## Model Test
The pre-trained model is available in Google Cloud Drive.[pre-trained model](https://drive.google.com/drive/folders/1SoO00z2BO_9sbZMNh2lx9WIanlOZb6B7?usp=sharing)

PEMS03, PEMS04, PEMS07, PEMS08:
```
python test.py --dataset PEMS08
```
