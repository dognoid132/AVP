
# Attentional Graph Neural Network for Parking Slot Detection

"yjq 2026/1/8交接仝科"

## Requirements

- python 3.6

- pytorch 1.4+

- other requirements: `pip install -r requirements.txt`

## Pretrained models

Two pre-trained models can be downloaded with following links.

| Link      | Code | Description |
| ----------- | ---- | ----------- |
| [Model0](https://pan.baidu.com/s/137ZHZnsEfyaO4yaa5YoBIQ) | bc0a | Trained with ps2.0 subset as in [1]|
| [Model1](https://pan.baidu.com/s/1qogTCwtjGEtR0y-PB4Ibmg)   | pgig  | Trained with full ps2.0 dataset      |

## Prepare data

The original ps2.0 data and label can be found [here](https://github.com/Teoge/DMPR-PS). Extract and organize as follows:

***使用ps1.0的架构格式***
```
├── datasets
│   └── parking_slot
│       ├── ps_json_label 
│       ├── testing
│       └── training
```
## Train & Test

Export current directory to `PYTHONPATH`:

```
export PYTHONPATH=`pwd`
( 切换到当前工作目录 )
```

- demo

```
python3 tools/demo.py -c config/ps_gat.yaml -m retrain_weight/checkpoint_epoch_210.pth
```

- train

```
python3 tools/train.py -c config/ps_gat.yaml
```

- test

```
python3 tools/test.py -c config/ps_gat.yaml -m retrain_weight/checkpoint_epoch_210.pth
```

# 修改
图像增强，loss，多尺度融合