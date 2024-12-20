# CutMix-CD: Advancing Semi-Supervised Change Detection via Mixed Sample Consistency

This repocitory contains the official implementation of our paper:  **CutMix-CD: Advancing Semi-Supervised Change Detection via Mixed Sample Consistency**.

## :speech_balloon: Requirements
This repo was tested with python 3.8, torch 1.7.1.
```bash
pip install -r requirements.txt
```

## :speech_balloon: Data preparation

Download [`LEVIR-CD`](https://justchenhao.github.io/LEVIR/) and [`S2Looking`](https://github.com/S2Looking/Dataset) datasets. The preprocess is introduced in the paper.

Check the file `loaders/datasets.py` and you can adjust it appropriately for your own dataset.

Modify the argument `--data_root` in training script.

## :speech_balloon: Training

We provide training scripts on LEVIR-CD and S2looking datasets. 

To train the model, first download ImageNet-pretrained `3x3resnet50-imagenet.pth` file and save it to the path "models/backbones/pretrained".

Detailed training arguments are described in the training script. You can simply train a model by:

```bash
python train_LEVIR.py
```
During the training, the losses and metrics are reported in the file `train.log` saved in `--log`. 



## :speech_balloon: Inference

```bash
python inference_LEVIR.py
```

Evaluation metrics are saved in the file `test.log` saved in `--log`.

## :speech_balloon: Citation

If you find this repo useful for your research, please consider citing the paper as follows:

```
```
#### Acknowledgements
Thanks to the following open source efforts:
- [cutmix-semisup-seg](https://github.com/Britefury/cutmix-semisup-seg).
- [SemiCD](https://github.com/wgcban/SemiCD).
- [FPA-SSCD](https://github.com/zxt9/FPA-SSCD).
