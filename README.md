# CutMix-CD: Advancing Semi-Supervised Change Detection via Mixed Sample Consistency

This repocitory contains the official implementation of our [`paper`](https://doi.org/10.1109/TGRS.2024.3520630): **CutMix-CD: Advancing Semi-Supervised Change Detection via Mixed Sample Consistency**.

<img width="1179" alt="image" src="https://github.com/user-attachments/assets/717388c3-c9b3-4659-a5b7-ba29cc6c47de" />


## :speech_balloon: Requirements
This repo was tested with python 3.8, torch 1.7.1.
```bash
pip install -r requirements.txt
```

## :speech_balloon: Data preparation

Download [`LEVIR-CD`](https://justchenhao.github.io/LEVIR/) and [`S2Looking`](https://github.com/S2Looking/Dataset) datasets. The data preprocessing is introduced in the paper.

Check the file `loaders/datasets.py` and you can adjust it appropriately for your own dataset.

Modify the argument `--data_root` in training scripts.

## :speech_balloon: Training

We provide training scripts on LEVIR-CD and S2looking datasets. 

To train the model, first download ImageNet-pretrained [`3x3resnet50-imagenet.pth`](https://github.com/yassouali/CCT/releases/download/v0.1/3x3resnet50-imagenet.pth) file and save it to the path `models/backbones/pretrained`.

Detailed training arguments are described in the training script. You can simply train a model by:

```bash
python train_LEVIR.py
```
During the training, the losses and metrics are reported in the file `train.log` saved in `--log`. The best model is saved as `best.pth`.



## :speech_balloon: Inference

Set the path `--weight_path` of the best model in inference script and evaluate on the test set by:

```bash
python inference_LEVIR.py
```

Evaluation metrics are saved in the file `test.log` saved in `--log`.

## :speech_balloon: Citation

If you find this repo useful for your research, please consider citing the paper as follows:

```
@ARTICLE{10810476,
  author={Shu, Qidi and Zhu, Xiaolin and Wan, Luoma and Zhao, Shuheng and Liu, Denghong and Peng, Longkang and Chen, Xiaobei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={CutMix-CD: Advancing Semi-Supervised Change Detection via Mixed Sample Consistency}, 
  year={2024},
  doi={10.1109/TGRS.2024.3520630}}
```
#### Acknowledgements
Thanks to the following open source efforts:
- [cutmix-semisup-seg](https://github.com/Britefury/cutmix-semisup-seg).
- [SemiCD](https://github.com/wgcban/SemiCD).
- [FPA-SSCD](https://github.com/zxt9/FPA-SSCD).
