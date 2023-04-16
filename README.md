![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

> This repository contains the official implementation code of the paper Transformer-based Feature Reconstruction Network for Robust Multimodal Sentiment Analysis, accepted at ACMMM 2021.

**Note:** We strongly recommend that you browse the overall structure of our code at first. If you have any question, feel free to contact us.

## Support Models

In this framework, we support the following methods:

|     Type    |   Model Name      |     From                |
|:-----------:|:----------------:|:------------------------:|
| Baselines |[TFN](models/singleTask/TFN.py)|[Tensor-Fusion-Network](https://github.com/A2Zadeh/TensorFusionNetwork)|
| Baselines |[MulT](models/singleTask/MulT.py)(without CTC) |[Multimodal-Transformer](https://github.com/yaohungt/Multimodal-Transformer)|
| Baselines |[MISA](models/singleTask/MISA.py) |[MISA](https://github.com/declare-lab/MISA)|
| Missing-Task  |[TFR-Net](models/missingTask/TFR_NET)|      [TFR-Net](https://github.com/Columbine21/TFR-Net)  |

## Usage


- Clone this repo and install requirements.
```
git clone https://github.com/Columbine21/TFR-Net.git
cd TFR-Net
```

### Data Preprocessing

1. Download datasets from the following links.

- MOSI
> download from [CMU-MultimodalSDK](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/)

- SIMS
> download from [Baidu Yun Disk](https://pan.baidu.com/share/init?surl=XmobKHUqnXciAm7hfnj2gg) [code: `mfet`] or [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk)  
> **Notes:** Please download new features `unaligned_39.pkl` from [Baidu Yun Disk](https://pan.baidu.com/share/init?surl=XmobKHUqnXciAm7hfnj2gg) [code: `mfet`] or [Google Drive](https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk), which is compatible with our new code structure. The `md5 code` is `a5b2ed3844200c7fb3b8ddc750b77feb`.

1. Download [Bert-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) from [Google-Bert](https://github.com/google-research/bert).  

2. Convert Tensorflow into pytorch using [transformers-cli](https://huggingface.co/transformers/converting_tensorflow_models.html)  

3. Install python dependencies

4. Organize features and save them as pickle files with the following structure.

> **Notes:** `unaligned_39.pkl` is compatible with the following structure

###### Dataset Feature Structure

```python
{
    "train": {
        "raw_text": [],
        "audio": [],
        "vision": [],
        "id": [], # [video_id$_$clip_id, ..., ...]
        "text": [],
        "text_bert": [],
        "audio_lengths": [],
        "vision_lengths": [],
        "annotations": [],
        "classification_labels": [], # Negative(< 0), Neutral(0), Positive(> 0)
        "regression_labels": []
    },
    "valid": {***}, # same as the "train" 
    "test": {***}, # same as the "train"
}
```

5. Modify `config/config_regression.py` to update dataset pathes.


### Run

```
sh test.sh
```

## Paper

- [CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotations of Modality](https://www.aclweb.org/anthology/2020.acl-main.343/)
- [Transformer-based Feature Reconstruction Network for Robust Multimodal Sentiment Analysis](https://dl.acm.org/doi/pdf/10.1145/3474085.3475585?casa_token=-wxKWlUW7LkAAAAA:ebkynOJtEO-2T49_kkPj5gc-AvHKAfPKkzbR9Vu1Z8pLS6ht3rWORg04JjV4ACbUhuZVbDmjIgcdqQ)

Please cite our paper if you find our work useful for your research:

```
@inproceedings{yu2020ch,
  title={CH-SIMS: A Chinese Multimodal Sentiment Analysis Dataset with Fine-grained Annotation of Modality},
  author={Yu, Wenmeng and Xu, Hua and Meng, Fanyang and Zhu, Yilin and Ma, Yixiao and Wu, Jiele and Zou, Jiyun and Yang, Kaicheng},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={3718--3727},
  year={2020}
}
```

```
@inproceedings{yuan2021transformer,
  title={Transformer-based Feature Reconstruction Network for Robust Multimodal Sentiment Analysis},
  author={Yuan, Ziqi and Li, Wei and Xu, Hua and Yu, Wenmeng},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  pages={4400--4407},
  year={2021}
}
```