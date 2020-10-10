# Universal Weighting Metric Learning for Cross-Modal Matching
This repository contains PyTorch implementation of our paper [Universal Weighting Metric Learning for Cross-Modal Matching](https://arxiv.org/abs/2010.03403).
The paper is accepted by CVPR2020. It is built on top of the [SCAN](https://github.com/kuanghuei/SCAN) in PyTorch.


## Requirements and Installation
We recommended the following dependencies.

* Python 3.7
* [PyTorch](http://pytorch.org/) 1.1
* [NumPy](http://www.numpy.org/) 

* Punkt Sentence Tokenizer:
```python
import nltk
nltk.download()
> d punkt
```


## Data preparation

Download the dataset files. We use splits produced by [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/deepimagesent/). The raw images can be downloaded from from their original sources [here](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html), [here](http://shannon.cs.illinois.edu/DenotationGraph/) and [here](http://mscoco.org/).

The precomputed image features are extracted from the raw images using the bottom-up attention model from [here](https://github.com/peteanderson80/bottom-up-attention). Image features for training set, validation set and testing set should be merged in order into one `.npy` file, respectively. More details about the image feature extraction can also be found in SCAN(https://github.com/kuanghuei/SCAN).

Data files can be found in SCAN (We use the same dataset split as theirs):

```bash
wget https://scanproject.blob.core.windows.net/scan-data/data_no_feature.zip
```

## Training
Arguments used to train Flickr30K models and MSCOCO models are similar with those of SCAN:

For Flickr30K:
```bash
python train.py --data_path "$DATA_PATH" --data_name coco_precomp --vocab_path "$VOCAB_PATH" --logger_name runs/coco_scan/log --model_name runs/coco_scan/log --max_violation --bi_gru  --agg_func=Mean --cross_attn=i2t --lambda_softmax=4
```
## Pretrained model
If you don't want to train from scratch, you can download the pretrained model (Flickr30K) from Google Driver [here](https://drive.google.com/file/d/1DTYFioI1BRIQZssMRpW6lQpTVHeQV3vA/view?usp=sharing).
```
rsum: 460.7
Average i2t Recall: 84.9
Image to text: 69.4 89.9 95.4 1.0 4.1
Average t2i Recall: 68.7
Text to image: 47.5 75.5 83.1 2.0 12.4
```

## Reference

If you found this code useful, please cite the following paper:
```
@inproceedings{wei2020universal,
  title={Universal Weighting Metric Learning for Cross-Modal Matching},
  author={Wei, Jiwei and Xu, Xing and Yang, Yang and Ji, Yanli and Wang, Zheng and Shen, Heng Tao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13005--13014},
  year={2020}
}
```
