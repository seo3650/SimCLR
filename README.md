# SimCLR
Implementation of SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)


* **Requirements**
  * numpy
  * torch
  * torchvision
  * opencv-python

## Quick start
### Encoder training
```
python main.py --batch_size 512 --option 'unlabel_train' --epochs 40 --dataset 'STL-10'
```
### Classification training
```
python main.py --batch_size 512 --option 'label_train' --epochs 1000 --dataset 'STL-10' --prev_model [prev_model]
```
### Fine tuning
```
python main.py --batch_size 512 --option 'fine_tuning' --epochs 1000 --dataset 'STL-10' --prev_model [prev_model]
```
### Test
```
python main.py --batch_size 512 --option test --prev_model [prev_model]
```
  
## Result
|         Top-1 Acc.   |Baseline|Linear evaluation|Fine-Tuning|
|----------------------|--------|----------------|-----------|
|STL-10|29.1%|48.44%|50.00%|

### Reference
1. A Simple Framework for Contrastive Learning of Visual Representations (https://arxiv.org/abs/2002.05709)
2. STL-10 Dataset (https://cs.stanford.edu/~acoates/stl10/)
