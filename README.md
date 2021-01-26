# SimCLR
Implementation of SimCLR (A Simple Framework for Contrastive Learning of Visual Representations)


* **Requirements**
  * numpy
  * torch
  * torchvision
  * opencv-python
  
## Result
|         Top-1 Acc.   |Baseline|Linear evaluation||Fine-Tuning| |
|----------------------|--------|----------------|-|-----------|-|
|||High color distortion (0.8)|Low color distortion(0.2)|High color distortion (0.8)|Low color distortion(0.2)|
|STL-10|29.3%|35.74%|35.55%|44.14%|45.31%|
|CIFAR-10|13.87%|17.77%|21.48%|23.83%|27.15%|

### Reference
1. A Simple Framework for Contrastive Learning of Visual Representations (https://arxiv.org/abs/2002.05709)
2. STL-10 Dataset (https://cs.stanford.edu/~acoates/stl10/)
