# Semantic Segmentation DeconvNet PyTorch

#### Data:
- PASCAL VOC 2012 segmentation dataset (train + val)
- Download http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
- extract to ./data

#### Data Preprocessing:
color map (3 * 224 * 224) --> one-hot class map (21 * 224 * 224)

##### DeconvNet training:
- epochs = 50
- Batch size = 64
- Learning rate = 0.001
- Loss func =  CrossEntropyLoss

#### Reference:
- http://cvlab.postech.ac.kr/research/deconvnet
- https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation
