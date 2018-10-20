from fcn8s import VGG16_backbone
import numpy as np
model = VGG16_backbone()

import os

import preprocessing
image_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/JPEGImages/"
segmentation_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/SegmentationClass/"
image_train_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
image_val_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"


train_img_labels, val_img_labels = preprocessing.label_generator(image_train_path, image_val_path)

train_img_segmentation = preprocessing.data_generator(image_path, segmentation_path, train_img_labels)