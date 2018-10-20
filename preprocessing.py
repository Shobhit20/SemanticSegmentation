import itertools
import cv2
import numpy as np
from PIL import Image


VOC_CLASSES = 21
VGG_IMG_WIDTH, VGG_IMG_HEIGHT = 224, 224

def preprocess_input(x, dim_ordering='default'):

    x[:, :, 0] -= 104.00698793
    x[:, :, 1] -= 116.66876762
    x[:, :, 2] -= 122.67891434

    return x


def load_images(img_path):
    img = Image.open(img_path)

    img = img.resize((VGG_IMG_WIDTH, VGG_IMG_HEIGHT), Image.ANTIALIAS)
    img = np.array(img).astype(float)
    print(img.shape)
    img = preprocess_input(img)
    return img


def load_segmented_images(segmented_img_path):
    seg_voc_layers = np.zeros((VGG_IMG_WIDTH, VGG_IMG_HEIGHT, VOC_CLASSES))
    segmented_img = cv2.imread(segmented_img_path, 1)
    segmented_img = cv2.resize(segmented_img, (VGG_IMG_WIDTH, VGG_IMG_HEIGHT))
    segmented_img = segmented_img[:, :, 0]
    for i in range(VOC_CLASSES):
        seg_voc_layers[:, :, i] = (segmented_img == i).astype(int)
    print(seg_voc_layers.shape)
    return seg_voc_layers



def label_generator(training_img_path, val_img_path):
    with open(training_img_path, 'r') as f:
        train_img_labels = f.read().split('\n')
        f.close()

    with open(val_img_path, 'r') as f:
        val_img_labels = f.read().split('\n')
        f.close()

    return train_img_labels[:-1], val_img_labels[:-1]


def data_generator(image_path, segmentation_path, img_labels, val=False, batch_size=64):
    image_set, image_segmentation_set = [], []
    if not val:
        for i in range(len(img_labels)):
            image_set.append(image_path + img_labels[i] + ".jpg")
            image_segmentation_set.append(segmentation_path + img_labels[i]+".jpg")

    img_segmentation_pair = itertools.cycle(zip(image_set, image_segmentation_set))
    while True:
        X, labels = [], []
        for i in range(batch_size):
            img, segmentation = next(img_segmentation_pair)
            X.append(load_images(img))
            labels.append(load_segmented_images(segmentation))
