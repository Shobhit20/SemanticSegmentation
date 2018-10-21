from fcn8s import VGG16_backbone
import preprocessing
from keras.optimizers import Adam

image_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/JPEGImages/"
segmentation_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/SegmentationClass/"
image_train_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
image_val_path = "../Documents/Datasets/Pascal/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"


train_img_labels, val_img_labels = preprocessing.label_generator(image_train_path, image_val_path)
train_segmentation = preprocessing.data_generator(image_path, segmentation_path, train_img_labels, batch_size=1)
val_segmentation = preprocessing.data_generator(image_path, segmentation_path, val_img_labels, batch_size=1)
model = VGG16_backbone()
print model.summary()


save_path = "Model/"
model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.0001, decay=0.0002), metrics=['accuracy'])

for epoch in range(2):
    model.fit_generator(train_segmentation, steps_per_epoch=len(train_img_labels), validation_data=val_segmentation, validation_steps=2, epochs=1 )
    model.save_weights( save_path+ "Weights.h5")
