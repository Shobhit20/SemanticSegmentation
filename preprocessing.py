
def label_generator(training_img_path, val_img_path):
    with open(training_img_path, 'r') as f:
        train_img_labels = f.read().split('\n')
        f.close()

    with open(val_img_path, 'r') as f:
        val_img_labels = f.read().split('\n')
        f.close()

    return train_img_labels[:-1], val_img_labels[:-1]


def data_generator(image_path, segmentation_path, img_labels):
