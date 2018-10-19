from fcn8s import VGG16_backbone
import numpy as np
model = VGG16_backbone()
print(model.layers[1].get_weights())