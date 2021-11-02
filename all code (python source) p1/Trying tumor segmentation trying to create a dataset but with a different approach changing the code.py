import os
import sys
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
import h5py
import scipy

## Seeding 
seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed


class DataGen(keras.utils.Sequence):
    def __init__(self, images_path, masks_path, batch_size=8, image_size=128):
        self.batch_size = batch_size
        self.image_size = image_size
        
        
        h5f = h5py.File(images_path,'r')
        images = h5f['all_data'][:]
        h5f.close()

        images = images/(np.max(images)-np.min(images))


        h5f = h5py.File(masks_path,'r')
        masks = h5f['all_data'][:]
        h5f.close()
        
        self.data_size = images.shape[0]
        
        ##resizing the arrays

        rescaled_images = []
        rescaled_masks = []

        for n, image in enumerate(images):
            z_zoom = image_size/image.shape[0]
            y_zoom = image_size/image.shape[1]
            x_zoom = image_size/image.shape[2]
            
            rescaled_images.append(scipy.ndimage.zoom(image, (z_zoom, y_zoom, x_zoom)))

        for n, mask in enumerate(masks):
            z_zoom = image_size/mask.shape[0]
            y_zoom = image_size/mask.shape[1]
            x_zoom = image_size/mask.shape[2]
            
            rescaled_masks.append(scipy.ndimage.zoom(mask, (z_zoom, y_zoom, x_zoom)))


        channeled_images = []
        channeled_masks = []
        
        for image in rescaled_images:
            channeled_images.append(np.stack([image], axis = 3))

        for mask in rescaled_masks:
            channeled_masks.append(np.stack([mask], axis = 3))
            
            
        
        
        self.images = np.array(channeled_images)
        self.masks = np.array(channeled_masks)

        self.on_epoch_end()
        
        
     
    def __getitem__(self, index):
        if(index+1)*self.batch_size > self.images.shape[0]:
            self.batch_size = self.images.shape[0] - index*self.batch_size
        
        images = []
        masks  = []

        for image in self.images[index*self.batch_size : (index+1)*self.batch_size]:
            images.append(image)
        for mask in self.masks[index*self.batch_size : (index+1)*self.batch_size]:
            masks.append(mask)
            
        images = np.array(images)
        masks  = np.array(masks)
        
        return images, masks
    
    def on_epoch_end(self):
        pass
    
    def __len__(self):
        return int(np.ceil(self.data_size/float(self.batch_size)))


image_size = 128
train_images_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation/Train Images.h5"
train_masks_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation/Train Masks.h5"
test_images_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation/Test Images.h5"
test_masks_path = "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation/Test Masks.h5"
epochs = 5
batch_size = 2





train_gen = DataGen(train_images_path, train_masks_path, batch_size=batch_size, image_size=image_size)
x, y = train_gen.__getitem__(0)
print(train_gen.data_size)
print(x.shape, y.shape)


r = random.randint(0, len(x)-1)

#fig = plt.figure()
#fig.subplots_adjust(hspace=0.4, wspace=0.4)
#ax = fig.add_subplot(1, 2, 1)
#ax.imshow(x[r][100])
#ax = fig.add_subplot(1, 2, 2)
#ax.imshow(y[r][100])

#plt.show()

def down_block(x, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    p = keras.layers.MaxPool3D((2, 2, 2), (2, 2, 2))(c)
    return c, p

def up_block(x, skip, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
    us = keras.layers.UpSampling3D((2, 2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def bottleneck(x, filters, kernel_size=(3, 3, 3), padding="same", strides=1):
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
    c = keras.layers.Conv3D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, image_size, 1))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv3D(1, (1, 1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model


model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

train_gen = DataGen(train_images_path, train_masks_path, batch_size=batch_size, image_size=image_size)
valid_gen = DataGen(test_images_path, test_masks_path, batch_size=batch_size, image_size=image_size)

x, y = train_gen.__getitem__(0)
print("dab")
print(valid_gen.images.shape[0])


train_steps = train_gen.data_size//batch_size
valid_steps = valid_gen.data_size//batch_size

model.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs)

## Save the Weights
model.save_weights("UNetW.h5")

## Dataset for prediction
x, y = valid_gen.__getitem__(1)
result = model.predict(x)

result = result > 0.5



fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(1, 2, 1)
ax.imshow(np.reshape(y[0]*255, (image_size, image_size)), cmap="gray")

ax = fig.add_subplot(1, 2, 2)
ax.imshow(np.reshape(result[0]*255, (image_size, image_size)), cmap="gray")

plt.show()
