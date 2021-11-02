import os
import numpy as np
from nibabel.testing import data_path
import nibabel as nib
import h5py
import tensorflow as tf

def load_data(images_path, masks_path):
    h5f = h5py.File(images_path,'r')
    images = h5f['all_data'][:]
    h5f.close()

    h5f = h5py.File(masks_path,'r')
    masks = h5f['all_data'][:]
    h5f.close()


    total = images.shape[0]

    sliced = round(0.7 * total)

    train_images = images[:sliced]
    test_images = images[sliced:]

    train_masks = masks[:sliced]
    test_masks = masks[sliced:]

    train_images = tf.constant(train_images)
    train_masks = tf.constant(train_masks)
    
    test_images = tf.constant(test_images)
    test_masks = tf.constant(test_masks)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))

    return train_dataset, test_dataset, sliced
    




train, test, train_len = load_data("/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation/Images.h5",
                    "/media/jiangl/50EC5AFF0AA889DF/Mini Database/Images and Masks For Tumor Segmentation/Masks.h5")

    
def normalize(input_array, input_mask):
  input_image = tf.cast(input_array, tf.float32) / 255.0
  return input_image, input_mask



def load_image_train(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    input_mask = tf.image.flip_left_right(input_mask)

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask


TRAIN_LENGTH = train_len
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train, test = dataset['train'], dataset['test']

iterator = train.__iter__()
next_element = iterator.get_next()
print(next_element)

train = train.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test = test.map(load_image_test)

iterator = train.__iter__()
next_element = iterator.get_next()
print("asdf")
print(next_element)


train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)

iterator = train_dataset.__iter__()
next_element = iterator.get_next()
print(next_element)
