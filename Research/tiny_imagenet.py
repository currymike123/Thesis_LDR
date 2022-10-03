# %%
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd
import os
import PIL
import PIL.Image
import glob

import cv2

import matplotlib.pyplot as plt
from matplotlib import gridspec

# %%
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# %%
ds_train = image_dataset_from_directory(
'tiny_imagenet/train',
image_size=(64,64),
batch_size=32)

# %%
ds_val = image_dataset_from_directory(
'tiny_imagenet/val',
image_size=(64,64),
batch_size=32)

# %%
data_augmentation = keras.Sequential(
[
layers.RandomFlip("horizontal"),
layers.RandomRotation(0.1),
layers.RandomZoom(0.2),
]
)

inputs = keras.Input(shape=(64,64,3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

for size in [32,64,128,256,512]:
    residual = x
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])


x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(200, activation="softmax")(x)
model = keras.Model(inputs, outputs)

opt = tf.optimizers.Adam(learning_rate=0.0001)


model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks = [
keras.callbacks.ModelCheckpoint(
filepath="tiny_imagenet_feature_extraction_with_data_augmentation.keras",
save_best_only=True,
monitor="val_loss")
]

# %%
history = model.fit(
ds_train,
epochs=100,
validation_data=ds_val,
callbacks=callbacks)

# %%
x, y = zip(*ds_train)
x_train = np.concatenate(x)
y_train = np.concatenate(y)

x, y = zip(*ds_val)
x_val = np.concatenate(x)
y_val = np.concatenate(y)

x_train = np.uint8(x_train)

# %%
from hdr_blend import*

end = len(x_train)

x_dataAug = []
y_dataAug = []
for image in range(0,end):
    x_dataAug.append(hdr(x_train[image]))
    y_dataAug.append(y_train[image])


x_dataAug = np.asarray(x_dataAug, dtype=int)
y_dataAug = np.asarray(y_dataAug)

# %%
data_augmentation = keras.Sequential(
[
layers.RandomFlip("horizontal"),
layers.RandomRotation(0.1),
layers.RandomZoom(0.2),
]
)

inputs = keras.Input(shape=(64,64,3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = layers.Conv2D(filters=32, kernel_size=5, use_bias=False)(x)

for size in [32,64,128,256,512]:
    residual = x
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(size, 3, padding="same", use_bias=False)(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    residual = layers.Conv2D(size, 1, strides=2, padding="same", use_bias=False)(residual)
    x = layers.add([x, residual])


x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(200, activation="softmax")(x)
model = keras.Model(inputs, outputs)

opt = tf.optimizers.Adam(learning_rate=0.0001)


model.compile(
    optimizer=opt,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

callbacks = [
keras.callbacks.ModelCheckpoint(
filepath="tiny_imagenet_feature_extraction_with_data_augmentation_HDR.keras",
save_best_only=True,
monitor="val_loss")
]

# %%
history = model.fit(
x_train, y_train,
epochs=100,
validation_data=(x_val,y_val),
callbacks=callbacks)


