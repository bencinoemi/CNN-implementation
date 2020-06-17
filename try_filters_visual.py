#%% LIBRERIE

import tensorflow as tf
import scipy.ndimage
#from scipy.misc import imsave
import matplotlib.pyplot as plt

import numpy as np 
import pandas as pd 
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D
from pathlib import Path
import inspect
import random
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau


#%% 
# take the path of the test and the train

os.chdir('/home/noe/Università/in_corso/MASL/CNN/dogs-vs-cats')
CURRENT_DIR = Path(inspect.getsourcefile(lambda: 0)).resolve().parent
TRAIN_IMG_DIR = CURRENT_DIR / 'train' / 'train'
TEST_IMG_DIR = CURRENT_DIR / 'test' / 'test'
files = os.listdir(TRAIN_IMG_DIR)

#%%
#sample = random.choice(files)
sample = "cat.191.jpg"
image = load_img(TRAIN_IMG_DIR/sample)
plt.imshow(image)

os.chdir('/home/noe/Università/in_corso/MASL/CNN/dogs-vs-cats/train/train')
image = cv2.imread(sample)
IMAGE_DIM = image.shape

#%%
def show_img(model, img):
    img_batch = np.expand_dims(img, axis=0)
    conv_img = model.predict(img_batch)
    conv_img = np.squeeze(conv_img, axis=0)

    (fig, subplots) = plt.subplots(1,4, figsize=(12,6))
    n = 0
    for j in range(4):
        ax = subplots[j]
        ax.imshow(conv_img[:,:,n])
        n+=1
    plt.show()
    fig.savefig("/home/noe/Università/in_corso/MASL/CNN/images/example_conv_03.png")

    
   
#%% ONE LAYER CONVOLUTION

model = Sequential()
model.add(Conv2D(4, (3,3), input_shape = IMAGE_DIM))
model.summary()

show_img(model, image)
#plt.savefig("/home/noe/Università/in_corso/MASL/CNN/images/example_conv_02.png")

#%%

def create_dataset_labeled(filenames):
    category = []
    for filename in filenames:
        category.append(filename.split('.')[0])
#        if cat == 'dog':
#            category.append(1)
#        else:
#            category.append(0)
    data = pd.DataFrame({'filename': filenames, 'category': category})
    return data

data = create_dataset_labeled(files)
#%%
example_df = data.loc[data['filename'] == 'cat.191.jpg']
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    TRAIN_IMG_DIR, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)
plt.figure(figsize=(8,6))
for i in range(4):
    plt.subplot(1, 4, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.savefig("/home/noe/Università/in_corso/MASL/CNN/images/data_aug_07.png")

plt.show()


#%% TWO LAYER CONVOLUTION AND MAX POOLING
model = Sequential()
model.add(Conv2D(4, (3,3), input_shape = IMAGE_DIM))
model.add(MaxPooling2D(2,2))
#model.add(Conv2D(4, (3,3)))
#model.add(MaxPooling2D(2,2))
model.summary()

show_img(model, image)
#%% 3 LAYER CONVOLUTION, MAX POOLING and CONVOLUTION
model = Sequential()
model.add(Conv2D(4, (3,3), input_shape = IMAGE_DIM))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(4, (3,3)))
#model.add(MaxPooling2D(2,2))
model.summary()

show_img(model, image)

#%% 4 LAYER CONVOLUTION, MAX POOLING and CONVOLUTION
model = Sequential()
model.add(Conv2D(4, (3,3), input_shape = IMAGE_DIM))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(4, (3,3)))
model.add(MaxPooling2D(2,2))
model.summary()

show_img(model, image)


# %%
