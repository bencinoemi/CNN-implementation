#%% LIBRERIE
import pandas as pd
import numpy as np 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D,MaxPooling2D
from pathlib import Path
import inspect
import random
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

#%% IMPOSTO LE DIRECTORY DELLE IMMAGINI DEL TRAIN E DEL TEST
os.chdir("/home/noe/Università/in_corso/MASL/CNN/the-simpsons-characters-dataset")
CURRENT_DIR = Path(inspect.getsourcefile(lambda: 0)).resolve().parent
DATA_DIR = CURRENT_DIR / 'data'
TRAIN_CHAR_DIR = CURRENT_DIR/ 'simpsons_dataset'
TEST_IMG_DIR = CURRENT_DIR / 'kaggle_simpson_testset' / 'kaggle_simpson_testset'
folders = os.listdir(TRAIN_CHAR_DIR)


# %%  ESEMPIO DI IMMAGINE
sample = random.choice(folders)
files = os.listdir(TRAIN_CHAR_DIR/sample)
rnd_img = random.choice(files)
image = load_img(TRAIN_CHAR_DIR/sample/rnd_img)
plt.imshow(image)


#%%
def show_sample_images(folders):
    (fig, subplots) = plt.subplots(1,4, figsize=(10,10))

    for j in range(4):
        sample = random.choice(folders)
        files = os.listdir(TRAIN_CHAR_DIR/sample)
        rnd_img = files[1]
        image = load_img(TRAIN_CHAR_DIR/sample/rnd_img)
        ax = subplots[j]
        ax.set_yticklabels([])
        ax.imshow(image)
    plt.show()
    fig.savefig("/home/noe/Università/in_corso/MASL/CNN/images/simpson_example_06.png")


show_sample_images(folders)

#%%
for i in range(nchars):
    sample = folders[i]
    files = os.listdir(TRAIN_CHAR_DIR/sample)
    rnd_img = random.choice(files)
    image = load_img(TRAIN_CHAR_DIR/sample/rnd_img)
    plt.imshow(image)


#%% CREAZIONE DEL DATASET
def labeled_df_creator(folders):
    category = []
    filnames = []
    n_chars = 0
    for fold in folders:
        n_chars +=1
        files = os.listdir(TRAIN_CHAR_DIR/fold)
        for img in files:
            category.append(fold)
            filnames.append(fold+'/'+img)
    data = pd.DataFrame({'y': category, 'filename':filnames})
    return(data, n_chars, filnames)

df, nchars, filenames = labeled_df_creator(folders)


#%%

df['y'].value_counts().plot.bar()
fig.savefig("/home/noe/Università/in_corso/MASL/CNN/images/char_count_01.png")

#%%
QUICK_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANNELS = 3
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

#%% COSTRUZIONE DEL MODELLO

model = Sequential()
model.add(Conv2D(16, (3,3), activation = 'relu', input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.1))

model.add(Conv2D(32, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.1))

model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
#model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense((nchars), activation= 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

#%%

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
callbacks = [learning_rate_reduction]                              
                            
#%% TRAIN E VALIDATION SET

train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 3)
train_df = train_df.reset_index(drop= True)
valid_df = valid_df.reset_index(drop= True)
total_train = train_df.shape[0]
total_valid = valid_df.shape[0]
n_batch = 15


#%%

train_datagen = ImageDataGenerator(rotation_range=15,
                                   rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    TRAIN_CHAR_DIR, 
    x_col='filename',
    y_col='y',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=n_batch
)                                       

# %%

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    valid_df, 
    TRAIN_CHAR_DIR, 
    x_col='filename',
    y_col='y',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=n_batch
)

#%% ESEMPIO DI UN'IMMAGINE GENERATA

example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    TRAIN_CHAR_DIR, 
    x_col='filename',
    y_col='y',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

#%% APPLICAZIONE MODELLO

epochs=3 if QUICK_RUN else 20
nchars = len(train_df['y'].unique())
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_valid//n_batch,
    steps_per_epoch=total_train//n_batch,
    callbacks=callbacks
)

#%%
predicted = 
#%%
print(history.history)
# %%
from datetime import datetime
now = datetime.utcnow().strftime("%d_%H_%M_%S")
model.save_weights(f'mod_{now}')

#%%
weights = model.load_weights("mod_22_15_46_52")

#%%
fig, ax = plt.subplots(1, 1, figsize=(6,4))

ax.plot(history.history['acc'], color='b', label="Training accuracy")
ax.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
fig.savefig(f"/home/noe/Università/in_corso/MASL/CNN/images/accuracy_model_{now}.png")


#%%
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['acc'], color='b', label="Training accuracy")
ax2.plot(history.history['val_acc'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()
fig.savefig(f"/home/noe/Università/in_corso/MASL/CNN/images/accuracy_model_{now}.png")


#%%

#%%

test_filenames = os.listdir(TEST_IMG_DIR)
category = []
pattern = re.compile('_\d+.\w+')
for i in test_filenames:
    to_elim = pattern.search(i)
    cat = i.split(to_elim.group())[0]
    category.append(cat)
test_df = pd.DataFrame({
    'filename': test_filenames,
    'category': category
})
nb_samples = test_df.shape[0]

#%%

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    TEST_IMG_DIR, 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=n_batch,
    shuffle=False
)

#%%

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/n_batch))
test_df['predicted_cat'] = np.argmax(predict, axis=-1)
label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['predicted_cat'] = test_df['predicted_cat'].replace(label_map)

#%%
test_df['well_pred'] = test_df['predicted_cat']==test_df['category']

#%%
(fig, (sub1, sub2)) = plt.subplots(1, 2, figsize=(20,12))
sub1.plot(test_df['category'].value_counts().plot.bar())
sub2.plot(test_df['category'][test_df['well_pred']].value_counts().plot.bar())


#%%
width = 0.8

category = df['y']
cats   = test_df['category']
good    = test_df['category'][test_df['well_pred']]

indices = np.arange(nchars)

cats.value_counts(normalize = True).plot.bar(width=width, 
        color='yellow', label='Test images')
good.value_counts(normalize = True).plot.bar(width=0.5*width, color='red', alpha=0.5, label='Well predictes images')
category.value_counts(normalize = True).plot.bar(width=0.5*width, color='Green', alpha=0.5, label='Train images')

plt.legend(loc=0)
plt.show()
#%%
from sklearn.metrics import accuracy_score

#%%
accuracy_score(test_df['category'], test_df['predicted_cat'])
#%%
def getPercentageOfGoodPredicted(test_df):
    count = 0
    for i in range(test_df.shape[0]):
        if test_df['predicted_cat'][i] == test_df['category'][i]:
            count+=1

    acc = count/test_df.shape[0]
    return ('Percentage of well predicted:', acc)

#%%
getPercentageOfGoodPredicted(test_df)    

#%%
sample_test = test_df.head(8)
sample_test.head()
(fig, subplots) = plt.subplots(2, 4, figsize=(20,12))
n = 0
for i in range(2):
    for j in range(4):
        filename = sample_test['filename'][n]
        category = sample_test['predicted_cat'][n]
        img = load_img(TEST_IMG_DIR / filename, target_size=IMAGE_SIZE)
        ax = subplots[i][j]
        ax.imshow(img)
        ax.set_xlabel(filename + '(' + "{}".format(category) + ')' )
        n+=1
plt.tight_layout()
plt.show()
fig.savefig('/home/noe/Università/in_corso/MASL/CNN/images/test_img.png')


# %%


# %%
