import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import random

import filter_weights

import sys

# cmdline args: 1 = cover path 2 = stego pat 3 = % data 4 = # epochs

import re

def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))

def log_loss(y_true, y_pred):
    return tf.losses.log_loss(y_true, y_pred)

#cover_images_path = "../data/raw/boss/cover"
#hugo_images_path = "../data/raw/boss/stego"

cover_images_path = sys.argv[1]
stego_images_path = sys.argv[2]


print(sys.argv)


#image = read_pgm(cover_images_path + "/150.pgm")
#plt.imshow(image)
#print(image)

cover_images = []
failed_cover = []

stego_images = []
failed_stego = []


percentage = float(sys.argv[3])

for i in range(1, int(10001*percentage)+1):
    try:
        image = read_pgm(cover_images_path + "/" + str(i) + ".pgm")
        cover_images.append(image)
    except:
        failed_cover.append(i)
        
for i in range(1, int(10001*percentage)+1):
    try:
        image = read_pgm(stego_images_path + "/" + str(i) + ".pgm")
        stego_images.append(image)
    except:
        failed_stego.append(i)



print(len(failed_cover))
print(failed_cover)
print(len(failed_stego))
print(failed_stego)


# get same number in both sets
# TODO: these are not removing the same images in both

stego_images.pop(0)
print(len(cover_images))
print(len(stego_images))



#split = .8
#
#split_index = int(len(cover_images)*split)
#rem_count = len(cover_images) - split_index
#
#split_hugo_index = int(len(hugo_images)*split)
#rem_hugo_count = int(len(hugo_images) - split_hugo_index

# make training data

# add cover
x = cover_images
y = [(0,1)]*len(cover_images)

# add stego
x.extend(stego_images)
y.extend([(1,0)]*len(stego_images))


#def randomize(a, b):
#    # Generate the permutation index array.
#    permutation = np.random.permutation(a.shape[0])
#    # Shuffle the arrays by giving the permutation in the square brackets.
#    shuffled_a = a[permutation]
#    shuffled_b = b[permutation]
#    return shuffled_a, shuffled_b

# numpyify data
x = np.array(x)
y = np.array(y)

x_resized = []

# resize all images
for image in x:
    resized = np.array(Image.fromarray(image).resize((256, 256)))
    x_resized.append(resized)

x = np.asarray(x_resized)

# normalize
#x = x.astype("float32") / 255
#mean = np.mean(x)
#std = np.std(x)
#x -= mean
#x *= (1/std)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=random.randint(1,100))


#x, y = randomize(x, y)
#
#
## split into training and testing
#split = .8
#
#split_index = int(len(x)*split)
#rem_count = len(x) - split_index
#
#x_train = x[0:split_index]
#y_train = y[0:split_index]
#
#x_test = x[split_index:]
#y_test = y[split_index:]

# normalize data
#x_train = x_train.astype("float32") / 255
#x_test = x_test.astype("float32") / 255

# get the data into the right shape
w, h = 256, 256
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# print shape information
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


weight_decay = .0001
reg = tf.keras.regularizers.l2(weight_decay)


model = tf.keras.Sequential()


# (don't know if below is needed, it's sort of preprocessing)
#model.add(tf.keras.layers.ZeroPadding2D(padding=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=5, padding='same', activation='linear', input_shape=(256,256,1), use_bias=False))
model.layers[0].set_weights(filter_weights.hp_filters)
model.layers[0].trainable = False


# NOTE: that scale layer is incorporated into batch norm: https://github.com/flyyufelix/DenseNet-Keras/issues/5


#model.add(tf.keras.layers.MaxPooling2D(pool_size=8))
#model.add(tf.keras.layers.Dropout(0.3))


model.add(tf.keras.layers.ZeroPadding2D(padding=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=5, padding='valid', activation='linear', kernel_initializer='glorot_normal', use_bias=False))
model.add(tf.keras.layers.Lambda(lambda x: tf.keras.backend.abs(x)))
model.add(tf.keras.layers.BatchNormalization(momentum=.95))
model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -3, 3)))


model.add(tf.keras.layers.ZeroPadding2D(padding=(2,2)))
model.add(tf.keras.layers.Conv2D(filters=30, kernel_size=5, padding='valid', activation='linear', kernel_initializer='glorot_normal', use_bias=False))
model.add(tf.keras.layers.BatchNormalization(momentum=.95))
model.add(tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, -2, 2)))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(5,5), strides=2, padding="same"))


model.add(tf.keras.layers.ZeroPadding2D(padding=(1,1)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid', activation='linear', kernel_initializer='glorot_normal', use_bias=False))
model.add(tf.keras.layers.BatchNormalization(momentum=.95))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.AveragePooling2D(pool_size=(5,5), strides=2, padding="same"))

model.add(tf.keras.layers.ZeroPadding2D(padding=(1,1)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid', activation='linear', kernel_initializer='glorot_normal', use_bias=False))
model.add(tf.keras.layers.BatchNormalization(momentum=.95))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.AveragePooling2D(pool_size=(5,5), strides=2, padding="same"))


model.add(tf.keras.layers.ZeroPadding2D(padding=(1,1)))
model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid', activation='linear', kernel_initializer='glorot_normal', use_bias=False))
model.add(tf.keras.layers.BatchNormalization(momentum=.95))
model.add(tf.keras.layers.ReLU())
model.add(tf.keras.layers.GlobalAveragePooling2D())


model.add(tf.keras.layers.Flatten()) # note that in the paper they use reshape to 64x4 instead of flattening
model.add(tf.keras.layers.Dense(256, activation='relu', kernel_initializer='glorot_normal'))
model.add(tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Softmax())


#model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=509, padding='same', activation='tanh'))

#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.3))

#model.add(tf.keras.layers.Flatten()) # note that in the paper they use reshape to 64x4 instead of flattening
#model.add(tf.keras.layers.Dense(2))
#model.add(tf.keras.layers.Activation(tf.nn.log_softmax))
#model.add(tf.keras.layers.Softmax())

#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Take a look at the model summary
model.summary()

# https://stackoverflow.com/questions/48198031/keras-add-variables-to-progress-bar/48206009#48206009
def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

# https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
def step_decay(epoch):
    initial_lrate = .01
    drop = .9
    epochs_drop = 900/10 # in paper, they used 900 as maximum epochs, and we should decay after each 10% of the total number of epochs
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = tf.keras.callbacks.LearningRateScheduler(step_decay)

# compile model
sgd = tf.keras.optimizers.SGD(lr=.01, decay=0.0, momentum=.95)
lr_metric = get_lr_metric(sgd)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy', lr_metric])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1, save_best_only=True)

model.fit(x_train, y_train, batch_size=16, epochs=int(sys.argv[4]), shuffle=True, validation_split=.15, verbose=2, callbacks=[lrate])


#model.load_weights('model.weights.best')
score = model.evaluate(x_test, y_test, verbose=0)
print("\n", "Test accuracy:", score[1])

#outputs = model.predict(x_test[0:3])
#print(outputs)

