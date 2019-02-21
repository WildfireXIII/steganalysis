import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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



#cover_images_path = "../data/raw/boss/cover"
#hugo_images_path = "../data/raw/boss/stego"

cover_images_path = sys.argv[1]
hugo_images_path = sys.argv[2]




image = read_pgm(cover_images_path + "/150.pgm")
#plt.imshow(image)
#print(image)

cover_images = []
failed_cover = []

hugo_images = []
failed_hugo = []


percentage = float(sys.argv[3])

for i in tqdm(range(1, int(10001*percentage)+1)):
    try:
        image = read_pgm(cover_images_path + "/" + str(i) + ".pgm")
        cover_images.append(image)
    except:
        failed_cover.append(i)
        
for i in tqdm(range(1, int(10001*percentage)+1)):
    try:
        image = read_pgm(hugo_images_path + "/" + str(i) + ".pgm")
        hugo_images.append(image)
    except:
        failed_hugo.append(i)



print(len(failed_cover))
print(failed_cover)
print(len(failed_hugo))
print(failed_hugo)


# get same number in both sets
# TODO: these are not removing the same images in both

hugo_images.pop(0)
print(len(cover_images))
print(len(hugo_images))



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
y = [(1,0)]*len(cover_images)

# add stego
x.extend(hugo_images)
y.extend([(0,1)]*len(hugo_images))


def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

# numpyify data
x = np.array(x)
y = np.array(y)

x, y = randomize(x, y)


# split into training and testing
split = .8

split_index = int(len(x)*split)
rem_count = len(x) - split_index

x_train = x[0:split_index]
y_train = y[0:split_index]

x_test = x[split_index:]
y_test = y[split_index:]

# normalize data
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# get the data into the right shape
w, h = 512, 512
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# print shape information
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding='valid', activation='tanh', input_shape=(512,512,1)))
#model.add(tf.keras.layers.MaxPooling2D(pool_size=8))
#model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=509, padding='valid', activation='tanh'))
#model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=509, padding='same', activation='tanh'))

#model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
#model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten()) # note that in the paper they use reshape to 64x4 instead of flattening
model.add(tf.keras.layers.Dense(2))
model.add(tf.keras.layers.Softmax())
#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Take a look at the model summary
model.summary()

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1, save_best_only=True)

model.fit(x_train, y_train, batch_size=1, epochs=int(sys.argv[4]), callbacks=[checkpointer], shuffle=True)


#model.load_weights('model.weights.best')

score = model.evaluate(x_test, y_test, verbose=0)
print("\n", "Test accuracy:", score[1])
