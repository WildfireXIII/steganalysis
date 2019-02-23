import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
stego_images_path = sys.argv[2]




#image = read_pgm(cover_images_path + "/150.pgm")
#plt.imshow(image)
#print(image)

cover_images = []
failed_cover = []

stego_images = []
failed_stego = []


percentage = float(sys.argv[3])

for i in tqdm(range(1, int(10001*percentage)+1)):
    try:
        image = read_pgm(cover_images_path + "/" + str(i) + ".pgm")
        cover_images.append(image)
    except:
        failed_cover.append(i)
        
for i in tqdm(range(1, int(10001*percentage)+1)):
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
y = [(1.0,0.0)]*len(cover_images)

# add stego
x.extend(stego_images)
y.extend([(0.0,1.0)]*len(stego_images))


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


# normalize
x = x.astype("float32") / 255
mean = np.mean(x)
std = np.std(x)
x -= mean
x *= (1/std)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)


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
model.add(tf.keras.layers.Activation(tf.nn.log_softmax))
#model.add(tf.keras.layers.Softmax())

#model.add(tf.keras.layers.Dropout(0.5))
#model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

# Take a look at the model summary
model.summary()

# compile model
sgd = tf.keras.optimizers.SGD(lr=.005, decay=5e-7, momentum=0)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint

checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1, save_best_only=True)

model.fit(x_train, y_train, batch_size=20, epochs=int(sys.argv[4]), shuffle=True, validation_split=.15)


model.load_weights('model.weights.best')
score = model.evaluate(x_test, y_test, verbose=0)
print("\n", "Test accuracy:", score[1])
