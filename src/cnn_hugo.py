import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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



cover_images_path = "../data/raw/boss/cover"
hugo_images_path = "../data/raw/boss/stego"




image = read_pgm(cover_images_path + "/150.pgm")
#plt.imshow(image)
#print(image)

cover_images = []
failed_cover = []

hugo_images = []
failed_hugo = []


percentage = .1

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



split = .8

split_index = int(len(cover_images)*split)
rem_count = len(cover_images) - split_index

# make training data

# add cover images based on split
x_train = cover_images[0:split_index]
y_train = [(1,0)]*split_index

# add hugo images based on split
x_train.extend(hugo_images[0:split_index])
y_train.extend([(0,1)]*split_index)


# make test data

# add cover images based on split
x_test = cover_images[split_index:]
y_test = [0]*rem_count

# add hugo images based on split
x_test.extend(hugo_images[split_index:])
y_test.extend([1]*rem_count)


print(len(x_train), len(y_train))
print(len(x_test), len(y_test))


def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b


# numpyify data
x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

# shuffle training data
x_train, y_train = randomize(x_train, y_train)

# normalize data
x_train = x_train.astype("float32") / 255
x_test = x_train.astype("float32") / 255


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
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#from keras.callbacks import ModelCheckpoint

#checkpointer = ModelCheckpoint(filepath='model.weights.best', verbose=1, save_best_only=True)

#model.fit(x_train, y_train, batch_size=1, epochs=10, callbacks=[checkpointer])
