#!/bin/python3

import os
import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]
algorithm_path = sys.argv[3]
num_diff_keys = int(sys.argv[4])

# get all image filenames
images = os.listdir(input_dir)

images_per_key = int(len(images)/num_diff_keys)
rem = len(images%num_diff_keys)


image_index = 1
for i in range(1, num_diff_keys + 1):

    # handle remainder on last key
    count = images_per_key
    if i == num_diff_keys: count = images_per_key + rem

    for j in range(0, images_per_key):
        image_path = input_dir + "/" + str(image_index) + ".pgm"
        os.system("{0} -v -i {1} -O {2} -a 0.4 -r {3}".format(algorithm_path, image_path, output_dir, i))
        image_index += 1
