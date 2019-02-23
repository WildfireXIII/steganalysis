#!/bin/python3

import os
import sys
from shutil import copyfile
from tqdm import tqdm

# [1] = original cover path
# [2] = where to make output folders

try: os.mkdir(sys.argv[2])
except: pass
for i in range(1, 22):
    try: os.mkdir(sys.argv[2] + "/" + str(i))
    except: pass

for path in tqdm(os.listdir(sys.argv[1])):
    filename_only = os.path.splitext(path)[0]

    num = int(filename_only)
    folder = int(num/500)+1

    copyfile(sys.argv[1] + "/" + path, sys.argv[2] + "/" + str(folder) + "/" + path)
