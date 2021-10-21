import os
import shutil
import random
from distutils.dir_util import copy_tree

# Call this script to randomly choose N food classes folders from food101 data folder
N = 50
dir = "/home/wei-bshg/Documents/datasets/food-101/images/"
dst = f"/home/wei-bshg/Documents/datasets/food-{N}/images/"
files = [file for file in os.listdir(dir)]
file_list = random.sample(files, N)
for file in file_list:
    copy_tree(os.path.join(dir, file), os.path.join(dst, file))
