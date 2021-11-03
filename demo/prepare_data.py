import os
from shutil import copy
import collections

# Helper method to split dataset into train and test folders
def prepare_data(filepath, src, dest):
  classes_images = collections.defaultdict(list)
  with open(filepath, 'r') as txt:
      paths = [read.strip() for read in txt.readlines()]
      for p in paths:
        food = p.split('/')
        classes_images[food[0]].append(food[1] + '.jpg')

  for food in classes_images.keys():
    print("\nCopying images into ",food)
    if not os.path.exists(os.path.join(dest,food)):
      os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")

model_dir = "/home/wei-bshg/Documents/datasets/food-101"
# Creating train and test data
prepare_data(model_dir+'/meta/train.txt', 
             model_dir+'/images', 
             model_dir+'train')

prepare_data(model_dir+'/meta/test.txt', 
             model_dir+'/images', 
             model_dir+'test')            
