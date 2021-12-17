import sys, os
p = os.path.abspath('.')
sys.path.insert(1, p)
from shutil import copy
import collections
from demo import dataset_dir

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
      if os.path.exists(os.path.join(src,food)):
        os.makedirs(os.path.join(dest,food))
    for i in classes_images[food]:
      if os.path.exists(os.path.join(src,food)):
        copy(os.path.join(src,food,i), os.path.join(dest,food,i))
  print("Copying Done!")


prepare_data(dataset_dir+'/meta/train.txt', 
             dataset_dir+'/images', 
             dataset_dir+'train')

prepare_data(dataset_dir+'/meta/test.txt', 
             dataset_dir+'/images', 
             dataset_dir+'test')            
