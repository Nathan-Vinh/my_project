# DATASET CAN BE FIND HERE:
# https://www.kaggle.com/tingzen/thyroid-for-pretraining
# All images are .BMP, I want to conver them into png
# And create train, validation and test directory


import os
import shutil
import glob
from pathlib import Path


# It will create a directory called "png" 
# with all images in .png
# we need to use both 0 and 1 directory

current_dir = Path('.').resolve()
out_dir = current_dir / png
os.mkdir(out_dir)

for img in glob.glob(str(current_dir / "*.bmp")):
    filename = Path(img).stem
    Image.open(img).save(str(out_dir / f'{filename}.png'))


# We take all absolute path for each image 
# we are going to split them that way :
# 79% for train
# 19% for validation
# 2% for test
# I need to add the code directory creation

file_path = []
for filename in os.listdir('data\0\png'):
    full_path = os.path.join('data\0\png', filename )
    file_path.append(full_path)
    
# function to take a percentage of the path list
def to_pct(file_path, prct):
    index_file = int(len(file_path) * prct)
    return file_path[:index_file], file_path[index_file:]

f_train, f_val = partition_pct(file_path, .79)
f_val, f_test = partition_pct(f_21, 0.19)

# move the percentage of file to the right directory
for (file_list, dirname) in ((f_train, 'data\train\0'),
                             (f_val, 'data\0'),
                             (f_test, 'data\0')):
    for f in file_list:
        shutil.move(f, dirname)

