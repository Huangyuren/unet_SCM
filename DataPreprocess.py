from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from skimage import img_as_uint
from PIL import Image
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
import warnings

warnings.filterwarnings("ignore")

muscle = [255, 255, 255]
background = [0, 0, 0]
# COLOR_DICT = np.array([BackGround, road])
# one = [128, 128, 128]
# two = [128, 0, 0]
# three = [192, 192, 128]
# four = [255, 69, 0]
# five = [128, 64, 128]
# six = [60, 40, 222]
# seven = [128, 128, 0]
# eight = [192, 128, 128]
# nine = [64, 64, 128]
# ten = [64, 0, 128]
# eleven = [64, 64, 0]
# twelve = [0, 128, 192]
# COLOR_DICT = np.array([one, two,three,four,five,six,seven,eight,nine,ten,eleven,twelve])



class data_preprocess:
    def __init__(self, train_path=None, image_folder=None, label_folder=None,
                 valid_path=None,valid_image_folder =None,valid_label_folder = None,
                 target_rows=None, target_cols=None,
                 flag_multi_class=False,
                 num_classes = 2, img_type = 'tif'):
        self.train_path = train_path
        self.valid_path = valid_path
        self.valid_image_folder = valid_image_folder
        self.valid_label_folder = valid_label_folder
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.data_gen_args = dict(rotation_range=0.2,
                                  width_shift_range=0.05,
                                  height_shift_range=0.05,
                                  shear_range=0.05,
                                  zoom_range=0.05,
                                  vertical_flip=True,
                                  horizontal_flip=True,
                                  fill_mode='nearest')
        self.image_color_mode = "rgb"
        self.label_color_mode = "rgb"
        self.target_size = (target_rows, target_cols)
        self.img_type = img_type
        
    def adjustData(self, img, label):
        # Simple image & label normalization.
        min_, max_ = float(np.min(img)), float(np.max(img))
        img = (img - min_) / (max_ - min_)
        min_label, max_label = float(np.min(label)), float(np.max(label))
        label = (label - min_label) / (max_label - min_label)
        label[label > 0.5] = 1.0
        label[label <= 0.5] = 0.0
        return (img, label)
    
    def trainGenerator(self, batch_size, image_save_prefix="image", label_save_prefix="label",
                       save_to_dir=None, seed=9):
        '''
        can generate image and label at the same time
        use the same seed for image_datagen and label_datagen to ensure the transformation for image and label is the same
        if you want to visualize the results of generator, set save_to_dir = "your path"
        '''
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.train_path,
            classes=[self.image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=image_save_prefix,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.train_path,
            classes=[self.label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            save_to_dir=save_to_dir,
            save_prefix=label_save_prefix,
            seed=seed)
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjustData(img, label)
#             print('check img shape: '+img.shape+', check label shape: '+label.shape+'\n')
            yield (img, label)
            
    def validation_load(self, batch_size,seed=9):
        image_datagen = ImageDataGenerator(**self.data_gen_args)
        label_datagen = ImageDataGenerator(**self.data_gen_args)
        image_generator = image_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_image_folder],
            class_mode=None,
            color_mode=self.image_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            seed=seed)
        label_generator = label_datagen.flow_from_directory(
            self.valid_path,
            classes=[self.valid_label_folder],
            class_mode=None,
            color_mode=self.label_color_mode,
            target_size=self.target_size,
            batch_size=batch_size,
            seed=seed)
        train_generator = zip(image_generator, label_generator)
        for (img, label) in train_generator:
            img, label = self.adjustData(img, label)
            yield (img, label)
        
