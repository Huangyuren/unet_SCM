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
                 test_path=None, test_fullscale_path=None, save_path=None, 
                 save_post_path = None, save_roi_restored=None,
                 target_rows=None, target_cols=None,
                 flag_multi_class=False,
                 flag_isdemo=False,
                 demo_result_path=None,
                 demo_filename=None,
                 num_classes = 2, img_type = 'tif'):
        self.train_path = train_path
        self.valid_path = valid_path
        self.valid_image_folder = valid_image_folder
        self.valid_label_folder = valid_label_folder
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.test_path = test_path
        self.test_fullscale_path = test_fullscale_path
        self.save_path = save_path
        self.save_post_path = save_post_path
        self.save_roi_restored=save_roi_restored
        self.demo_result_path = demo_result_path
        self.demo_filename = demo_filename
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
        self.flag_multi_class = flag_multi_class
        self.num_class = num_classes
        self.target_size = (target_rows, target_cols)
        self.img_type = img_type
        
    def adjustData(self, img, label):
        if (self.flag_multi_class):
            img = img / 255.
            label = label[:, :, :, 0] if (len(label.shape) == 4) else label[:, :, 0]
            new_label = np.zeros(label.shape + (self.num_class,))
            for i in range(self.num_class):
                new_label[label == i, i] = 1
            label = new_label
        elif (np.max(img) > 1):
#             print('np.max > 1')
            min_, max_ = float(np.min(img)), float(np.max(img))
            img = (img - min_) / (max_ - min_)
            min_label, max_label = float(np.min(label)), float(np.max(label))
            label = (label - min_label) / (max_label - min_label)
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
        return (img, label)

    def trainGenerator(self, batch_size, image_save_prefix="image", label_save_prefix="label",
                       save_to_dir=None, seed=7):
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

    def testGenerator(self):
        filenames = os.listdir(self.test_path)
        for filename in filenames:
            img = io.imread(os.path.join(self.test_path, filename), as_gray=False)
            img = img / 255.
            img = trans.resize(img, self.target_size, mode='constant')
            img = np.reshape(img, img.shape + (1,)) if (not self.flag_multi_class) else img
            img = np.reshape(img, (1,) + img.shape)
            yield img
            
    def validation_load(self, batch_size,seed=7):
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
            
    def image_normalized(self, file_path):
        '''
            tif，size:*，gray64
            :param dir_path: path to your images directory
            :return:
        '''
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img_shape = img.shape
        image_size = (img_shape[1],img_shape[0])
        img_standard = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
        image_new = img_standard
        min_, max_ = float(np.min(image_new)), float(np.max(image_new))
        image_new = (image_new - min_) / (max_ - min_)
#         print(image_new.shape)
        image_new = np.asarray([image_new])
#         print(image_new.shape)
        return image_new,image_size
    
    def overlay_img_single(self):
        demo_filename_split = os.path.splitext(self.demo_filename)[0]
        orig = self.test_fullscale_path
        fore = str(self.save_post_path+'/'+demo_filename_split+'_median27.png')
        im_orig = Image.open(orig)
        im_fore = Image.open(fore)
#           converting grayscale to rgb image
        rgb_im_fore = Image.new('RGBA', im_fore.size)
        rgb_im_fore.paste(im_fore)
        datas = rgb_im_fore.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((0, 255, 255, 100))                                               #using red mask
            else:
                newData.append((0, 0, 0, 0))                                                   #otherwise set to transparent
        rgb_im_fore.putdata(newData)
        im_orig.paste(rgb_im_fore, (92,65), rgb_im_fore)                                       #paste to the original images
        im_orig.save(str(self.save_roi_restored)+'/'+self.demo_filename+'_restored.png', 'PNG')
        self.demo_result_path = str(self.save_roi_restored + '/' + self.demo_filename + '_restored.png')
        print(self.demo_filename)
    
    def overlay_img(self):
        i = 0
        for orig, fore in zip(sorted(glob.glob(str(self.test_fullscale_path)+'/*.png')), sorted(glob.glob(str(self.save_post_path)+'/*.png'))):
            i = i+1
            no_jpgdot_orig = os.path.splitext(orig)[0]
            no_jpgdot_fore = os.path.splitext(fore)[0]
            base_orig = os.path.basename(no_jpgdot_orig)
            base_fore = os.path.basename(no_jpgdot_fore)
            im_orig = Image.open(orig)
            im_fore = Image.open(fore)
#             converting grayscale to rgb image
            rgb_im_fore = Image.new('RGBA', im_fore.size)
            rgb_im_fore.paste(im_fore)
            datas = rgb_im_fore.getdata()
            newData = []
            for item in datas:
                if item[0] == 255 and item[1] == 255 and item[2] == 255:
                    newData.append((0, 255, 255, 100))                                               #using red mask
                else:
                    newData.append((0, 0, 0, 0))                                                   #otherwise set to transparent
            rgb_im_fore.putdata(newData)
            im_orig.paste(rgb_im_fore, (92,65), rgb_im_fore)                          #paste to the original images(92,65)
            im_orig.save(str(self.save_roi_restored) +'/' +base_fore+'_restored.png', 'PNG')
            self.demo_result_path = str(self.save_roi_restored + '/' + base_fore + '_restored.png')
        print(i)
    
    def saveResult(self, npyfile, size, name, threshold=127):
        for i, item in enumerate(npyfile):
            img = item
#             print(img.shape)
            img_std = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
#             print(img_std.shape)
            if self.flag_multi_class:
                for row in range(len(img)):
                    for col in range(len(img[row])):
                        num = np.argmax(img[row][col])
                        img_std[row][col] = COLOR_DICT[num]
            else:
                for k in range(len(img)):
                    for j in range(len(img[k])):
                        num = img[k][j]          
                        if num < (threshold/255.0):
                            img_std[k][j] = background
                        else:
                            img_std[k][j] = muscle
            img_std = cv2.resize(img_std, size, interpolation=cv2.INTER_CUBIC)
            kernel = np.ones((5,5),np.uint8)
            ret,thresh_bi = cv2.threshold(img_std,127,255,cv2.THRESH_BINARY)
            median27 = cv2.medianBlur(thresh_bi,27)
            print(str(name)+'.'+str(self.img_type))
            cv2.imwrite(os.path.join(self.save_path, ("%s_predict." + self.img_type) % (name)), img_std)
            cv2.imwrite(os.path.join(self.save_post_path, ("%s_median27." + self.img_type) % (name)), median27)
