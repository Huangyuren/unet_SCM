import cv2
import numpy as np
import os
import glob
from PIL import Image

muscle = [255, 255, 255]
background = [0, 0, 0]


class data_postprocess:
    def __init__(self, test_path=None, test_path_label=None, test_fullscale_path=None, save_path=None, 
            save_post_path=None, save_roi_restored=None, 
            target_rows=None, target_cols=None, img_type=None):
        self.test_path = test_path
        self.test_path_label = test_path_label
        self.test_fullscale_path = test_fullscale_path
        self.save_path = save_path
        self.save_post_path = save_post_path
        self.save_roi_restored=save_roi_restored
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.target_size = (target_rows, target_cols)
        self.img_type = img_type
    
    def testGenerator(self):
        filenames_x = os.listdir(self.test_path)
        filenames_y = os.listdir(self.test_path_label)
        filenames_x.sort()
        filenames_y.sort()
        for filename_x, filename_y in zip(filenames_x, filenames_y):
            if filename_x == ".ipynb_checkpoints" or filename_y == ".ipynb_checkpoints":
                pass
            else:
                img_x = cv2.imread(os.path.join(self.test_path, filename_x), cv2.IMREAD_COLOR)
                img_y = cv2.imread(os.path.join(self.test_path_label, filename_y), cv2.IMREAD_COLOR)
                img_x = cv2.resize(img_x, self.target_size, interpolation=cv2.INTER_CUBIC)
                img_y = cv2.resize(img_y, self.target_size, interpolation=cv2.INTER_CUBIC)
                min_, max_ = float(np.min(img_x)), float(np.max(img_x))
                min_label, max_label = float(np.min(img_y)), float(np.max(img_y))
                img_x = (img_x - min_) / (max_ - min_)
                img_y = (img_y - min_label) / (max_label - min_label)
                img_y[img_y > 0.5] = 1
                img_y[img_y <= 0.5] = 0
                #  img_x = np.reshape(img_x, img_x.shape + (1,)) if (not self.flag_multi_class) else img_x
                img_x = np.reshape(img_x, (1,) + img_x.shape)
                img_y = np.reshape(img_y, (1,) + img_y.shape)
                yield (img_x, img_y)
    
    def image_normalized(self, file_path):
        '''
            Only for visualizing Unet results.
            tif，size:*，gray64
            :param file_path: path to your images directory
            :return: single normalized image, image's size (should be 2d!)
        '''
        img = cv2.imread(file_path, cv2.IMREAD_COLOR)
        img_size = (img.shape[0], img.shape[1])
        image_new = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
        min_, max_ = float(np.min(image_new)), float(np.max(image_new))
        image_new = (image_new - min_) / (max_ - min_)
        image_new = np.asarray([image_new])
        return image_new, img_size
    
    def saveResult(self, results, size, name, threshold=127):
        for img in results:
            img_std = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if img[i, j] < (threshold/255.0):
                        img_std[i][j] = background
                    else:
                        img_std[i][j] = muscle
            img_std = cv2.resize(img_std, size, interpolation=cv2.INTER_CUBIC)
            median27 = cv2.medianBlur(img_std,27)
            cv2.imwrite(os.path.join(self.save_path, ("%s_predict." + self.img_type) % (name)), img_std)
            cv2.imwrite(os.path.join(self.save_post_path, ("%s_median27." + self.img_type) % (name)), median27)

    def overlay_img(self):
        i = 0
        for orig, fore in zip(sorted(glob.glob(str(self.test_path)+'/*.tif')), \
                              sorted(glob.glob(str(self.save_post_path)+'/*.tif'))):
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
#                    item = np.array([0,255,255,100])
                    newData.append((0, 255, 255, 100))                                               #using tiffany blue mask
                else:
#                    item = np.array([0,0,0,0])
                    newData.append((0, 0, 0, 0))                                                  #otherwise set to transparent
            rgb_im_fore.putdata(newData)
            #  im_orig.paste(rgb_im_fore, (65,65), rgb_im_fore)                          #paste to the original images(90,65)
            im_orig.paste(rgb_im_fore, (0,0), rgb_im_fore)
            im_orig.save(str(self.save_roi_restored) +'/' +base_fore+'_restored.png', 'PNG')
            self.demo_result_path = str(self.save_roi_restored + '/' + base_fore + '_restored.png')
        print("Total overlaied images:",i)
    
    def overlay_img_single(self):
        demo_filename_split = os.path.splitext(self.demo_filename)[0]
        orig = self.test_fullscale_path
        fore = str(self.save_post_path+'/'+demo_filename_split+'_median27.png')
        im_orig = Image.open(orig)
        im_fore = Image.open(fore)
        rgb_im_fore = Image.new('RGBA', im_fore.size)
        rgb_im_fore.paste(im_fore)
        datas = rgb_im_fore.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((0, 255, 255, 100))
            else:
                newData.append((0, 0, 0, 0))
        rgb_im_fore.putdata(newData)
        im_orig.paste(rgb_im_fore, (92,65), rgb_im_fore)
        im_orig.save(str(self.save_roi_restored)+'/'+self.demo_filename+'_restored.png', 'PNG')
        self.demo_result_path = str(self.save_roi_restored + '/' + self.demo_filename + '_restored.png')
        print(self.demo_filename)
        
