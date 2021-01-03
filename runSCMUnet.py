#!/usr/bin/env python
# coding: utf-8

# In[7]:


from model import *
from DataPostprocess import *
from DataPreprocess import *
from lr_reducer import *
from cosine_annealing import CosineAnnealingScheduler
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import keras
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
import os
import argparse


parser = argparse.ArgumentParser()

#  basic model parameters
parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--image_type', type=str, default='tif')
parser.add_argument('--T_i', type=int, default=100)
parser.add_argument('--increaseTMAX', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--input_u_wi', type=int, default=256)
parser.add_argument('--input_u_he', type=int, default=256)
parser.add_argument('--test_path', type=str, default='data/SCM/test/crop_image')
parser.add_argument('--test_path_label', type=str, default='data/SCM/test/crop_label')
parser.add_argument('--test_fullscale_path', type=str, default='data/SCM/test/origin_imageTIF')
parser.add_argument('--save_path', type=str, default='testResult/test_unet1/result_unet')
parser.add_argument('--save_post_path', type=str, default='testResult/test_unet1/resultPost_unet')
parser.add_argument('--save_roi_restored', type=str, default='testResult/test_unet1/ROIrestored_unet')
parser.add_argument('--train_path', type=str, default='data/SCM/train')
parser.add_argument('--valid_path', type=str, default='data/SCM/validation')
parser.add_argument('--image_folder', type=str, default='image')
parser.add_argument('--label_folder', type=str, default='label')
parser.add_argument('--valid_image_folder', type=str, default='image')
parser.add_argument('--valid_label_folder', type=str, default='label')
parser.add_argument('--model_dest', type=str, default='trainResult/model/model_unet.hdf5')
parser.add_argument('--CSVLogger_dest', type=str, default='trainResult/csvLogger/csvLogger_unet.log')
parser.add_argument('--TensorBoard_dest', type=str, default='trainResult/tensorboard/graph_unet')
parser.add_argument('--LR_patience', type=int, default=15)
parser.add_argument('--EarlyStopping_patience', type=int, default=60)
parser.add_argument('--loss_fig_dest', type=str, default='trainResult/result_plot/loss_unet.png')
parser.add_argument('--dice_fig_dest', type=str, default='trainResult/result_plot/dice_unet.png')
parser.add_argument('--evaluation_steps', type=int, default=90)
args = parser.parse_args()

# python runSCMUnet.py \
# --mode test \
# --batch_size 8 \
# --input_u_wi 256 \
# --input_u_he 256 \
# --test_path HNC34/HNC34_img \
# --test_path_label HNC34/HNC34_label \
# --save_path testResult/forPaper_unet20200223_1080/HNC34/result_unet20200223_1080 \
# --save_post_path testResult/forPaper_unet20200223_1080/HNC34/resultPost_unet20200223_1080 \
# --save_roi_restored testResult/forPaper_unet20200223_1080/HNC34/ROIrestored_unet20200223_1080 \
# --model_dest trainResult/model/model_unet20200223_1080.hdf5 \
# --evaluation_steps 12

#  commands for testing models:
 # python runSCMUnet.py \
 # --mode test \
 # --input_u_wi 256 \
 # --input_u_he 256 \
 # --test_path data/SCM/test/crop_image \
 # --test_fullscale_path data/SCM/test/origin_imageTIF \
 # --save_path testResult/test_unet20200218/result_unet20200218_1 \
 # --save_post_path testResult/test_unet20200218/resultPost_unet20200218_1 \
 # --save_roi_restored testResult/test_unet20200218/ROIrestored_unet20200218_1 \
 # --model_dest trainResult/model/model_unet20200218.hdf5

#  commands for training models:
#  python runSCMUnet.py \
#  --mode train \
#  --batch_size 64 \
#  --input_u_wi 64 \
#  --input_u_he 64 \
#  --model_dest trainResult/model/model_unet20200220_1080.hdf5 \
#  --CSVLogger_dest trainResult/csvLogger/csvLogger_unet20200220_1080.log \
#  --TensorBoard_dest trainResult/tensorboard/graph_unet20200220_1080 \
#  --LR_patience 20 \
#  --EarlyStopping_patience 80 \
#  --loss_fig_dest trainResult/result_plot/loss_unet20200220_1080.png \
#  --dice_fig_dest trainResult/result_plot/dice_unet20200220_1080.png

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def test():
    dp = data_postprocess(test_path=args.test_path, test_path_label=args.test_path_label, save_path=args.save_path,
                         test_fullscale_path=args.test_fullscale_path,
                         save_post_path=args.save_post_path, save_roi_restored=args.save_roi_restored, 
                         target_rows = args.input_u_wi, target_cols = args.input_u_he, img_type = args.image_type)
    test_data = dp.testGenerator()
    model = load_model(args.model_dest, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
    score, acc = model.evaluate(test_data, steps=args.evaluation_steps)
    try:
      os.makedirs(args.save_path, exist_ok=False)
      os.makedirs(args.save_post_path, exist_ok=False)
      os.makedirs(args.save_roi_restored, exist_ok=False)
      print("path creat successfully.")
    except FileExistsError:
      print("path existed.")
    for i, name in enumerate(os.listdir(dp.test_path)):
      if name == '.ipynb_checkpoints':
          print('.ipynb_checkpoints occur.')
      else:
          image_path = os.path.join(dp.test_path, name)
          # print("Image path:", image_path)
          #  resize and do simplist normalization
          img_new, img_size = dp.image_normalized(image_path)
          results = model.predict(img_new)
          #  resize back to its origin size and do bluring
          dp.saveResult(results, img_size, name.split('.')[0])
    dp.overlay_img()
    print("Final test set score, loss: {}, acc: {}.".format(score, acc))

def train():
    try:
        os.makedirs(args.TensorBoard_dest, exist_ok=False)
        print("Path creat successfully.")
    except FileExistsError:
        print("Path existed.")
    dp = data_preprocess(train_path=args.train_path, image_folder=args.image_folder, label_folder=args.label_folder,
                             valid_path=args.valid_path, valid_image_folder=args.valid_image_folder,
                             valid_label_folder=args.valid_label_folder,
                             target_rows = args.input_u_wi, target_cols = args.input_u_he)
    train_data = dp.trainGenerator(batch_size=args.batch_size)
    valid_data = dp.validation_load(batch_size=args.batch_size)
    model = unet_v1(input_size = (args.input_u_wi, args.input_u_he, 3))
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.EarlyStopping_patience)
    csv_logger = CSVLogger(args.CSVLogger_dest, separator=',', append=False)
    tbcallback = TensorBoard(log_dir=args.TensorBoard_dest, histogram_freq=0, batch_size=args.batch_size, write_graph=True, write_grads=True, 
            write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    reduce_lr = LrReducer(monitor='val_loss', patience=args.LR_patience, reduce_rate=0.2,  verbose=1)
    model_checkpoint = ModelCheckpoint(args.model_dest, monitor='val_loss',verbose=1, save_best_only=True)
    if args.increaseTMAX:
        callbacks = [model_checkpoint, tbcallback, csv_logger, reduce_lr, es, CosineAnnealingScheduler(T_i=args.T_i, increaseTMAX=args.increaseTMAX, 
            eta_max=1e-2, eta_min=1e-6)]
    else:
        callbacks = [model_checkpoint, tbcallback, csv_logger, reduce_lr, es]
    start = time.time()
    his = model.fit_generator(train_data, steps_per_epoch=50,
                        validation_data=valid_data, validation_steps=10,
                        epochs = 2000, callbacks=callbacks) 
    end = time.time()
    print('training time amount: '+str(end - start))

    # plot training history:loss
    plt.figure()
    plt.plot(his.history['loss'], label='loss')
    plt.plot(his.history['val_loss'], label='val_loss')
    plt.title("Training and validation dice loss")
    plt.xlabel("epoch #")
    plt.ylabel("dice loss")
    plt.legend()
    plt.savefig(args.loss_fig_dest)
    #  plt.show()

    # plot training history:dice
    plt.figure()
    plt.plot(his.history['dice_coef'], label='dice_coef')
    plt.plot(his.history['val_dice_coef'], label='val_dice_coef')
    plt.title("Training and validation dice score")
    plt.xlabel("epoch #")
    plt.ylabel("dice score")
    plt.legend()
    plt.savefig(args.dice_fig_dest)
    #  plt.show()


if __name__ == "__main__":
    if args.mode == "train":
        train()
    else:
        test()




