import numpy as np
import matplotlib.pyplot as plt
import glob
from keras.models import *
from model import create_model, residual_model, residual_to_depth_model
import cv2
from utils import *
from tqdm import tqdm
from keras.callbacks import *

limit_gpu()

print ("Loading images")
image_folder = '/data/frank/images/'
filelist = glob.glob(image_folder + '*.jpg')[:230]
images = []
for i in tqdm(range(len(filelist))):
    images.append(plt.imread(filelist[i])[:2048, :2048].astype(np.uint8))
print ("Done")
print ("Loading Validation Set")
val_x, val_y = get_val_test_data(filelist[-30:], images[:-30])
print ("Done")
print ("Creating Model")
checkpoint = ModelCheckpoint("saved_models/residual_to_depth/weights.{epoch:d}-{val_loss:f}.hdf5",
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=False,
                             mode='auto',
                             period=5)

tensorboard = TensorBoard(log_dir='./logs/residual_to_depth')
model = residual_to_depth_model()
model.summary()
print ("Done")
print ("Start Training")
history = model.fit_generator(image_generator(filelist, images),
                              steps_per_epoch=1000, 
                              epochs=100, 
                              validation_data = (val_x, val_y),
                              callbacks=[checkpoint, tensorboard])
                              # )