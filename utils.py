import numpy as np
import tensorflow as tf
import os
from keras import backend as K
import subprocess
import cv2
import matplotlib.pyplot as plt

def limit_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
    print (os.environ["CUDA_VISIBLE_DEVICES"])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    return

def mosaic(A):
    output = np.zeros(A.shape)
    H, W, D = A.shape

    R_locations = np.zeros([H, W])
    R_locations[::2, ::2] = 1

    B_locations = np.zeros([H, W])
    B_locations[1::2, 1::2] = 1

    G_locations = np.zeros([H, W])
    G_locations[::2, 1::2] = 1
    G_locations[1::2, ::2] = 1

    output = R_locations * A[:, :, 0] + G_locations * A[:, :, 1] + B_locations * A[:, :, 2]

    return output

def get_val_test_data(filelist, images):
    val_x = []
    val_y = []
    for i in range(30):
        img = images[i][:128, :128]
        bayer_img = mosaic(img)
        debayered_img = cv2.demosaicing(bayer_img.astype(np.uint8), cv2.COLOR_BayerBG2RGB)
        val_x.append(debayered_img / 255)
        val_y.append(img / 255)
    val_x = np.array(val_x)
    val_y = np.array(val_y)

    # test_x = []
    # test_y = []
    # for i in range(30):
    #     f = np.random.randint(0, len(filelist))
    #     if images[f].shape[0] < 1024 or images[f].shape[1] < 1024:
    #         continue
    #     img = images[f][:1024, :1024]
    #     bayer_img = mosaic(img)
    #     debayered_img = cv2.demosaicing(bayer_img.astype(np.uint8), cv2.COLOR_BayerBG2RGB)
    #     test_x.append(debayered_img / 255)
    #     test_y.append(img / 255)
    # test_x = np.array(test_x)
    # test_y = np.array(test_y)
    # print (val_x.shape, test_x.shape)
    return val_x, val_y

def sample_images(filelist, num_imgs=10):
    imgs = []
    for i in range(10):
        img_file = filelist[i]
        img = plt.imread(img_file)
        imgs.append(img)
    return imgs


def image_generator(filelist, images):
    h, w = 128, 128
    while 1:
        train_X = []
        train_y = []
        f = np.random.randint(0, len(filelist), 32)
        for i in f:
            img = images[i]
            x = np.random.randint(0, img.shape[1] - w)
            y = np.random.randint(0, img.shape[0] - h)
            train_y.append(img[y:y+h, x:x+w])
            bayer_img = mosaic(img[y:y+h, x:x+w])
            debayered_img = cv2.demosaicing(bayer_img.astype(np.uint8), cv2.COLOR_BayerBG2RGB)
            train_X.append(debayered_img)
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        yield (train_X / 255, train_y / 255)



