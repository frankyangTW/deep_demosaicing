from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras
import keras.backend as K
import tensorflow as tf


def conv_lrelu_conv_lrelu(inputs, filters):
	conv = Conv2D(filters, [3, 3], padding='same')(inputs)
	lrelu = LeakyReLU(alpha=0.3)(conv)
	conv = Conv2D(filters, [3, 3], padding='same')(lrelu)
	lrelu = LeakyReLU(alpha=0.3)(conv)	
	return conv, lrelu

def conv_lrelu_conv_lrelu_pool(inputs, filters):
	conv, lrelu = conv_lrelu_conv_lrelu(inputs, filters)
	pool = MaxPooling2D(pool_size=(2, 2), padding='same')(lrelu)
	return conv, pool

def upconv_concat_conv_lrelu_conv_lrelu(inputs, concat, filters):
	upconv = Conv2DTranspose(filters, kernel_size=2, strides=[2, 2], padding='same')(inputs)
	upconv = Concatenate(axis=3)([upconv, concat])
	conv, lrelu = conv_lrelu_conv_lrelu(upconv, filters)
	return lrelu

def space_to_depth(x):
    return tf.space_to_depth(x, 2)

def depth_to_space(x):
	return tf.depth_to_space(x, 2)

def PSNR(y_true, y_pred):
	def log10(x):
		numerator = K.log(x)
		denominator = K.log(K.constant(10, dtype=numerator.dtype))
		return numerator / denominator

	mse = K.mean((y_pred - y_true) ** 2)
	return 10 * log10(1 / mse)

def create_model(depth=True):
	inputs = Input((None, None, 3))
	if depth:
		to_depth = Lambda(space_to_depth)(inputs)
		conv1, pool1 = conv_lrelu_conv_lrelu_pool(inputs=to_depth, filters=32)
	else:
		conv1, pool1 = conv_lrelu_conv_lrelu_pool(inputs=inputs, filters=32)
	conv2, pool2 = conv_lrelu_conv_lrelu_pool(inputs=pool1, filters=64)
	conv3, pool3 = conv_lrelu_conv_lrelu_pool(inputs=pool2, filters=128)
	conv4, lrelu = conv_lrelu_conv_lrelu(inputs=pool3, filters=256)
	lrelu = upconv_concat_conv_lrelu_conv_lrelu(lrelu, conv3, 128)
	lrelu = upconv_concat_conv_lrelu_conv_lrelu(lrelu, conv2, 64)
	lrelu = upconv_concat_conv_lrelu_conv_lrelu(lrelu, conv1, 32)

	if depth:
		out = Conv2D(12, [1, 1])(lrelu)
		out = Lambda(depth_to_space)(out)

	else:
		out = Conv2D(3, [1, 1])(lrelu)

	model = Model(input = [inputs], output = [out])

	# model.summary()

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics=[PSNR])
	
	return model

def conv_lrelu_conv_lrelu_conv_lrelu_residual(inputs, filters):
	conv = Conv2D(filters, [1, 1], padding='same')(inputs)
	lrelu = LeakyReLU(alpha=0.3)(conv)
	conv = Conv2D(filters, [3, 3], padding='same')(lrelu)
	lrelu = LeakyReLU(alpha=0.3)(conv)
	conv = Conv2D(filters, [1, 1], padding='same')(lrelu)
	lrelu = LeakyReLU(alpha=0.3)(conv)
	out = Add()([lrelu, inputs])
	return LeakyReLU(alpha=0.3)(out)

def residual_model():
	inputs = Input((None, None, 3))
	conv1 = Conv2D(32, [1, 1], padding='same')(inputs)
	conv1 = conv_lrelu_conv_lrelu_conv_lrelu_residual(inputs=conv1, filters=32)

	conv2 = Conv2D(64, [1, 1], padding='same')(conv1)
	conv2 = conv_lrelu_conv_lrelu_conv_lrelu_residual(inputs=conv2, filters=64)

	conv3 = Conv2D(128, [1, 1], padding='same')(conv2)
	conv3 = conv_lrelu_conv_lrelu_conv_lrelu_residual(inputs=conv3, filters=128)
	out = Conv2D(3, [1, 1])(conv3)

	model = Model(input = [inputs], output = [out])

	# model.summary()

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics=[PSNR])
	
	return model

def residual_to_depth_model():
	inputs = Input((None, None, 3))

	to_depth = Lambda(space_to_depth)(inputs)
	conv1 = Conv2D(32, [1, 1], padding='same')(to_depth)
	conv1 = conv_lrelu_conv_lrelu_conv_lrelu_residual(inputs=conv1, filters=32)

	conv2 = Conv2D(64, [1, 1], padding='same')(conv1)
	conv2 = conv_lrelu_conv_lrelu_conv_lrelu_residual(inputs=conv2, filters=64)
	
	conv3 = Conv2D(128, [1, 1], padding='same')(conv2)
	conv3 = conv_lrelu_conv_lrelu_conv_lrelu_residual(inputs=conv3, filters=128)

	out = Conv2D(12, [1, 1])(conv3)
	out = Lambda(depth_to_space)(out)

	model = Model(input = [inputs], output = [out])

	# model.summary()

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics=[PSNR])
	
	return model








