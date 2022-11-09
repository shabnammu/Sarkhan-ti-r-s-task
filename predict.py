
import tensorflow as tf
import numpy
from numpy import argmax
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.models import load_model

def load_image(filename):
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	img = img_to_array(img)
	img = img.reshape(1, 28, 28, 1)
	img = img.astype('float32')
	img = img / 255.0
	return img

def run_example():
	img = load_image('zero.png')
	model = load_model('final_model.h5')
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)

run_example()
