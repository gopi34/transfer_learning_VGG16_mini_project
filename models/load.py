import numpy as np
import keras.models
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow
import tensorflow as tf
import keras.backend as K

config = K.tf.ConfigProto()
config.gpu_options.allow_growth = True
session = K.tf.Session(config=config)


def init(): 
	json_file = open('/home/gopi34/Desktop/mini_project/models/model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	
	#load weights into new model
	loaded_model.load_weights("/home/gopi34/Desktop/mini_project/models/model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	
	#initialize the loaded model and weights in default graph	
	graph = tf.get_default_graph()

	return loaded_model, graph


