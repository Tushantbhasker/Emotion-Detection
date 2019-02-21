import numpy as np 
import keras.models
from keras.models import model_from_json
import tensorflow as tf 
from scipy.misc import imsave, imshow, imresize


def init():
	# Opening the json save file which has the weights.
	json_file = open('model.json','r')
	model_json = json_file.read()
	json_file.close()

	load_model = model_from_json(model_json)
	#Loading the weights
	# loaded_model.load_weights("model.h5")
	load_model.load_weights("model.h5")
	print("Model Is Loaded")

	load_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	graph = tf.get_default_graph()

	return load_model,graph