from sklearn.externals import joblib

import pandas as pd

from sys import argv

def detect_model_type(model_name):
	if 'charm' in model_name:
		return 'charm'
	elif 'sift' in model_name:
		return 'sift'
	else:
		return None

def extract_features(img_name, model_name):
	model_type = detect_model_type(model_name)
	
	if model_type == 'charm':
		#handle charm...
		pass
	elif model_type == 'sift':
		#handle sift
		pass
	else:
		raise ValueError('Unknown model type.')

if len(argv) != 3:
	print('Usage: python classify.py {image} {model}')
	exit()

#extract features
instance = extract_features(argv[1], argv[2])

#load model
model = joblib.load(argv[2])
prediction = model.predict(instance)

#print result
print('Instance {} is classified as {}.'.format(argv[1], 'neuronal cell' if prediction[0] else 'non-neuronal cell'))
