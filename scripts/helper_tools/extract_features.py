import pandas as pd
from wndcharm.FeatureSpace import *
from os.path import isdir
from os import system
from sys import argv

input_dir = 'data_reduced'
output = 'data.csv'

if len(argv) >= 2 and isdir(argv[1]):
	input_dir = argv[1]
elif len(argv) >= 2:
	print('Directory ' + argv[1] + ' doesn\'t exist.')
	exit()

if len(argv) >= 3:
	output = argv[2]

data = FeatureSpace.NewFromDirectory(input_dir, n_jobs=True)
data_pd = pd.DataFrame(data.data_matrix, columns = data.feature_names)
data_pd['class'] = [item for sublist in data.ground_truth_labels for item in sublist]
data_pd.to_csv(output, index= False)

#remove *.sig files
system('rm ' + input_dir + '/*/*.sig')
