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
pd.DataFrame(data.data_matrix, columns = data.feature_names).to_csv(output)

#remove *.sig files
system('rm ' + input_dir + '/*/*.sig')
