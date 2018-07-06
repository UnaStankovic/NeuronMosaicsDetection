#This file reduces number of pictures in each folder. Each folder contains partitions number of pictures. 
#For example, if we had 10000 pictures and partitions = 10,they would be divided into 10 folders with 1000 pictures in it.
from os import makedirs, walk
from os.path import isdir, join
from shutil import copy, rmtree
from sys import argv

input_dir = 'reduced_data'
output_dir = 'output_data'
partitions = 10

if len(argv) >= 2 and isdir(argv[1]):
	input_dir = argv[1]
elif len(argv) >= 2:
	print('Directory ' + argv[1] + ' doesn\'t exist.')
	exit()

if len(argv) >= 3:
	output_dir = argv[2]

if len(argv) >= 4:
	partitions = int(argv[3])

output_dirs_mask = dict((output_dir + '_' + str(i), False) for i in range(0, partitions))

for (i, img_name) in enumerate([join(dp, f) for dp, dn, fn in walk(input_dir) for f in fn if f.endswith(".tif")]):
	dir_name = output_dir + '_' + str(i % partitions)

	if output_dirs_mask[dir_name] == False:
		if isdir(dir_name):
			rmtree(dir_name)

		makedirs(join(dir_name, 'true'))
		makedirs(join(dir_name, 'false'))
		output_dirs_mask[dir_name] = True

	if 'true' in img_name:
		copy(img_name, join(dir_name, 'true'))
	elif 'false' in img_name:
		copy(img_name, join(dir_name, 'false'))
