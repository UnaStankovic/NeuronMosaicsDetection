from os import walk
from os.path import isdir
from sys import argv

input_dir = 'data_reduced'
total = 10

if len(argv) >= 2 and isdir(argv[1]):
	input_dir = argv[1]
elif len(argv) >= 2:
	print('Directory ' + argv[1] + ' doesn\'t exist.')
	exit()

if len(argv) >= 3:
	total = int(argv[2])

print('%.2f%%' % (100.0 * len([_ for _, _, fn in walk(input_dir) for f in fn if f.endswith('.sig')]) / total))
