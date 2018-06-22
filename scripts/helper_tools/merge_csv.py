from sys import argv
from os.path import isfile
import pandas as pd

result = pd.DataFrame()

for f in argv[1:]:
	if isfile(f):
		result = result.append(pd.read_csv(f), ignore_index = True)

result.to_csv('data.csv', index = False)
