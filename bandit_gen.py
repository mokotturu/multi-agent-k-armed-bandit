import numpy as np

# for i in range(100000):
# 	for i in range(10):
# 		print(f'{np.random.normal(0, 1.0)} ', end='')
# 	print('')

# To read the numbers
with open('bandits.txt') as numfile:
	with open('bandits.txt') as infile:
		mylist = [np.array([float(x) for x in next(infile).split()[:2]], dtype=np.longdouble) for run in range(10)]

	print(mylist)
