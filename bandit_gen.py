import numpy as np

def dumpNumbers():
	bandits = 100000
	arms = 50

	'''
	To dump numbers
	prints to console, do py bandit_gen.py > output.txt to dump into a file
	'''
	for b in range(bandits):
		for i in range(arms):
			print(f'{np.random.normal(0, 10)} ', end='')
		print('')

def readNumbers():
	# To read the numbers
	with open('bandits_10.txt') as infile:
		mylist = [np.array([float(x) for x in next(infile).split()[:30]], dtype=np.longdouble) for run in range(10)]

	print(mylist)

if __name__ == '__main__':
	dumpNumbers()
	# readNumbers()
