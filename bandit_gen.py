import numpy as np

# for i in range(100000):
# 	for i in range(10):
# 		print(f'{np.random.normal(0, 1.0)},', end='')
# 	print('')

# To read the numbers
with open('bandits.txt') as nums:
	for i in range(4):
		mylist = nums.readline().strip()[:-1].split(',')
		print(np.array(mylist))
