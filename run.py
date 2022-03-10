from bandit import Bandit
import numpy as np
import matplotlib.pyplot as plt
import time

startTime = time.time()

numArms = 10
epsilons = [0.1, 0.01, 0]

vals = [[] for i in range(len(epsilons))]
regrets = [[] for i in range(len(epsilons))]
est_actvals = np.array([[0.0] * numArms for i in range(len(epsilons))])

for run in range(2000):	# 2000 randomly generated k-bandit problems
	# create new bandit for every run
	bd1 = Bandit(numArms, np.random.normal(0, 1.0, numArms), np.full(numArms, 1.0))
	for eps in range(len(epsilons)):
		# initial action value is 0
		vals[eps].append([0])
		regrets[eps].append([0])
	for i in range(999):	# 1000 timesteps; 1st timestep is 0
		for eps in range(len(epsilons)):	# get action for each epsilon
			if np.random.rand() < epsilons[eps]:
				# random action
				arm_to_pull = np.random.randint(numArms)
			else:
				# pull the arm with highest mean
				arm_to_pull = np.argmax(est_actvals[eps])

			act_val = bd1.act(arm_to_pull)
			vals[eps][run].append(act_val['value'])
			regrets[eps][run].append(act_val['regret'])
			est_actvals[eps][arm_to_pull] += (1 / (i + 2)) * (act_val['value'] - est_actvals[eps][arm_to_pull])

endTime = time.time()	# time took to complete all simulations

print(f'Took {endTime - startTime} seconds to finish all simulations')

for eps in range(len(epsilons)):
	plt.plot(np.average(np.array(vals[eps]), 0), label = f'ε = {epsilons[eps]}')

plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.legend()
plt.show()
