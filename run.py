import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from bandit import Bandit

def main() -> None:
	numArms = 2
	epsilons = np.array([0.1, 0.01, 0])

	timesteps = 1000 # timesteps
	samples = 2000

	values = np.zeros((len(epsilons), samples, timesteps))
	regrets = np.zeros((len(epsilons), samples, timesteps))

	# 2000 randomly generated k-bandit problems
	for run in tqdm(range(samples)):
		# one run for each epsilon
		runSample(values, regrets, numArms, epsilons, timesteps, run)

	for eps in range(len(epsilons)):
		plt.plot(np.mean(values[eps, :, :], axis=0), label = f'ε = {epsilons[eps]}')

	plt.xlabel('Steps')
	plt.ylabel('Average Reward')
	plt.ylim([0, 1])
	plt.legend()
	plt.show()

def runSample(values: np.ndarray, regrets: np.ndarray, numArms: int, epsilons: np.ndarray, timestep: int, run: int) -> None:
	# create new bandit for every run
	means = np.random.normal(0, 1.0, numArms)
	# means = np.array([0.1, 0.5])
	# print(means)
	bd = Bandit(means, np.full(numArms, 1.0))
	est_actvals = np.zeros((len(epsilons), numArms))
	arms = [i for i in range(numArms)]

	for i in range(timestep): # 1000 timesteps
		for eps in range(len(epsilons)): # get action for each epsilon
			arm_to_pull = np.argmax(est_actvals[eps]) if np.random.rand() >= epsilons[eps] else np.random.choice(arms)
			reward = bd.act(arm_to_pull)

			values[eps, run, i] = reward['value']
			regrets[eps, run, i] = reward['regret']
			est_actvals[eps, arm_to_pull] += 0.1 * (reward['value'] - est_actvals[eps, arm_to_pull])

if __name__ == '__main__':
	main()
