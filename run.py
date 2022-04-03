import matplotlib.pyplot as plt
import numpy as np

from policy import EpsilonGreedyPolicy, UCBPolicy

def main() -> None:
	numArms = 10
	samples = 2000
	T = 1000

	epsilons = [0, 0.01, 0.1]

	ucbSim = UCBPolicy(numArms, samples, T)
	ucbSimRes = ucbSim.run()

	epsGreedySims = [
		EpsilonGreedyPolicy(numArms, samples, T, epsilons[0]),
		EpsilonGreedyPolicy(numArms, samples, T, epsilons[1]),
		EpsilonGreedyPolicy(numArms, samples, T, epsilons[2]),
	]

	for simItr in enumerate(epsGreedySims):
		res = simItr[1].run()
		plt.plot(res, label=f'ε = {epsilons[simItr[0]]}')
	
	plt.plot(ucbSimRes, label='UCB')
	plt.xlabel('Steps')
	plt.ylabel('Average Reward')
	plt.legend()
	plt.show()

# def play(bandit, Policy, samples):
	# for run in range(samples):
		# res = Policy(bandit)

if __name__ == '__main__':
	main()
