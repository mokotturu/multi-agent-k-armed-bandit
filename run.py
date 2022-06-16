import math
from multiprocessing import Pool, cpu_count
from time import ctime, time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import laplacian

from bandit import Bandit
from policy import EpsilonGreedyPolicy, UCBPolicy

def main() -> None:
	print(f'Simulations started at {ctime(time())}')
	runMultiAgent()
	# runUCB()

# run epsilon greedy
def runEPSG() -> None:
	numArms = 10
	runs = 1000
	T = 500

	epsilons = [0, 0.01, 0.1]

	epsGreedySims = [
		EpsilonGreedyPolicy(numArms, runs, T, epsilons[0]),
		EpsilonGreedyPolicy(numArms, runs, T, epsilons[1]),
		EpsilonGreedyPolicy(numArms, runs, T, epsilons[2]),
	]

	for i, sim in enumerate(epsGreedySims):
		result = playBasic(sim, runs, T, numArms)
		plt.plot(result, label=f'ε = {epsilons[i]}')

	print(f'Epsilon Greedy simulations ended at {ctime(time())}')

	plt.xlabel('Steps')
	plt.ylabel('Average Reward')
	plt.legend()
	plt.show()

# run UCB
def runUCB() -> None:
	numArms = 10
	runs = 10000
	T = 1000

	ucbSim = UCBPolicy(numArms, runs, T)
	result = playBasic(ucbSim, runs, T, numArms)

	print(f'UCB simulations ended at {ctime(time())}')

	plt.plot(result, label='UCB')
	plt.xlabel('Steps')
	plt.ylabel('Average Reward')
	plt.legend()
	plt.show()

# simulations for epsilon greedy and UCB algorithms
def playBasic(policy, runs: int, T: int, numArms: int):
	# rewards = np.zeros((runs, T))
	regrets = np.zeros((runs, T))

	for run in range(runs):
		regrets[run] = playBasicRun(policy, T, numArms)

	return np.cumsum(np.mean(regrets, axis=0))

# one run for epsilon greedy and UCB
def playBasicRun(policy, T: int, numArms: int):
	bandit = Bandit(np.random.normal(0, 1.0, numArms), np.full(numArms, 1.0))
	rew = np.zeros((T))
	reg = np.zeros((T))

	for t in range(T):
		rew[t], reg[t] = policy.step(bandit)

	policy.clear()

	return reg

def generateP(A, kappa):
	dmax = np.max(np.sum(A, axis=0))
	L = laplacian(A, normed=False)
	M = np.shape(A)[0]
	I = np.eye(M)

	P = I - (kappa/dmax) * L
	return P

def colorGen():
	while True:
		yield 'red'
		yield 'tab:blue'
		yield 'tab:green'
		yield 'tab:purple'
		yield 'tab:orange'
		yield 'tab:brown'
		yield 'tab:pink'
		yield 'tab:gray'
		yield 'tab:olive'
		yield 'tab:cyan'

# run multi agent
def runMultiAgent() -> None:
	numArms = 10
	runs = 10000
	T = 1000
	networks = [
		# 'Example Network 1',
		# 'All-to-All',
		# 'Ring',
		'House: Not connected',
		'House: Connected to Agent 1',
		'House: Connected to Agent 2',
		'House: Connected to Agent 3',
		'House: Connected to Agent 4',
		'House: Connected to Agent 5',
		# 'Line',
		# 'Star',
	]
	Amats = [
		# np.array([
		# 	[0, 1, 1, 1],
		# 	[1, 0, 1, 0],
		# 	[1, 1, 0, 0],
		# 	[1, 0, 0, 0],
		# ]),
		# np.array([
		# 	[0, 1, 1, 1, 1],
		# 	[1, 0, 1, 1, 1],
		# 	[1, 1, 0, 1, 1],
		# 	[1, 1, 1, 0, 1],
		# 	[1, 1, 1, 1, 0],
		# ]),
		# np.array([
		# 	[0, 1, 0, 0, 1],
		# 	[1, 0, 1, 0, 0],
		# 	[0, 1, 0, 1, 0],
		# 	[0, 0, 1, 0, 1],
		# 	[1, 0, 0, 1, 0],
		# ]),
		np.array([
			[0, 1, 0, 0, 1],
			[1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0],
			[0, 0, 1, 0, 1],
			[1, 1, 0, 1, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1, 1],
			[1, 0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0, 0],
			[0, 0, 1, 0, 1, 0],
			[1, 1, 0, 1, 0, 0],
			[1, 0, 0, 0, 0, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1, 0],
			[1, 0, 1, 0, 1, 1],
			[0, 1, 0, 1, 0, 0],
			[0, 0, 1, 0, 1, 0],
			[1, 1, 0, 1, 0, 0],
			[0, 1, 0, 0, 0, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1, 0],
			[1, 0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0, 1],
			[0, 0, 1, 0, 1, 0],
			[1, 1, 0, 1, 0, 0],
			[0, 0, 1, 0, 0, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1, 0],
			[1, 0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0, 0],
			[0, 0, 1, 0, 1, 1],
			[1, 1, 0, 1, 0, 0],
			[0, 0, 0, 1, 0, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1, 0],
			[1, 0, 1, 0, 1, 0],
			[0, 1, 0, 1, 0, 0],
			[0, 0, 1, 0, 1, 0],
			[1, 1, 0, 1, 0, 1],
			[0, 0, 0, 0, 1, 0],
		]),
		# np.array([
		# 	[0, 1, 0, 0, 1],
		# 	[1, 0, 1, 0, 0],
		# 	[0, 1, 0, 0, 0],
		# 	[0, 0, 0, 0, 1],
		# 	[1, 0, 0, 1, 0],
		# ]),
		# np.array([
		# 	[0, 0, 1, 0, 0],
		# 	[0, 0, 1, 0, 0],
		# 	[1, 1, 0, 1, 1],
		# 	[0, 0, 1, 0, 0],
		# 	[0, 0, 1, 0, 0],
		# ]),
	]
	plotTitles = np.array([
		'Team Average Cumulative Regret: No Agent 6',
		'Team Average Cumulative Regret: Connected to Agent 1',
		'Team Average Cumulative Regret: Connected to Agent 2',
		'Team Average Cumulative Regret: Connected to Agent 3',
		'Team Average Cumulative Regret: Connected to Agent 4',
		'Team Average Cumulative Regret: Connected to Agent 5',
	])
	graphLabels = np.array([
		'No Faulty Agent',
		'Faulty Agent 1',
		'Faulty Agent 2',
		'Faulty Agent 3',
		'Faulty Agent 4',
		'Faulty Agent 5',
	])

	kappa = 0.02
	rows, cols = 3, 2
	fig, ax = plt.subplots(rows, cols, sharey=True)
	ax = ax.reshape(rows * cols)

	for networkCounter, (network, A) in enumerate(zip(networks, Amats)):
		P = generateP(A, kappa)
		# PCustom = np.array([[0.7 , 0.15, 0   , 0   , 0.15],
		# 					[0.55, 0.3 , 0.15, 0   , 0   ],
		# 					[0   , 0.15, 0.7 , 0.15, 0   ],
		# 					[0   , 0   , 0.15, 0.7 , 0.15],
		# 					[0.55, 0   , 0   , 0.15, 0.3 ]])

		cases = 6
		results = []
		axColors = colorGen()
		
		M, _ = A.shape

		for i in range(cases):
			results.append(playMultiAgent(runs, T, numArms, M, P, 2, i - 1))
			print(f'finished {network} - {i} at {ctime(time())}')
		
		results = np.array(results)

		for i, result in enumerate(results):
			ax[networkCounter].plot(np.mean(result, axis=0), label=graphLabels[i], color=next(axColors))
		
		# ax1.set_title('Agent wise cumulative regret: No Faulty Agent')
		# for i, r in enumerate(result1):
		# 	ax1.plot(r, label=f'Agent {i + 1}', color=next(ax1Colors))

		# ax2.set_title('Agent wise cumulative regret: Faulty Agent 1')
		# for i, r in enumerate(result2):
		# 	ax2.plot(r, label=f'Agent {i + 1}', color=next(ax2Colors))

		# ax3.set_title('Agent wise cumulative regret: Faulty Agent 2')
		# for i, r in enumerate(result3):
		# 	ax3.plot(r, label=f'Agent {i + 1}', color=next(ax3Colors))

		# ax4.set_title('Agent wise cumulative regret: Faulty Agent 3')
		# for i, r in enumerate(result4):
		# 	ax4.plot(r, label=f'Agent {i + 1}', color=next(ax4Colors))

		# ax5.set_title('Agent wise cumulative regret: Faulty Agent 4')
		# for i, r in enumerate(result5):
		# 	ax5.plot(r, label=f'Agent {i + 1}', color=next(ax5Colors))

		# ax6.set_title('Agent wise cumulative regret: Faulty Agent 5')
		# for i, r in enumerate(result6):
		# 	ax6.plot(r, label=f'Agent {i + 1}', color=next(ax6Colors))

	print(f'Multi-agent simulations ended at {ctime(time())}')

	for i in range(len(ax)):
		ax[i].set_title(plotTitles[i])
		ax[i].legend()
		ax[i].grid()
		ax[i].tick_params(labelbottom=True, labelleft=True)

	fig.set_size_inches(15, 12)
	plt.savefig('newfig.png')
	plt.show()

# one multi agent run
def playMultiAgentRun(T: int, N: int, M: int, P, malfunction, badAgent):
	sigma_g = 10
	eta = 3.2		# try 2, 2.2
	gamma = 2.9 	# try 1.9
	f = lambda t : math.sqrt(t)
	G = lambda eta : 1 - (eta ** 2)/16

	n = np.zeros((M, N))	# number of times an arm has been selected by each agent
	s = np.zeros((M, N))	# cumulative expected reward
	xsi = np.zeros((M, N))	# number of times that arm has been selected in that timestep
	rew = np.zeros((M, N))	# reward
	reg = np.zeros((M, T))	# regret

	Q = np.zeros((M, N))

	bandit = Bandit(np.random.normal(0, 1.0, N), np.full(N, 1.0))

	for t in range(T):
		if t < N:
			for k in range(M):
				action = t
				rew[k, action], reg[k, action], _ = bandit.act(action)
				xsi[k, action] += 1
		else:
			for k in range(M):
				for i in range(N):
					x1 = s[k, i]/n[k, i]
					x2 = 2 * gamma/G(eta)
					x3 = (n[k, i] + f(t - 1))/(M * n[k, i])
					x4 = np.log(t - 1)/n[k, i]
					Q[k, i] = x1 + sigma_g * (np.sqrt(x2 * x3 * x4))

				action = np.argmax(Q[k, :])
				rew[k, action], reg[k, t], true_mean = bandit.act(action)

				# retaliation step
				if k == badAgent:
					if malfunction == 1: rew[k, action] *= -1
					if malfunction == 2: rew[k, action] = np.random.normal(-1 * rew[k, action], 1.0)

				xsi[k, action] += 1

		for i in range(N):
			n[:, i] = P @ (n[:, i] + xsi[:, i])
			s[:, i] = P @ (s[:, i] + rew[:, i])

		xsi = np.zeros((M, N))

	return reg

'''
@param runs:	number of times to repeat the simulation
@param T:		timesteps in one simulation
@param N:		number of arms
@param M:		number of agents
@param P:		P matrix

pools each run into separate processes for multiprocessing
'''
def playMultiAgent(runs: int, T: int, N: int, M: int, P, malfunction, badAgent):
	pool = Pool(cpu_count())

	result_objs = [pool.apply_async(playMultiAgentRun, args=(T, N, M, P, malfunction, badAgent)) for run in range(runs)]
	results = np.array([r.get() for r in result_objs])

	pool.close()
	pool.join()

	return np.cumsum(np.mean(results, axis=0), axis=1)

if __name__ == '__main__':
	main()
