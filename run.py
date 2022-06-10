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
	M, _ = np.shape(A)
	I = np.eye(M)

	P = I - (kappa/(dmax)) * L
	return P

def calcGraphEps(P):
	M, _ = P.shape
	eigvalarr = np.sort(np.linalg.eigvals(P))

	print(eigvalarr)
	eigvalarr = eigvalarr[:-1]

	temp = np.sqrt(M) * np.sum(np.abs(eigvalarr) / (1 - np.abs(eigvalarr)))

	return temp

# def a()

def calcNodalEps(P):
	M, _ = P.shape
	eigvalarr = np.linalg.eigvals(P)
	eigvalarrP = eigvalarr[1:]
	eigvalarrJ = eigvalarr[2:]

	temp = np.sqrt(M) * np.sum(np.sum(np.abs(eigvalarrP * eigvalarrJ) / (1 - np.abs(eigvalarrP * eigvalarrJ) + 1e-6)))

	return temp

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
		'Network 1',
		'All-to-All',
		'Ring',
		'House',
		'Line',
		'Star'
	]
	Amats = [
		np.array([
			[0, 1, 1, 1],
			[1, 0, 1, 0],
			[1, 1, 0, 0],
			[1, 0, 0, 0],
		]),
		np.array([
			[0, 1, 1, 1, 1],
			[1, 0, 1, 1, 1],
			[1, 1, 0, 1, 1],
			[1, 1, 1, 0, 1],
			[1, 1, 1, 1, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1],
			[1, 0, 1, 0, 0],
			[0, 1, 0, 1, 0],
			[0, 0, 1, 0, 1],
			[1, 0, 0, 1, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1],
			[1, 0, 1, 0, 1],
			[0, 1, 0, 1, 0],
			[0, 0, 1, 0, 1],
			[1, 1, 0, 1, 0],
		]),
		np.array([
			[0, 1, 0, 0, 1],
			[1, 0, 1, 0, 0],
			[0, 1, 0, 0, 0],
			[0, 0, 0, 0, 1],
			[1, 0, 0, 1, 0],
		]),
		np.array([
			[0, 0, 1, 0, 0],
			[0, 0, 1, 0, 0],
			[1, 1, 0, 1, 1],
			[0, 0, 1, 0, 0],
			[0, 0, 1, 0, 0],
		]),
	]
	kappa = 0.02
	teamColors = colorGen()

	for network, A in zip(networks, Amats):
		P = generateP(A, kappa)
		eps = calcGraphEps(P)
		print(eps)
		# result = playMultiAgent(runs, T, numArms, A.shape[0], P)
		# print(f'eps {network}:\t{eps}')

		# agentColors = colorGen()
		# plt.plot(np.mean(result, axis=0), label=network, color=next(teamColors))
		# print(f'finished {network} at {ctime(time())}')


		# # for agent-wise plot
		# for i, r in enumerate(result):
		# 	plt.plot(r, label=f'Agent {i + 1}', color=next(agentColors))

	print(f'Multi-agent simulations ended at {ctime(time())}')

	# plt.xlabel('Steps')
	# plt.ylabel('Average Regret')
	# plt.legend()
	# plt.savefig('new_fig.png')
	# plt.show()

# one multi agent run
def playMultiAgentRun(T: int, N: int, M: int, P):
	sigma_g = 1
	eta = 0.1		# try 2, 2.2
	gamma = 1.9 	# try 1.9
	f = lambda t : np.sqrt(t)
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
				rew[k, action], reg[k, action] = bandit.act(action)
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
				rew[k, action], reg[k, t] = bandit.act(action)
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
def playMultiAgent(runs: int, T: int, N: int, M: int, P):
	pool = Pool(cpu_count())

	result_objs = [pool.apply_async(playMultiAgentRun, args=(T, N, M, P)) for run in range(runs)]
	results = np.array([r.get() for r in result_objs])

	pool.close()
	pool.join()

	return np.cumsum(np.mean(results, axis=0), axis=1)

if __name__ == '__main__':
	main()
