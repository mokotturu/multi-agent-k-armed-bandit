import math
from multiprocessing import Pool, cpu_count
from time import ctime, time

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.optimize import minimize, LinearConstraint, Bounds

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
	numArms = 2
	runs = 50
	T = 250
	network = 'Four Agent Ring'
	A = np.array([
		[0, 1, 0, 1],
		[1, 0, 1, 0],
		[0, 1, 0, 1],
		[1, 0, 1, 0],
	])
	kappa = 0.02
	M, _ = A.shape

	# # to show the difference in areas under the cumulative regret graphs with different P matrices
	# PLabels = [
	# 	'Initial P',
	# 	'Final P',
	# ]
	# PMats = [
	# 	np.array([
	# 		[0.98, 0.01, 0   , 0.01],
	# 		[0.01, 0.98, 0.01, 0.  ],
	# 		[0   , 0.01, 0.98, 0.01],
	# 		[0.01, 0   , 0.01, 0.98]
	# 	]),
	# 	np.array([
	# 		[8.57953828e-01, 1.16622683e-06, 0             , 1.42045005e-01],
	# 		[1.16622683e-06, 9.31244198e-01, 6.87546356e-02, 0             ],
	# 		[0             , 6.87546356e-02, 9.14084714e-01, 1.71606505e-02],
	# 		[1.42045005e-01, 0             , 1.71606505e-02, 8.40794344e-01]
	# 	]),
	# ]

	# for label, P in zip(PLabels, PMats):
	# 	res = playMultiAgent(P, P, runs, T, numArms, M, 1, 0)
	# 	plt.plot(res, label=label)

	# plt.xlabel('Timesteps')
	# plt.ylabel('Average cumulative reward')
	# plt.legend()
	# plt.show()

	# originalP = generateP(A, kappa)
	originalP = np.array([
		[0.4, 0.3, 0  , 0.3],
		[0.3, 0.4, 0.3, 0  ],
		[0  , 0.3, 0.4, 0.3],
		[0.3, 0  , 0.3, 0.4],
	])

	# reshape to 1D for the minimize function, reshape to 2D later for the actual run
	P0 = originalP[np.triu_indices(M)]
	P0 = np.array(list(filter(lambda x: x, P0)))
	newM = P0.shape[0]

	# bnds = ((0, 1) for i in range(M**2))
	bnds = Bounds(np.zeros(newM), np.ones(newM))

	cons1 = LinearConstraint(
		np.array([
			[1, 1, 1, 0, 0, 0, 0, 0],
			[0, 1, 0, 1, 1, 0, 0, 0],
			[0, 0, 0, 0, 1, 1, 1, 0],
			[0, 0, 1, 0, 0, 0, 1, 1],
		]),
		np.ones(4),
		np.ones(4),
	)

	areas = []

	res = minimize(
		playMultiAgent,
		P0,
		args=(originalP, runs, T, numArms, M, 1, 0),
		method='trust-constr',
		bounds=bnds,
		constraints=cons1,
		# callback=lambda xk, status: print(f'{ctime(time())}\n{status}'),
		callback=lambda xk, status: areas.append(status.fun),
		options={
			'xtol': 1e-6,
			'disp': True,
			'maxiter': 100,
			'verbose': 3,
		},
	)

	print(f'Simulations ended at {ctime(time())}')

	print(res.x)
	plt.xlabel('Number of minimization iterations')
	plt.ylabel('Area under the graph of average cumulative reward')
	plt.plot(np.array(areas))
	plt.savefig('newfig.png')
	plt.show()

# one multi agent run
def playMultiAgentRun(P: np.ndarray, T: int, N: int, M: int, bandit: np.ndarray, malfunction: int, badAgent: int) -> float:
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

	# bandit = Bandit(np.random.normal(0, 1.0, N), np.full(N, 1.0))

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
				rew[k, action], reg[k, t], _ = bandit.act(action)

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

def genBandits(N: int, runs: int) -> np.ndarray:
	with open('bandits.txt') as infile:
		mylist = [Bandit(np.array([float(x) for x in next(infile).split()[:N]], dtype=np.longdouble), np.ones(N)) for run in range(runs)]
	
	return mylist

def convertP(P: np.ndarray, originalP: np.ndarray):
	M, _ = originalP.shape
	upperHalf = originalP[np.triu_indices(M)]
	newPUpperHalf = np.zeros(upperHalf.shape[0])
	newP = np.zeros_like(originalP)

	for i, elem in enumerate(upperHalf):
		if elem != 0: newPUpperHalf[i], P = P[0], P[1:]

	newP[np.triu_indices(M)] = newPUpperHalf

	return newP + newP.T - np.diag(np.diag(newP))

'''
@param runs:	number of times to repeat the simulation
@param T:		timesteps in one simulation
@param N:		number of arms
@param M:		number of agents
@param P:		P matrix

pools each run into separate processes for multiprocessing
'''
def playMultiAgent(P: np.ndarray, originalP: np.ndarray, runs: int, T: int, N: int, M: int, malfunction: int, badAgent: int):
	bandits = genBandits(N, runs)
	P = convertP(P, originalP)

	# multiprocessing for more runs
	# pool = Pool(cpu_count())
	# result_objs = [pool.apply_async(playMultiAgentRun, args=(P.reshape(M, M), T, N, M, bandits[run], malfunction, badAgent)) for run in range(runs)]
	# results = np.array([r.get() for r in result_objs])

	# pool.close()
	# pool.join()

	# iterative solution for shorter runs where multiprocessing is worse
	results = []
	for run in range(runs):
		results.append(playMultiAgentRun(P, T, N, M, bandits[run], malfunction, badAgent))
	
	return np.trapz(np.cumsum(np.mean(np.mean(results, axis=0), axis=0)))
	# return np.cumsum(np.mean(np.mean(results, axis=0), axis=0))

if __name__ == '__main__':
	main()
