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

def solidLineStyleGen():
	yield 'r-'
	yield 'g-'
	yield 'b-'
	yield 'k-'
	yield 'c-'
	yield 'y-'

def dottedLineStyleGen():
	yield 'r:'
	yield 'g:'
	yield 'b:'
	yield 'k:'
	yield 'c:'
	yield 'y:'

# run multi agent
def runMultiAgent() -> None:
	numArms = 10
	runs = 1000
	T = 500
	A = np.array([
		[0, 1, 1, 0, 0],
		[1, 0, 1, 1, 0],
		[1, 1, 0, 0, 1],
		[0, 1, 0, 0, 1],
		[0, 0, 1, 1, 0],
	])
	kappa = 0.02

	# to show the difference in areas under the cumulative regret graphs with different P matrices
	PLabels = [
		'Kappa = 0.02',
		# 'Max degree and Fastest Averaging Constant Weights',
		# 'Metropolis-Hastings Weights',
		# 'FDLA Weights',
		# 'LMSC',
	]
	PMats = [
		generateP(A, kappa),
		# np.array([
		# 	[0.33333333, 0.33333333, 0.33333333, 0.        , 0.        ],
		# 	[0.33333333, 0.        , 0.33333333, 0.33333333, 0.        ],
		# 	[0.33333333, 0.33333333, 0.        , 0.        , 0.33333333],
		# 	[0.        , 0.33333333, 0.        , 0.33333333, 0.33333333],
		# 	[0.        , 0.        , 0.33333333, 0.33333333, 0.33333333],
		# ]),
		# np.array([
		# 	[0.33333333, 0.33333333, 0.33333333, 0.        , 0.        ],
		# 	[0.33333333, 0.        , 0.33333333, 0.33333333, 0.        ],
		# 	[0.33333333, 0.33333333, 0.        , 0.        , 0.33333333],
		# 	[0.        , 0.33333333, 0.        , 0.16666667, 0.5       ],
		# 	[0.        , 0.        , 0.33333333, 0.5       , 0.16666667],
		# ]),
		# np.array([
		# 	[0.23809522, 0.38095239, 0.38095239, 0.        , 0.        ],
		# 	[0.38095239, 0.0952381 , 0.0952381 , 0.42857142, 0.        ],
		# 	[0.38095239, 0.0952381 , 0.0952381 , 0.        , 0.42857142],
		# 	[0.        , 0.42857142, 0.        , 0.28571429, 0.28571429],
		# 	[0.        , 0.        , 0.42857142, 0.28571429, 0.28571429],
		# ]),
		# np.array([
		# 	[0.36161435, 0.31919283, 0.31919283, 0.        , 0.        ],
		# 	[0.31919283, 0.17267105, 0.17267102, 0.33546511, 0.        ],
		# 	[0.31919283, 0.17267102, 0.17267105, 0.        , 0.33546511],
		# 	[0.        , 0.33546511, 0.        , 0.33226746, 0.33226743],
		# 	[0.        , 0.        , 0.33546511, 0.33226743, 0.33226746],
		# ]),
	]
	M, _ = PMats[0].shape

	solidLineStyle = solidLineStyleGen()
	dottedLineStyle = dottedLineStyleGen()

	plt.rc('font', size=22)
	plt.rc('axes', titlesize=22, labelsize=22)
	plt.rc('xtick', labelsize=22)
	plt.rc('ytick', labelsize=22)

	fileName = 'sim_data_houses_comparison.txt'
	with open(fileName, 'w') as outfile:
		outfile.write('')

	for type in ['healthy', 'faulty']:
		for i, (label, P) in enumerate(zip(PLabels, PMats)):
			malfunctions = [-1, -1] if type == 'healthy' else [0, 4]
			res = playMultiAgent(P, P, runs, T, numArms, M, malfunctions[0], malfunctions[1])
			with open(fileName, 'a') as outfile:
				outfile.write(str(res) + '\n')
			plt.plot(
				res,
				next(solidLineStyle) if type == 'healthy' else next(dottedLineStyle),
				label=f'{label} - {type}',
				lw=3,
			)
			print(f'+1 {ctime(time())}')

	# with open(fileName, 'r') as infile:
	# 	for type in ['healthy', 'faulty']:
	# 		for i, (label, P) in enumerate(zip(PLabels, PMats)):
	# 			arr = np.array([float(x) for x in next(infile).strip().split()])
	# 			plt.plot(
	# 				arr,
	# 				next(solidLineStyle) if type == 'healthy' else next(dottedLineStyle),
	# 				label=label,
	# 				lw=3
	# 			)

	print(f'Simulations ended at {ctime(time())}')
	plt.xlabel('Timesteps')
	plt.ylabel('Average cumulative regret')
	plt.title('House Network')
	# plt.yscale('log')
	plt.legend()
	plt.grid()
	plt.show()

	# optimization part ------------------------------------------------------
	# originalP = generateP(A, kappa)

	# # reshape to 1D for the minimize function, reshape to 2D later for the actual run
	# P0 = originalP[np.triu_indices(M)]
	# P0 = np.array(list(filter(lambda x: x, P0)))
	# newM = P0.shape[0]

	# # bnds = ((0, 1) for i in range(M**2))
	# bnds = Bounds(np.zeros(newM), np.ones(newM))

	# cons1 = LinearConstraint(
	# 	np.array([
	# 		# [1, 1, 1, 0, 0, 0, 0, 0],
	# 		# [0, 1, 0, 1, 1, 0, 0, 0],
	# 		# [0, 0, 0, 0, 1, 1, 1, 0],
	# 		# [0, 0, 1, 0, 0, 0, 1, 1],
	# 		[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
	# 		[0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
	# 		[0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
	# 		[0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
	# 		[0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
	# 	]),
	# 	np.ones(5),
	# 	np.ones(5),
	# )

	# areas = []

	# res = minimize(
	# 	playMultiAgent,
	# 	P0,
	# 	args=(originalP, runs, T, numArms, M, 1, 0),
	# 	method='trust-constr',
	# 	bounds=bnds,
	# 	constraints=cons1,
	# 	# callback=lambda xk, status: print(f'{ctime(time())}\n{status}'),
	# 	callback=lambda xk, status: areas.append(status.fun),
	# 	options={
	# 		'xtol': 1e-6,
	# 		'disp': True,
	# 		'maxiter': 150,
	# 		'verbose': 3,
	# 	},
	# )

	# print(f'Simulations ended at {ctime(time())}')

	# print(convertP(res.x, originalP))
	# plt.xlabel('Number of minimization iterations')
	# plt.ylabel('Area under the graph of average cumulative reward')
	# plt.plot(np.array(areas))
	# plt.savefig('newfig.png')
	# plt.show()

# one multi agent run
def playMultiAgentRun(P: np.ndarray, T: int, N: int, M: int, bandit: Bandit, malfunction: int, badAgent: int) -> float:
	sigma_g = 1		# try 10
	eta = 3.2		# try 2, 2.2, 3.2
	gamma = 2.9 	# try 1.9, 2.9
	f = lambda t : np.sqrt(np.log(t))
	G = 1 - (eta ** 2)/16

	n = np.zeros((M, N))	# number of times an arm has been selected by each agent
	s = np.zeros((M, N))	# cumulative expected reward
	xsi = np.zeros((M, N))	# number of times that arm has been selected in that timestep
	rew = np.zeros((M, N))	# reward
	reg = np.zeros((M, T))	# regret
	Q = np.zeros((M, N))

	for t in range(T):
		if t < N:
			for k in range(M):
				action = t
				rew[k, action], reg[k, t], _ = bandit.act(action)
				xsi[k, action] += 1
		else:
			for k in range(M):
				for i in range(N):
					x1 = s[k, i]/n[k, i]
					x2 = 2 * gamma/G
					x3 = (n[k, i] + f(t - 1))/(M * n[k, i])
					x4 = np.log(t - 1)/n[k, i]
					Q[k, i] = x1 + sigma_g * (np.sqrt(x2 * x3 * x4))
					# Q[k, i] = (s[k, i] / n[k, i]) + sigma_g * ((2 * gamma * (n[k, i] + f(t - 1)) * np.log(t - 1)) / (G * M * n[k, i] * n[k, i]))

				xsi[k, :] = np.zeros(N)
				rew[k, :] = np.zeros(N)

				action = np.argmax(Q[k, :])
				xsi[k, action] = 1
				rew[k, action], reg[k, t], _ = bandit.act(action)

				# faulty step
				if k == badAgent:
					if malfunction == 0: rew[k, action] *= -1 * np.abs(np.random.normal(0, 1))
					if malfunction == 1: rew[k, action] = np.random.normal(-1 * rew[k, action], 1.0)

		# update estimates using running consensus
		for i in range(N):
			n[:, i] = P @ (n[:, i] + xsi[:, i])
			s[:, i] = P @ (s[:, i] + rew[:, i])

	return reg

def genBandits(N: int, runs: int) -> np.ndarray:
	with open('bandits_10.txt') as infile:
		banditsList = [Bandit(np.array([float(x) for x in next(infile).split()[:N]], dtype=np.longdouble), np.full(N, 30)) for run in range(runs)]
	
	return banditsList

def convertP(P: np.ndarray, originalP: np.ndarray):
	M, _ = originalP.shape
	upperHalf = originalP[np.triu_indices(M)]
	newPUpperHalf = np.zeros(upperHalf.shape[0])
	newP = np.zeros_like(originalP)

	for i, elem in enumerate(upperHalf):
		if elem != 0: newPUpperHalf[i], P = P[0], P[1:]

	newP[np.triu_indices(M)] = newPUpperHalf

	return newP + newP.T - np.diag(np.diag(newP))

def playMultiAgent(P: np.ndarray, originalP: np.ndarray, runs: int, T: int, N: int, M: int, malfunction: int, badAgent: int):
	bandits = genBandits(N, runs)
	# P = convertP(P, originalP) # reshape

	# multiprocessing for more runs
	pool = Pool(cpu_count())
	result_objs = [pool.apply_async(playMultiAgentRun, args=(P, T, N, M, bandits[run], malfunction, badAgent)) for run in range(runs)]
	results = np.array([r.get() for r in result_objs])

	pool.close()
	pool.join()

	# iterative solution for shorter runs where multiprocessing is worse
	# results = [playMultiAgentRun(P, T, N, M, bandits[run], malfunction, badAgent) for run in range(runs)]
	
	# return np.trapz(np.cumsum(np.mean(np.mean(results, axis=0), axis=0)))
	# return np.cumsum(np.mean(np.mean(results, axis=0), axis=0))
	return np.mean(np.mean(np.cumsum(results, axis=2), axis=0), axis=0)

if __name__ == '__main__':
	main()
