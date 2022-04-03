import math

import numpy as np
from scipy.sparse.csgraph import laplacian
from bandit import Bandit

def generateP(A, kappa):
	dmax = np.max(np.sum(A, axis=0))
	L = laplacian(A, normed=False)
	M = np.shape(A)[0]
	I = np.eye(M)

	P = I - (kappa/dmax) * L
	return P

def main(arms: np.ndarray, agents: np.ndarray, T):
	sigma_g = 2
	eta = 2
	gamma = 3
	f = lambda t: math.sqrt(t)

	N = np.shape(arms)
	M = np.shape(agents)

	n = np.zeros(N, M)
	s = np.zeros(N, M)

	i = np.zeros(M, T)
	s = np.zeros(M, T)

	bd = Bandit(np.random.normal(0, 1.0, N), np.full(N, 1,0))

	for t in range(T):
		if t < N:
			for k in enumerate(agents):
				i[k[0], t] = t + 1
				s[k[0], t] = bd.act(arms[t])
		else:
			for k in enumerate(agents):


if __name__ == "__main__":
	main()
