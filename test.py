from scipy.sparse.csgraph import laplacian
import numpy as np

def generateP(A, kappa):
	dmax = np.max(np.sum(A, axis=0))
	L = laplacian(A, normed=False)
	M = np.shape(A)[0]
	I = np.eye(M)

	P = I - (kappa/dmax) * L
	return P

def main():
	A = np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
	kappa = 1
	res = generateP(A, kappa)
	print(res)
	
if __name__ == "__main__":
	main()