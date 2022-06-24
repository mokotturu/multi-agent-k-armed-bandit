import numpy as np
from scipy.optimize import minimize

def rosen(x):
	return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def main():
	x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
	res = minimize(rosen, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True}, callback=lambda xk: print(xk))
	print(res)

if __name__ == "__main__":
	main()
