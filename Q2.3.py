"""
Continuous Time Finance 2 - Hand-In #3

Sunday, April 14, 2024

Anna Caroline Lestrøm Bertelsen (mlh392)
"""

### PACKAGES & SEED
###================
import numpy as np
from scipy.integrate import quad
import time
np.random.seed(571)


### PARAMETERS
###===========
S0 = 100
v0 = 0.06
r = 0.05
kappa = 1
theta = 0.06
sigma = 0.3
rho = -0.5
lambdaval = 0.01
tau = 1
K = 100




### FUNCTIONS
###===========
def characteristicFunctionHeston(u, St, vt, r, kappa, theta, sigma, rho, lambdaval, tau, j):
    i = complex(0, 1) 
    a = kappa * theta
    b1 = kappa + lambdaval - rho * sigma
    b2 = kappa + lambdaval
    u1 = 0.5
    u2 = -0.5

    if j == 1:
        b_j = b1
        u_j = u1
    else:
        b_j = b2
        u_j = u2

    d_j = np.sqrt((rho * sigma * i * u - b_j)**2 - sigma**2 * (2 * u_j * i * u - u**2))
    g_j = (b_j - rho * sigma * i * u + d_j) / (b_j - rho * sigma * i * u - d_j)
    C_j = r * i * u * tau + (a / sigma**2) * ((b_j - rho * sigma * i * u + d_j) * tau - 2 * np.log((1 - g_j * np.exp(d_j * tau)) / (1 - g_j)))
    D_j = ((b_j - rho * sigma * i * u + d_j) / sigma**2 )* ((1 - np.exp(d_j * tau)) / (1 - g_j * np.exp(d_j * tau)))

    return np.exp(C_j + D_j * vt + i * u * np.log(St))




def priceHestonCallViaOriginalFT(St, vt, r, kappa, theta, sigma, rho, lambdaval, tau, K):
    umax = 50
    integrandQj = lambda u, j: np.real(np.exp(-complex(0, 1) * u * np.log(K)) * characteristicFunctionHeston(u, St, vt, r, kappa, theta, sigma, rho, lambdaval, tau, j) / (complex(0, 1) * u))
    Q1 = 0.5 + (1 / np.pi) * quad(integrandQj, 0, umax, args=(1,))[0]
    Q2 = 0.5 + (1 / np.pi) * quad(integrandQj, 0, umax, args=(2,))[0]
    
    return St * Q1 - np.exp(-r * tau) * K * Q2



### COMPUTATIONS
###=============
start = time.time()
call_price = priceHestonCallViaOriginalFT(S0, v0, r, kappa, theta, sigma, rho, lambdaval, tau, K)
stop = time.time()
computation_time = stop - start


### OUTPUT
###=======
print(f"The call price with Heston' original formula: {call_price}")
print(f"The computing time with Heston' original formula: {computation_time} seconds")














