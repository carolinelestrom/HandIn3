"""
Continuous Time Finance 2 - Hand-In #3

Sunday, April 14, 2024

Anna Caroline Lestrøm Bertelsen (mlh392)
"""

### PACKAGES & SEED
###================
import numpy as np
import time
np.random.seed(3)


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
T = 1
n = 100
N = 1000
K = 100



### FUNCTIONS
###===========
def generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambdaparam, T, n):
    kappa_tilde = kappa + lambdaparam
    theta_tilde = (kappa * theta)/(kappa + lambdaparam)
    dt = T / n
    S = np.zeros(n + 1)
    S[0] = S0
    v = np.zeros(n + 1)
    v[0] = v0
    count_zero = 0

    for i in range(1, n + 1):
        Z1 = np.random.normal(0, 1)
        Z2 = np.random.normal(0, 1)
        Zv = Z1 
        Zs = rho * Z1 + np.sqrt(1 - rho**2) * Z2 
        
        dv =  kappa_tilde * (theta_tilde - v[i-1]) * dt + sigma * np.sqrt(v[i-1] * dt) * Zv + 0.25 * sigma**2 * dt * (Zv**2 - 1) 
        v[i] = v[i-1] + dv
        if v[i] <= 0:
            count_zero += 1
            v[i] = 0 

        dS = r * S[i-1] * dt + np.sqrt(max(v[i-1], 0) * dt) * S[i-1] * Zs 
        S[i] = S[i-1] + dS

    return S, count_zero



def priceHestonCallViaMilsteinMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K):
    start = time.time()
    chi = np.zeros(N)
    
    for i in range(N):
        S, count_zero = generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
        chi[i] = max(S[-1] - K, 0)
        
    call_price = np.exp(-r * T) * np.mean(chi)
    std_dev = np.std(chi) / np.sqrt(N)
    stop = time.time()
    computation_time = stop - start
    
    return call_price, std_dev, computation_time, count_zero


### COMPUTATIONS
###=============
call_price, std_dev, computation_time, count_zero = (priceHestonCallViaMilsteinMC(S0, v0, r, kappa, theta, sigma, rho, lambdaval, T, n, N, K))
S_test = generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambdaval, T, n)

### OUTPUT
###=======
print(f"Stock price for MC with Milstein scheme: {S_test}")
print(f"Call price for MC with Milstein scheme: {call_price}")
print(f"Standard deviation of call payoff for MC with Milstein scheme: {std_dev}")
print(f"Computation time for MC with Milstein scheme: {computation_time} in seconds")
print(f"Zero-Variance-Count for MC with Milstein scheme: {count_zero}")






