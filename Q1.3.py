"""
Continuous Time Finance 2 - Hand-In #3

Sunday, April 14, 2024

Anna Caroline Lestrøm Bertelsen (mlh392)
"""

### PACKAGES & SEED
###================
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(571)


### LATEX
###=======
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 13
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts,amsthm}'



### PARAMETERS
###===========
param = {
    'St': 100,
    'vt': 0.06,
    'r': 0.05,
    'kappa': 1,
    'theta': 0.06,
    'sigma': 0.3,
    'rho': -0.5,
    'lambdaval': 0.01,
    'tau': 1
}
u = np.linspace(-20, 20, 400)



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
    elif j == 2:
        b_j = b2
        u_j = u2
    else:
        pass
    d_j = np.sqrt((rho * sigma * i * u - b_j)**2 - sigma**2 * (2 * u_j * i * u - u**2))
    g_j = (b_j - rho * sigma * i * u + d_j) / (b_j - rho * sigma * i * u - d_j)

    C_j = r * i * u * tau + (a / sigma**2) * ((b_j - rho * sigma * i * u + d_j) * tau - 2 * np.log((1 - g_j * np.exp(d_j * tau)) / (1 - g_j)))
    D_j = ((b_j - rho * sigma * i * u + d_j) / sigma**2) * ((1 - np.exp(d_j * tau)) / (1 - g_j * np.exp(d_j * tau)))

    return np.exp(C_j + D_j * vt + i * u * np.log(St))





### COMPUTATIONS
###=============
Psi1 = [characteristicFunctionHeston(i, j=1, **param) for i in u]
Psi2 = [characteristicFunctionHeston(i, j=2, **param) for i in u]



### PLOT
###=====
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r'\textbf{Real and Imaginary Parts of $\Psi_1(u)$ and $\Psi_2(u)$ for $u \in [-20, 20]$}', fontsize=22)

axs[0, 0].plot(u, np.real(Psi1), label='Real Part of $\Psi_1(u)$', color='hotpink')
axs[0, 0].legend()
axs[0, 0].set_xlabel('$u$', fontsize=18) 
axs[0, 0].set_ylabel('$\mathfrak{Re}(\Psi_1(u))$', fontsize=18) 

axs[0, 1].plot(u, np.imag(Psi1), label='Imaginary Part of $\Psi_1(u)$', color='cadetblue')
axs[0, 1].legend()
axs[0, 1].set_xlabel('$u$', fontsize=18) 
axs[0, 1].set_ylabel('$\mathfrak{Im}(\Psi_1(u))$', fontsize=18)

axs[1, 0].plot(u, np.real(Psi2), label='Real Part of $\Psi_2(u)$', color='salmon')
axs[1, 0].legend()
axs[1, 0].set_xlabel('$u$', fontsize=18) 
axs[1, 0].set_ylabel('$\mathfrak{Re}(\Psi_2(u))$', fontsize=18)

axs[1, 1].plot(u, np.imag(Psi2), label='Imaginary Part of $\Psi_2(u)$', color='darkolivegreen')
axs[1, 1].legend()
axs[1, 1].set_xlabel('$u$', fontsize=18) 
axs[1, 1].set_ylabel('$\mathfrak{Im}(\Psi_2(u))$', fontsize=18)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


