"""
Continuous Time Finance 2 - Hand-In #3

Sunday, April 14, 2024

Anna Caroline Lestrøm Bertelsen (mlh392)
"""

### PACKAGES & SEED
###================
import numpy as np
import matplotlib.pyplot as plt


### LATEX
###=======
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amssymb,amsfonts,amsthm}'


### PARAMETERS
###===========
kappa = 1
lambdabar = 0.4 ### lambdabar > 0
rho = -0.5 ### rho < 0
sigma = 0.3
T = 10
#tau = np.array([0/12, 1/12, 2/12, 3/12, 4/12, 5/12, 6/12, 7/12, 8/12, 9/12, 10/12, 11/12, 12/12]) 
delta = 1/(252)
t = np.array([m*delta for m in range(0,(252*T) + 1)])
tau = T - t
p1 = 0.5
p2 = -0.5



### FUNCTIONS
###===========
def optimalpi(sigma, rho, lambdabar, p, kappa, tau):
    k0 = (p * lambdabar**2)/(1 - p)
    k1 = kappa - (p * lambdabar * sigma * rho)/(1 - p)
    k2 = sigma**2 + (p * sigma**2 * rho**2)/(1 - p)
    k3 = np.sqrt(k1**2 - k0 * k2)
    b = k0 * (np.exp(k3 * tau) - 1)/(np.exp(k3 * tau) * (k1 + k3) - k1 + k3)
    pi = lambdabar/(1 - p) + b * (sigma * rho)/(1 - p)

    return pi



### COMPUTATIONS AND PLOTS
###========================


### Sigma

p1 = 0.5
p2 = -0.5

piBM1 = optimalpi(sigma, rho, lambdabar, p1, kappa, tau)
piSigma11 = optimalpi(0.05, rho, lambdabar, p1, kappa, tau)
piSigma21 = optimalpi(0.5, rho, lambdabar, p1, kappa, tau)
piSigma31 = optimalpi(0.8, rho, lambdabar, p1, kappa, tau)



fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\sigma$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM1, s = 3, color = 'hotpink', marker = ".",label="BM, $\sigma = 0.3$")
p2 = ax.scatter(t, piSigma11, s = 3, color = 'cadetblue', marker = ".",label="$\sigma = 0.05$")
p3 = ax.scatter(t, piSigma21, s = 3, color = 'salmon', marker = ".",label="$\sigma = 0.5$")
p4 = ax.scatter(t, piSigma31, s = 3, color = 'darkolivegreen', marker = ".",label="$\sigma = 0.8$")
plots = [p2, p1, p3, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()


piBM2 = optimalpi(sigma, rho, lambdabar, p2, kappa, tau)
piSigma12 = optimalpi(0.05, rho, lambdabar, p2, kappa, tau)
piSigma22 = optimalpi(0.5, rho, lambdabar, p2, kappa, tau)
piSigma32 = optimalpi(0.8, rho, lambdabar, p2, kappa, tau)



fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\sigma$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.266, 0.268, 0.27, 0.272, 0.274]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM2, s = 3, color = 'hotpink', marker = ".",label="BM, $\sigma = 0.3$")
p2 = ax.scatter(t, piSigma12, s = 3, color = 'cadetblue', marker = ".",label="$\sigma = 0.05$")
p3 = ax.scatter(t, piSigma22, s = 3, color = 'salmon', marker = ".",label="$\sigma = 0.5$")
p4 = ax.scatter(t, piSigma32, s = 3, color = 'darkolivegreen', marker = ".",label="$\sigma = 0.8$")
plots = [p2, p1, p3, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()



### Rho


p1 = 0.5
p2 = -0.5

piBM1 = optimalpi(sigma, rho, lambdabar, p1, kappa, tau)
piRho11 = optimalpi(sigma, (-0.2), lambdabar, p1, kappa, tau)
piRho12 = optimalpi(sigma, (-0.8), lambdabar, p1, kappa, tau)
piRho13 = optimalpi(sigma, (-0.05), lambdabar, p1, kappa, tau)


fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\\rho$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.76, 0.77, 0.78, 0.79, 0.8]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM1, s = 3, color = 'hotpink', marker = ".",label="BM, $\\rho = -0.5$")
p2 = ax.scatter(t, piRho11, s = 3, color = 'cadetblue', marker = ".",label="$\\rho = -0.2$")
p3 = ax.scatter(t, piRho12, s = 3, color = 'salmon', marker = ".",label="$\\rho = -0.8$")
p4 = ax.scatter(t, piRho13, s = 3, color = 'darkolivegreen', marker = ".",label="$\\rho = -0.05$")
plots = [p3, p1, p2, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()




piBM2 = optimalpi(sigma, rho, lambdabar, p2, kappa, tau)
piRho21 = optimalpi(sigma, (-0.2), lambdabar, p2, kappa, tau)
piRho22 = optimalpi(sigma, (-0.8), lambdabar, p2, kappa, tau)
piRho23 = optimalpi(sigma, (-0.05), lambdabar, p2, kappa, tau)


fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\\rho$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.266, 0.268, 0.27, 0.272]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM2, s = 3, color = 'hotpink', marker = ".",label="BM, $\\rho = -0.5$")
p2 = ax.scatter(t, piRho21, s = 3, color = 'cadetblue', marker = ".",label="$\\rho = -0.2$")
p3 = ax.scatter(t, piRho22, s = 3, color = 'salmon', marker = ".",label="$\\rho = -0.8$")
p4 = ax.scatter(t, piRho23, s = 3, color = 'darkolivegreen', marker = ".",label="$\\rho = -0.05$")
plots = [p3, p1, p2, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()





### Kappa

p1 = 0.5
p2 = -0.5


piBM1 = optimalpi(sigma, rho, lambdabar, p1, kappa, tau)
piKappa11 = optimalpi(sigma, rho, lambdabar, p1, 2, tau)
piKappa12 = optimalpi(sigma, rho, lambdabar, p1, 0.5, tau)
piKappa13 = optimalpi(sigma, rho, lambdabar, p1, 3, tau)



fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\\kappa$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.75, 0.76, 0.77, 0.78, 0.79, 0.8]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM1, s = 3, color = 'hotpink', marker = ".",label="BM, $\\kappa = 1$")
p2 = ax.scatter(t, piKappa11, s = 3, color = 'cadetblue', marker = ".",label="$\\kappa = 2$")
p3 = ax.scatter(t, piKappa12, s = 3, color = 'salmon', marker = ".",label="$\\kappa = 0.5$")
p4 = ax.scatter(t, piKappa13, s = 3, color = 'darkolivegreen', marker = ".",label="$\\kappa = 3$")
plots = [p3, p1, p2, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()




p1 = 0.5
p2 = -0.5


piBM2 = optimalpi(sigma, rho, lambdabar, p2, kappa, tau)
piKappa21 = optimalpi(sigma, rho, lambdabar, p2, 2, tau)
piKappa22 = optimalpi(sigma, rho, lambdabar, p2, 0.5, tau)
piKappa23 = optimalpi(sigma, rho, lambdabar, p2, 3, tau)



fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\\kappa$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.266, 0.268, 0.27, 0.272, 0.274]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM2, s = 3, color = 'hotpink', marker = ".",label="BM, $\\kappa = 1$")
p2 = ax.scatter(t, piKappa21, s = 3, color = 'cadetblue', marker = ".",label="$\\kappa = 2$")
p3 = ax.scatter(t, piKappa22, s = 3, color = 'salmon', marker = ".",label="$\\kappa = 0.5$")
p4 = ax.scatter(t, piKappa23, s = 3, color = 'darkolivegreen', marker = ".",label="$\\kappa = 3$")
plots = [p3, p1, p2, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()




### Lambdabar


p1 = 0.5
p2 = -0.5


piBM1 = optimalpi(sigma, rho, lambdabar, p1, kappa, tau)
piLambdabar11 = optimalpi(sigma, rho, 0.1, p1, kappa, tau)
piLambdabar12 = optimalpi(sigma, rho, 0.9, p1, kappa, tau)
piLambdabar13 = optimalpi(sigma, rho, 3, p1, kappa, tau)


fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\\lambda$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.00, 1, 2, 3, 4, 5, 6]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM1, s = 3, color = 'hotpink', marker = ".",label="BM, $\overline{\lambda}  = 0.4$")
p2 = ax.scatter(t, piLambdabar11, s = 3, color = 'cadetblue', marker = ".",label="$\overline{\lambda} = 0.1$")
p3 = ax.scatter(t, piLambdabar12, s = 3, color = 'salmon', marker = ".",label="$\overline{\lambda} = 0.9$")
p4 = ax.scatter(t, piLambdabar13, s = 3, color = 'darkolivegreen', marker = ".",label="$\overline{\lambda} = 3$")
plots = [p2, p1, p3, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="center right",fontsize = 5)

plt.show()






p1 = 0.5
p2 = -0.5


piBM2 = optimalpi(sigma, rho, lambdabar, p2, kappa, tau)
piLambdabar21 = optimalpi(sigma, rho, 0.1, p2, kappa, tau)
piLambdabar22 = optimalpi(sigma, rho, 0.9, p2, kappa, tau)
piLambdabar23 = optimalpi(sigma, rho, 3, p2, kappa, tau)


fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $\\lambda$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.00, 0.5, 1, 1.5, 2, 2.5]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM2, s = 3, color = 'hotpink', marker = ".",label="BM, $\overline{\lambda}  = 0.4$")
p2 = ax.scatter(t, piLambdabar21, s = 3, color = 'cadetblue', marker = ".",label="$\overline{\lambda} = 0.1$")
p3 = ax.scatter(t, piLambdabar22, s = 3, color = 'salmon', marker = ".",label="$\overline{\lambda} = 0.9$")
p4 = ax.scatter(t, piLambdabar23, s = 3, color = 'darkolivegreen', marker = ".",label="$\overline{\lambda} = 3$")
plots = [p2, p1, p3, p4]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="center right",fontsize = 5)

plt.show()





### Utility

p = 0.5

wealth = np.array([i*1/100 for i in range(0,100*100 + 1)])
utility = wealth**p / p
p1 = -5
utility1 = wealth**p1 / p1
p2 = -2
utility2 = wealth**p2 / p2
p3 = - 0.9
utility3 = wealth**p3 / p3
p4 = -0.5
utility4 = wealth**p4 / p4
p5 = -0.1
utility5 = wealth**p5 / p5
p6 = 0.1
utility6 = wealth**p6 / p6
p7 = 0.9
utility7 = wealth**p7 / p7




fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Utility as a function of terminal wealth",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*100),int((2/5)*100),int((3/5)*100),int((4/5)*100),int(100)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Terminal wealth",fontsize = 7)
yticks1 = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Utility",fontsize = 7)
plot1 = ax.scatter(wealth, utility7, s = 3, color = 'palevioletred', marker = ".",label="$p=0.9$")
plot2 = ax.scatter(wealth, utility, s = 3, color = 'cadetblue', marker = ".",label="$p=0.5$")
plot3 = ax.scatter(wealth, utility6, s = 3, color = 'darkolivegreen', marker = ".",label="$p=0.1$")
plot4 = ax.scatter(wealth, utility5, s = 3, color = 'orchid', marker = ".",label="$p=-0.1$")
plot5 = ax.scatter(wealth, utility4, s = 3, color = 'salmon', marker = ".",label="$p=-0.5$")
plot6 = ax.scatter(wealth, utility3, s = 3, color = 'maroon', marker = ".",label="$p=-0.9$")
plot7 = ax.scatter(wealth, utility1, s = 3, color = 'yellowgreen', marker = ".",label="$p=-5$")
plots = [plot1, plot2, plot3, plot4, plot5, plot6, plot7]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()








### p


p1 = 0.5
p2 = -0.5


piBM1 = optimalpi(sigma, rho, lambdabar, p1, kappa, tau)
piBM2 = optimalpi(sigma, rho, lambdabar, p2, kappa, tau)
piP1 = optimalpi(sigma, rho, lambdabar, -2, kappa, tau)
piP5 = optimalpi(sigma, rho, lambdabar, -5, kappa, tau)
piP2 = optimalpi(sigma, rho, lambdabar, 0.8, kappa, tau)
piP3 = optimalpi(sigma, rho, lambdabar, 0.9, kappa, tau)
piP4 = optimalpi(sigma, rho, lambdabar, 0.1, kappa, tau)




fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $p$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T),int((2/5)*T),int((3/5)*T),int((4/5)*T),int(T)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.00, 1, 2, 3, 4]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p1 = ax.scatter(t, piBM1, s = 3, color = 'hotpink', marker = ".",label="BM, $p  = 0.5$")
p2 = ax.scatter(t, piP5, s = 3, color = 'cadetblue', marker = ".",label="$p = -5$")
p3 = ax.scatter(t, piP1, s = 3, color = 'palevioletred', marker = ".",label="$p = -2$")
p4 = ax.scatter(t, piP4, s = 3, color = 'orchid', marker = ".",label="$p = 0.1$")
p5 = ax.scatter(t, piP2, s = 3, color = 'salmon', marker = ".",label="$p = 0.8$")
p6 = ax.scatter(t, piP3, s = 3, color = 'darkolivegreen', marker = ".",label="$p = 0.9$")
p7 = ax.scatter(t, piBM2, s = 3, color = 'maroon', marker = ".",label="$p = -0.5$")
plots = [p2, p3, p7, p4, p1, p5, p6]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="upper left",fontsize = 5)

plt.show()



### T

p1 = 0.5
p2 = -0.5


T1 = 3
delta = 1/(252)
t1 = np.array([m*delta for m in range(0,(252*T1) + 1)])
tau1 = T1 - t1



T2 = 30
delta = 1/(252)
t2 = np.array([m*delta for m in range(0,(252*T2) + 1)])
tau2 = T2 - t2




piT21 = optimalpi(sigma, rho, lambdabar, p1, kappa, tau2)
piT01 = np.zeros(len(piT21))
piT01[0:2521] = optimalpi(sigma, rho, lambdabar, p1, kappa, tau)
piT11 = np.zeros(len(piT21))
piT11[0:757] = optimalpi(sigma, rho, lambdabar, p1, kappa, tau1)



fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $T$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T2),int((2/5)*T2),int((3/5)*T2),int((4/5)*T2),int(T2)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.77, 0.78, 0.79, 0.8]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p2 = ax.scatter(t2, piT01, s = 3, color = 'palevioletred', marker = ".",label="BM, $T=10$")
p1 = ax.scatter(t2, piT11, s = 3, color = 'cadetblue', marker = ".",label="$T=3$")
p3 = ax.scatter(t2, piT21, s = 3, color = 'darkolivegreen', marker = ".",label="$T=30$")
plots = [p1, p2, p3]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()





p1 = 0.5
p2 = -0.5



piT22 = optimalpi(sigma, rho, lambdabar, p2, kappa, tau2)
piT02 = np.zeros(len(piT22))
piT02[0:2521] = optimalpi(sigma, rho, lambdabar, p2, kappa, tau)
piT12 = np.zeros(len(piT22))
piT12[0:757] = optimalpi(sigma, rho, lambdabar, p2, kappa, tau1)



fig = plt.figure(constrained_layout=False,dpi=300,figsize=(5,3))
fig.suptitle(f"Optimal investment strategy as function of time for varying $T$",fontsize=10)
gs = fig.add_gridspec(nrows=1,ncols=1,left=0.12,bottom=0.2,right=0.88,top=0.90,wspace=0,hspace=0)
ax = fig.add_subplot(gs[0,0])
xticks = [0,int((1/5)*T2),int((2/5)*T2),int((3/5)*T2),int((4/5)*T2),int(T2)]
ax.set_xticks(xticks)
ax.set_xticklabels(xticks,fontsize = 6)
ax.set_xlim([xticks[0]-0.002,xticks[-1]+0.002])
plt.xlabel(f"Time",fontsize = 7)
yticks1 = [0.266, 0.267, 0.268, 0.269, 0.27]
ax.set_yticks(yticks1)
ax.set_yticklabels(yticks1,fontsize = 6)
ax.set_ylim([yticks1[0],yticks1[-1] + (yticks1[-1]-yticks1[0])*0.02])
plt.grid(axis = 'y', which='major', color=(0.7,0.7,0.7,0), linestyle='--')
ax.set_ylabel(f"Optimal investment strategy, $\pi^*$",fontsize = 7)
p2 = ax.scatter(t2, piT02, s = 3, color = 'palevioletred', marker = ".",label="BM, $T=10$")
p1 = ax.scatter(t2, piT12, s = 3, color = 'cadetblue', marker = ".",label="$T=3$")
p3 = ax.scatter(t2, piT22, s = 3, color = 'darkolivegreen', marker = ".",label="$T=30$")
plots = [p1, p2, p3]
labels = [item.get_label() for item in plots]
ax.legend(plots,labels,loc="lower right",fontsize = 5)

plt.show()




