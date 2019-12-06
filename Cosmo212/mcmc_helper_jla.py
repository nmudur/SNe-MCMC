import numpy as np
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, LambdaCDM
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go

D=3

def ln_likelihood_2d(mu_obs, inv_covmat, z_vector, theta, choice):
    if choice=='FL2':
        cosmo=FlatLambdaCDM(Om0=theta[0], H0=100*theta[1])
    if choice=='OL2':
        cosmo = LambdaCDM(H0=70, Om0=theta[0], Ode0=theta[1])
    if choice=='FW2':
        cosmo=FlatwCDM(H0=73.8, Om0=theta[0], w0=theta[1])
    if choice=='FW3':
        cosmo=FlatwCDM(Om0=theta[0], w0=theta[1], H0=100*theta[2])
    if choice=='OL3':
        cosmo=LambdaCDM(Om0=theta[0], Ode0=theta[1], H0=100*theta[2])
    mu_th=cosmo.distmod(z_vector).value
    r=(mu_obs-mu_th).reshape(-1, 1) #Check bracketing
    chi_2=np.sum(np.matmul(np.matmul(r.T, inv_covmat), r))
    return (-chi_2/2.0)

def ln_prior_2d(theta, choice): #omega_m, h
    if choice=='FL2':
        bool_prior=np.array([(theta[0]>=0) & (theta[0]<=1), (theta[1]>=0) & (theta[1]<=5)]) #Uniform: [0,1] for om_m, [0,5] for h
    if choice=='OL2':
        bool_prior=np.array([(theta[0]>=0) & (theta[0]<=1), (theta[0]>=0) & (theta[0]<=1)]) #Uniform: [0,1] for om_m, [0,5] for h
    if choice=='FW2':
        bool_prior = np.array([(theta[0] >= 0) & (theta[0] <= 1),
                               (theta[1] <= 0) & (theta[1] >= -5)])  # Uniform: [0,1] for om_m, [0,-5] for w
    if choice=='FW3':
        bool_prior = np.array([(theta[0] >= 0) & (theta[0] <= 1), (theta[1] <= 0) & (theta[1] >= -2),
                               (theta[2] >= 0) & (theta[2] < 2)])  # Uniform: [0,1] for om_m, w, for h
    if choice=='OL3':
        bool_prior = np.array([(theta[0] >= 0) & (theta[0] <= 1), (theta[1] >= 0) & (theta[1] <= 1),
                               (theta[2] >= 0) & (theta[2] < 2)])  # Uniform: [0,1] for om_m, w, for h
    if np.any(bool_prior==False): #does any param lie outside range
        return (-np.inf)
    else: return 0 #all in range

def ln_func_2d(mu_obs, cov_inv, z_vector, theta, choice):
    pri=ln_prior_2d(theta, choice)
    if np.isinf(pri): #prior=0
        return pri
    else: #calculate likelihood only for non-zero prior
        l=ln_likelihood_2d(mu_obs, cov_inv, z_vector, theta, choice)
        return pri+l

def init_theta_2d(numch, choice):
    if choice=='FL2':
        return np.array([np.random.uniform(0, 1, numch), np.random.uniform(0, 5, numch)]).T #O_m, h
    if choice=='OL2':
        return np.array([np.random.uniform(0, 1, numch), np.random.uniform(0, 1, numch)]).T #O_m, h
    if choice=='FW2':
        return np.array([np.random.uniform(0, 1, numch), np.random.uniform(-5, 0, numch)]).T ##O_m, w
    if choice=='FW3':
        return np.array([np.random.uniform(0, 1, numch), np.random.uniform(-2, 0, numch), np.random.uniform(0, 2, numch)]).T ##O_m, w, h
    if choice=='OL3':
        return np.array([np.random.uniform(0, 1, numch), np.random.uniform(0, 1, numch), np.random.uniform(0, 2, numch)]).T ##O_m, Om_lam, h
def propose (theta_prev, covar_t):
    return np.random.multivariate_normal(theta_prev, covar_t)

def accept_reject(mu_obs, cov_inv, z_vector, theta_prop, theta_prev, choice):
    r = np.random.uniform(0.0, 1.0, 1)
    lf_prop = ln_func_2d(mu_obs, cov_inv, z_vector, theta_prop, choice)
    lf_prev = ln_func_2d(mu_obs, cov_inv, z_vector, theta_prev, choice)

    if (lf_prop - lf_prev > np.log(r)):  # f(theta')/f(theta_k)>r
        return theta_prop
    else:
        return theta_prev
