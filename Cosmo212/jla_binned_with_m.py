import numpy as np
from astropy.cosmology import FlatLambdaCDM
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import multiprocessing
import functools
import time

from Cosmo212.mcmc_helper_jla import *
from Cosmo212.analysis import analysis_2d

#Global params
Ndp=31

#MCMC params
Numch=4
Num_iter=200000

#Directories
dir_path = '../Datasets/JLA Data/jla_likelihood_v6/data/'
fname_cov = 'jla_mub_covmatrix.dat'
fname_mub = 'jla_mub.txt'
fname_for_saving='Results_JLA/'

def data_load():
    covdat = np.loadtxt(dir_path + fname_cov, skiprows=1).reshape(Ndp, Ndp)
    mumat = np.loadtxt(dir_path + fname_mub, skiprows=1, delimiter=' ')
    return mumat[:, 0], mumat[:, 1], covdat

def proposal_covariance(choice):
    if (choice=='FL2') or (choice=='OW2'):
        sig=0.02
        diag=0.02**2/5
        offdiag=-diag/2
        covmat=np.array([[diag, offdiag], [offdiag, diag]]).reshape(2, 2)
        if choice=='OW2':
            covmat=100*covmat
        return covmat
    if (choice == 'FW2'):
        '''#Yields eta=0.90
        diag1, diag2 = 0.02 ** 2 / 5, 0.07**2/5
        offdiag = -diag1 / 2
        return np.array([[diag1, offdiag], [offdiag, diag1]]).reshape(2, 2)'''
        diag1, diag2 = 0.02 **2 , 0.07 **2
        offdiag = -diag1 / 2
        covmat=np.array([[diag1, offdiag], [offdiag, diag1]]).reshape(2, 2)
        return ((2.4) ** 2.0) * covmat *10 /2.0

#mcmc
def mcmc_per_chain(ch, theta_start, choice, covar_t, fname_prefix, mu_obs, cov_inv, z_arr):
    print('Chain ' + str(ch), ': ', theta_start)
    fname = fname_prefix + 'chfinal_'+str(ch) + '.txt'
    fname_cont= fname_prefix + 'chcont_'+str(ch) + '.txt'

    theta_mat = np.zeros((Num_iter, len(theta_start)))
    theta_mat[0, :] = theta_start
    print ('Init', theta_start)

    for iter in range(1, Num_iter):
        theta_prev = theta_mat[iter - 1, :].flatten()
        theta_cand = propose(theta_prev, covar_t)
        theta_mat[iter, :] = accept_reject(mu_obs, cov_inv, z_arr, theta_cand, theta_prev, choice)
        if (iter%1000==0) & (iter>0):
            with open(fname_cont, 'ab') as f:
                print ('Chain '+str(ch)+'Iter '+str(iter))
                print_start=(int(iter/1000)-1)*1000
                np.savetxt(f, theta_mat[print_start:iter])
        if (ch==0) & (iter==20000):
            analysis_2d(fname_prefix+'chcont_', fname_for_saving, 4, choice)

    np.savetxt(fname, theta_mat, header='Itns Final Chain ' + str(ch))
    return

def main_func(choice):
    # Data
    z_arr, mu_obs, covdat = data_load()  # both check out with f2, f3
    cov_inv = np.linalg.inv(covdat)

    # Proposal covariance
    cov_t = proposal_covariance(choice)
    print(cov_t)

    # mcmc
    theta_start = init_theta_2d(Numch, choice)
    mcmc_with_ds = functools.partial(mcmc_per_chain, choice= choice, covar_t=cov_t, fname_prefix=fname_for_saving, mu_obs=mu_obs,
                                     cov_inv=cov_inv, z_arr=z_arr)
    # MP
    processes = []
    for i in range(Numch):
        p = multiprocessing.Process(target=mcmc_with_ds, args=(i, theta_start[i, :],))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()
    return

if __name__=='__main__':
    model='FW2'
    #main_func(model)

    #analysis_2d(fname_for_saving+'FW2_eta29_h70/chfinal_', fname_for_saving, 4, model) #make sure you a) uncomment above and b) check whether that code has been adjusted for jla
    analysis_2d(fname_for_saving + 'FW2_eta21_h73/chfinal_', fname_for_saving+'FW2_eta21_h73/', 4,
                model)  # make sure you a) uncomment above and b) check whether that code has been adjusted for jla
    print(4)