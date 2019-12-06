'''Parallelized 4 chain mcmc for LCDM'''

import numpy as np
import pandas as pd
from Cosmo212.mcmc_helper import *
import multiprocessing
import functools
import time

np.seterr(all='raise')
#Params
D=2
Num_iter=200000
Num_chains=4
fname_for_saving='Results_MCMC/Main_LCDM/Main2/draws_'

#mcmc
def mcmc_per_chain(ch, theta_start, covar_t, fname_prefix, mu_obs, mu_sig, z_arr):
    print('Chain ' + str(ch), ': ', theta_start)
    fname = fname_prefix + 'chfinal_'+str(ch) + '.txt'
    fname_cont= fname_prefix + 'chcont_'+str(ch) + '.txt'

    theta_mat = np.zeros((Num_iter, len(theta_start)))
    theta_mat[0, :] = theta_start
    print ('Init', theta_start)

    for iter in range(1, Num_iter):
        theta_prev = theta_mat[iter - 1, :].flatten()
        theta_cand = propose(theta_prev, covar_t)
        theta_mat[iter, :] = accept_reject(mu_obs, mu_sig, z_arr, theta_cand, theta_prev)
        if (iter%1000==0) & (iter>0):
            with open(fname_cont, 'ab') as f:
                print ('Chain '+str(ch)+'Iter '+str(iter))
                print_start=(int(iter/1000)-1)*1000
                np.savetxt(f, theta_mat[print_start:iter])

    np.savetxt(fname, theta_mat, header='Itns Final Chain ' + str(ch))
    return

if __name__=='__main__':
    t1=time.time()
    #Cov calc
    fname_cov='Results_MCMC/Trial_Outputs/10k_1dim/chain1/covariances.txt'
    cov_o=np.loadtxt(fname_cov, skiprows=17)
    print (cov_o)
    var_err=0.02**2/5
    var_err_h=0.0174**2/5
    var_max=max(cov_o.diagonal())
    print (var_max)
    cov_t=cov_o/var_max
    #np.fill_diagonal(cov_t, [sigma_err**2, var_err, var_err_h])
    cov_t=np.multiply(cov_t, np.array([var_err, var_err, var_err_h]))
    vars=cov_t.diagonal()
    mult=np.abs(cov_t.min())
    cov_t=np.ones((D, D))*mult
    if D==3:
        cov_t[0, 2]=cov_t[2, 0]=cov_t[0, 2]*(-1)
    np.fill_diagonal(cov_t, vars)
    cov_t=cov_t[:2, :2]
    print (cov_t)
    '''#Dunkley et al formulation (2.1)
    cov_t = ((2.4) ** 2.0) * cov_o / D
    print (cov_t)'''

    #Load dataframe
    df_sne = pd.read_csv('sncosm09_fits/MLCS.FITRES', delim_whitespace=True, header=1, skip_blank_lines=True, error_bad_lines=False)
    df_sne=df_sne.drop('VARNAMES:', axis=1)
    theta_start=init_theta_2d(Num_chains)
    #Extract into arrays
    z_arr, mu_obs, mu_sig=df_sne['Z'].values, df_sne['MU'].values, df_sne['MUERR'].values

    mcmc_with_ds=functools.partial(mcmc_per_chain, covar_t=cov_t, fname_prefix=fname_for_saving, mu_obs=mu_obs, mu_sig=mu_sig, z_arr=z_arr)
    
    #MP
    processes = []
    for i in range(Num_chains):
        p = multiprocessing.Process(target=mcmc_with_ds, args=(i, theta_start[i, :], ))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    print ('End', time.time()-t1)

    print (4)