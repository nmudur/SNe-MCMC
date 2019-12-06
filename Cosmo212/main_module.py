import numpy as np
import pandas as pd
from mcmc_helper import *

'''Serial simultaneous update MCMC'''

np.seterr(all='raise')
#Params
D=3
Num_iter=1000000
Num_chains=4
'''#Trial covariance choices
c_pos1, c_pos2=np.ones((D, D))*0.3, np.ones((D, D))*0.3
np.fill_diagonal(c_pos1, 0.4)
np.fill_diagonal(c_pos2, 0.8)
c_neg1, c_neg2=np.ones((D, D))*0.3, np.ones((D, D))*0.3
np.fill_diagonal(c_neg1, -0.4)
np.fill_diagonal(c_neg2, -0.8)
covar_choices=[0.5*np.identity(D), 0.3*np.identity(D), 0.8*np.identity(D), c_pos1, c_pos2, c_neg1, c_neg2]
'''

fname_prefix='Results_MCMC/Trial_Outputs/Using_trial_covar/covar_'
fname_cov='Results_MCMC/Trial_Outputs/10k_1dim/chain1/covariances.txt'
cov_o=np.loadtxt(fname_cov, skiprows=17)
print (cov_o)

#Load dataframe
df_sne = pd.read_csv('sncosm09_fits/MLCS.FITRES', delim_whitespace=True, header=1, skip_blank_lines=True, error_bad_lines=False)
df_sne=df_sne.drop('VARNAMES:', axis=1)

#Extract into arrays
z_arr, mu_obs, mu_sig=df_sne['Z'].values, df_sne['MU'].values, df_sne['MUERR'].values

#mcmc
theta_start=init_theta() #Same init
for ch in range(2):
    for c in range(2):
        if c==0:
            covar=cov_o
        else:
            covar=((2.4)**2.0)*cov_o / D


        fname=fname_prefix+str(c)+'_'+str(ch)+'.txt'
        fname_cov_sav=fname_prefix+'all.txt'
        theta_mat = np.zeros((Num_iter, len(theta_start)))
        theta_mat[0, :]=init_theta()

        for iter in range(1, Num_iter):
            theta_prev=theta_mat[iter-1, :].flatten()
            theta_cand=propose(theta_prev, covar)
            theta_mat[iter, :]=accept_reject(mu_obs, mu_sig, z_arr, theta_cand, theta_prev)
        np.savetxt(fname, theta_mat, header='Covariance Choice '+str(c))
        with open(fname_cov_sav, "ab") as f:
            np.savetxt(f, np.cov(theta_mat.T),  header='\n Covariance '+str(c))
            print (np.cov(theta_mat.T))