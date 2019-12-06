import numpy as np
import pandas as pd
from mcmc_helper import *

'''One dimensional update'''

np.seterr(all='raise')
#Params
D=3
Num_iter=D*10000

#Trial covariance choices
'''c_pos1, c_pos2=np.ones((D, D))*0.3, np.ones((D, D))*0.3
np.fill_diagonal(c_pos1, 0.4)
np.fill_diagonal(c_pos2, 0.8)
c_neg1, c_neg2=np.ones((D, D))*0.3, np.ones((D, D))*0.3
np.fill_diagonal(c_neg1, -0.4)
np.fill_diagonal(c_neg2, -0.8)'''

#covar_choices=[0.5*np.identity(D), 0.3*np.identity(D), 0.8*np.identity(D), c_pos1, c_pos2, c_neg1, c_neg2]
sigma_choices=[np.ones(D)*0.5, np.ones(D)*0.25, np.ones(D)*0.8]
fname_prefix='Results_MCMC/Trial_Outputs/covar_'
fname_cov='Results_MCMC/Trial_Outputs/covariances.txt'

#Load dataframe
df_sne = pd.read_csv('sncosm09_fits/MLCS.FITRES', delim_whitespace=True, header=1, skip_blank_lines=True, error_bad_lines=False)
df_sne=df_sne.drop('VARNAMES:', axis=1)

#Extract into arrays
z_arr, mu_obs, mu_sig=df_sne['Z'].values, df_sne['MU'].values, df_sne['MUERR'].values

#mcmc
theta_start=init_theta() #Same init
for c in range(len(sigma_choices)):
    siglist=sigma_choices[c]
    fname=fname_prefix+str(c)+'.txt'

    theta_mat = np.zeros((Num_iter, len(theta_start)))
    theta_mat[0, :]=init_theta()

    for iter in range(1, Num_iter):
        theta_prev=theta_mat[iter-1, :].flatten() #all three

        if (iter%3)==1: #omega_m update
            theta_prop = propose_1dim(theta_prev[0], siglist[0])
            theta_cand=np.array([theta_prop, theta_prev[1], theta_prev[2]])
        elif (iter%3)==2: #omega_lam update
            theta_prop = propose_1dim(theta_prev[1], siglist[1])
            theta_cand = np.array([theta_prev[0], theta_prop, theta_prev[2]])
        else: #h update
            theta_prop = propose_1dim(theta_prev[2], siglist[2])
            theta_cand = np.array([theta_prev[0], theta_prev[1], theta_prop])

        theta_mat[iter, :]=accept_reject(mu_obs, mu_sig, z_arr, theta_cand, theta_prev)
    np.savetxt(fname, theta_mat, header='Covariance Choice '+str(c))
    with open(fname_cov, "ab") as f:
        np.savetxt(f, np.cov(theta_mat.T),  header='\n Covariance '+str(c))