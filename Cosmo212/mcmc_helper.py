import numpy as np
from astropy.cosmology import LambdaCDM
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go

D, h=2, 0.73
'''Helper functions and playground for test runs'''

def ln_prior_2d(theta): #omega_m, omega_lam, h
    bool_prior=np.array([(theta[0]>=0) & (theta[0]<=1), (theta[1]>=0) & (theta[1]<=1), (theta[2]>=0)]) #Uniform: [0,1] for om_m, om_lam, [0,5] for h
    if np.any(bool_prior==False): #does any param lie outside range
        return (-np.inf)
    else: return 0 #all in range

def ln_likelihood_2d(mu_obs, sig_obs, z_vector, theta):
    #print (h)
    cosmo=LambdaCDM(Om0=theta[0], Ode0=theta[1], H0=100*h)
    mu_pred=cosmo.distmod(z_vector).value
    #print ('pred', )
    chi_2=np.sum(np.power(((mu_obs-mu_pred)/sig_obs), 2.0))
    return (-chi_2/2.0)

def init_theta_2d(numch):
    return np.array([np.random.uniform(0, 1, numch), np.random.uniform(0, 1, numch)]).T

def ln_prior_2d(theta): #omega_m, omega_lam, h
    bool_prior=np.array([(theta[0]>=0) & (theta[0]<=1), (theta[1]>=0) & (theta[1]<=1)]) #Uniform: [0,1] for om_m, om_lam
    if np.any(bool_prior==False): #does any param lie outside range
        return (-np.inf)
    else: return 0 #all in range

def ln_func_2d(mu_obs, sig_obs, z_vector, theta):
    pri=ln_prior_2d(theta)
    if np.isinf(pri): #0 prior
        return pri
    else: #calculate likelihood only for non-zero prior
        l=ln_likelihood_2d(mu_obs, sig_obs, z_vector, theta)
        return pri+l

def ln_prior(theta): #omega_m, omega_lam, h
    bool_prior=np.array([(theta[0]>=0) & (theta[0]<=1), (theta[1]>=0) & (theta[1]<=1), (theta[2]>=0) & (theta[2]<=5)]) #Uniform: [0,1] for om_m, om_lam, [0,5] for h
    if np.any(bool_prior==False): #does any param lie outside range
        return (-np.inf)
    else: return 0 #all in range

def ln_likelihood(mu_obs, sig_obs, z_vector, theta):
    cosmo=LambdaCDM(Om0=theta[0], Ode0=theta[1], H0=100*theta[2])
    mu_pred=cosmo.distmod(z_vector).value
    print ('pred', )
    chi_2=np.sum(np.power(((mu_obs-mu_pred)/sig_obs), 2.0))
    return (-chi_2/2.0)

def ln_func(mu_obs, sig_obs, z_vector, theta):
    pri=ln_prior(theta)
    if np.isinf(pri): #0 prior
        return pri
    else: #calculate likelihood only for non-zero prior
        l=ln_likelihood(mu_obs, sig_obs, z_vector, theta)
        return pri+l

def init_theta(numch):
    return np.array([np.random.uniform(0, 1, numch), np.random.uniform(0, 1, numch), np.random.uniform(0, 5, numch)]).T

def accept_reject(mu_obs, sig_obs, z_vector, theta_prop, theta_prev):
    if D==3:
        r=np.random.uniform(0.0, 1.0, 1)
        lf_prop=ln_func(mu_obs, sig_obs, z_vector, theta_prop)
        lf_prev=ln_func(mu_obs, sig_obs, z_vector, theta_prev)

        if (lf_prop- lf_prev > np.log(r)): #f(theta')/f(theta_k)>r
            return theta_prop
        else:
            return theta_prev
    else:
        r = np.random.uniform(0.0, 1.0, 1)
        lf_prop = ln_func_2d(mu_obs, sig_obs, z_vector, theta_prop)
        lf_prev = ln_func_2d(mu_obs, sig_obs, z_vector, theta_prev)

        if (lf_prop - lf_prev > np.log(r)):  # f(theta')/f(theta_k)>r
            return theta_prop
        else:
            return theta_prev

def get_trial_covar(covar_o): # covariance
    D = covar_o.shape[0]
    sig_o2 = np.max(np.diagonal(covar_o))
    sig_t2 = sig_o2 * ((2.4) ** 2) / D
    covar_t = covar_o * sig_t2 / sig_o2
    return covar_t

def propose (theta_prev, covar_t):
    return np.random.multivariate_normal(theta_prev, covar_t)

def propose_1dim (theta_prev, sigma):
    return float(np.random.normal(theta_prev, sigma, 1))

#ANALYSIS
def trace_2d (mat, fname_for_saving, choice):
    if choice=='FL2':
        params=['Omega_m', 'h']
    if choice == 'OW2':
        params = ['Omega_m', 'Omega_lambda']
    if choice=='FW2':
        params = ['Omega_m', 'w']
    fig, ax=plt.subplots(1, 2)
    ax=ax.ravel()
    for i in range(mat.shape[1]):
        ax[i].plot(mat[:, i])
        ax[i].set_title(params[i])
    plt.savefig(fname_for_saving)
    plt.show()
    return


def trace (mat, fname_for_saving):
    params=['Omega_m', 'Omega_lambda', 'h']
    fig, ax=plt.subplots(2, 2)
    ax=ax.ravel()
    for i in range(mat.shape[1]):
        ax[i].plot(mat[:, i])
        ax[i].set_title(params[i])
    ax[3].plot(mat[:, 2])
    ax[3].set_ylim(0, 1)
    ax[3].set_title(params[i])
    plt.savefig(fname_for_saving)
    plt.show()
    return

def histogram (mat, bool_save, fname_for_saving):
    params=['Omega_m', 'Omega_lambda']
    tuples_posn=[(1, 1), (1, 2)]
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"secondary_y": True}, {"secondary_y": True}],
                               [{"secondary_y": True}, {"secondary_y": True}]], subplot_titles=params)
    # print ('Stp', start)
    for i in range(mat.shape[1]):
        fig.add_trace(go.Histogram(x=mat[:, i], histnorm='probability', name=params[i]), row=tuples_posn[i][0], col=tuples_posn[i][1])
    fig.update_xaxes(range=[0, 1])
    fig.show()
    if bool_save:
        fig.write_image(fname_for_saving)
    '''
    fig, ax=plt.subplots(2, 2)
    ax=ax.ravel()
    for i in range(mat.shape[1]):
        ax[i].histogram(mat[:, i], bins=25)
        ax[i].set_title(params[i])
    plt.savefig(fname_for_saving)
    plt.show()'''
    return


def r_hat(fnamelist, burnin): #list of files for each chain
    matlist=[]
    for f in fnamelist:
        chain=np.loadtxt(f)
        start=int(chain.shape[0]*burnin)
        #print ('Stp', start)
        matlist.append(chain[start:, :])


    N, D, M=matlist[0].shape[0], matlist[0].shape[1], len(matlist)
    print (N, D, M)

    ch_mean_list =[np.mean(mat, axis=0) for mat in matlist]         #mean of each chain
    sm_list=[(np.var(mat, axis=0)*N / (N-1)) for mat in matlist]    #var of each chain
    w_list= np.sum(np.array(sm_list), axis=0)/M                     #avg within chain variance, Needs to be adding across chains, not params--check!, shape=D
    all_mean=np.mean(np.array(ch_mean_list), axis=0)                #mean of means of all chains, Needs to be adding across chains, not params--check!
    diff=np.array(ch_mean_list) - all_mean                          #mean of each chain - mean of all

    b_list=np.sum(np.power(diff, 2), axis=0)*N/(M-1) #shape=D       #variance of within chain means
    var_est=w_list*(N-1)/N + b_list/N #shape=D                      #v^
    r_hat=np.sqrt(var_est/w_list) #shape=D

    '''print ('Diff', diff, diff.shape)
    print ('Checks', ch_mean_list[0].shape, sm_list[0].shape, w_list.shape, all_mean.shape, b_list.shape, var_est.shape, r_hat.shape)
    print('Checks per chain: ', ch_mean_list, sm_list)
    print ('Avg across chains: ', w_list, all_mean)
    print ('B, V^, R^: ', b_list, var_est, r_hat)'''
    return r_hat

if __name__=='__main__':
    '''print ("R^, Using trial covar, V0, Bi=0: ", r_hat(['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_0.txt', 'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_0.txt'], 0.0))
    print("R^, Using trial covar, V0, Bi=50%: ", r_hat(['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_0.txt',
                                                'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_0.txt'],
                                               0.3))'''
    print("R^, Using trial covar, V1, Bi=0: ", r_hat(
        ['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_1.txt',
         'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_1.txt'], 0.3))
    '''print("R^, Using trial covar, V1, Bi=50%: ", r_hat(
        ['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_1.txt',
         'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_1.txt'], 0.3))
    print("R^, 1 dim, V0, Bi=0: ", r_hat(
        ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_0.txt',
         'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_0.txt'], 0.0))
    print("R^, 1 dim, V0, Bi=50%: ", r_hat(
        ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_0.txt',
         'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_0.txt'], 0.3))
    print("R^, 1 dim, V1, Bi=0%: ", r_hat(
        ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_1.txt',
         'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_1.txt'], 0.0))
    print("R^, 1 dim, V1, Bi=50%: ", r_hat(
        ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_1.txt',
         'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_1.txt'], 0.3))
    print("R^, 1 dim, V2, Bi=0%: ", r_hat(
        ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_2.txt',
         'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_2.txt'], 0.0))
    print("R^, 1 dim, V2, Bi=50%: ", r_hat(
        ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_2.txt',
         'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_2.txt'], 0.3))'''
    dirname='Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/'
    histogram(dirname+'covar_1.txt', 0.3, dirname+'hist1_b30.png')
    
    '''with open('Results_MCMC/Trial_Outputs/Rhat.txt', 'ab') as f:
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_0.txt',
             'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_0.txt'], 0.0), header="R^, Using trial covar, V0, Bi=0: ")
        np.savetxt(f, r_hat(['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_0.txt',
                     'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_0.txt'],
                    0.1), header="R^, Using trial covar, V0, Bi=10%: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_1.txt',
             'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_1.txt'], 0.0), header="R^, Using trial covar, V1, Bi=0: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain1/covar_1.txt',
             'Results_MCMC/Trial_Outputs/Using_trial_covar/10k/chain2/covar_1.txt'], 0.1), header="R^, Using trial covar, V1, Bi=10%: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_0.txt',
             'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_0.txt'], 0.0), header="R^, 1 dim, V0, Bi=0: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_0.txt',
             'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_0.txt'], 0.1), header="R^, 1 dim, V0, Bi=10%: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_1.txt',
             'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_1.txt'], 0.0), header="R^, 1 dim, V1, Bi=0%: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_1.txt',
             'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_1.txt'], 0.1), header="R^, 1 dim, V1, Bi=10%: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_2.txt',
             'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_2.txt'], 0.0), header="R^, 1 dim, V2, Bi=0%: ")
        np.savetxt(f, r_hat(
            ['Results_MCMC/Trial_Outputs/10k_1dim/chain1/covar_2.txt',
             'Results_MCMC/Trial_Outputs/10k_1dim/chain2/covar_2.txt'], 0.1), header="R^, 1 dim, V2, Bi=10%: ")
'''





