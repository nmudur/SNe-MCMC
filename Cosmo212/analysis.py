from matplotlib.patches import Ellipse
import matplotlib
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm, chi2

from Cosmo212.mcmc_helper import *

matplotlib.use('TkAgg')

def cov_ellipse(cov, q=None, nsig=None, **kwargs): #Source: https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations.
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation

def plot_confidence_ellipses_CL(mat, bool_save, dir_save, choice): #Confidence level ellipses
    if choice=='FL2':
        params = ['Omega_m', 'h']
    if choice=='OL2':
        params = ['Omega_m', 'Omega_lambda']
    if choice == 'FW2':
        params = ['Omega_m', 'w']
    if (choice == 'FW3') or (choice == 'OL3'):
        fig, ax = plt.subplots()
        sigcolors = [[0.683, 'red', '-'], [0.954, 'green', '--'], [0.997, 'black', ':']]
        ax.scatter(x=mat[:, 0], y=mat[:, 1], s=0.25, color='blue')
        covar = np.cov(mat.T)
        for scomb in sigcolors:
            w, h, r = cov_ellipse(covar, scomb[0])
            ellipse = Ellipse((np.mean(mat[:, 0]), np.mean(mat[:, 1])), width=w, height=h, angle=r, edgecolor=scomb[1],
                              linestyle=scomb[2], fill=False)
            ax.add_patch(ellipse)
        ax.set_xlabel(params[0])
        ax.set_ylabel(params[1])
        fig.suptitle('Confidence Ellipses')
        if bool_save:
            plt.savefig(dir_save)
        plt.show()
    print (params)
    fig, ax = plt.subplots()
    sigcolors=[[0.683, 'red', '-'], [0.954, 'green', '--'], [0.997, 'black', ':']]
    ax.scatter(x=mat[:, 0], y=mat[:, 1], s=0.25, color='blue')
    covar=np.cov(mat.T)
    for scomb in sigcolors:
        w, h, r=cov_ellipse(covar, scomb[0])
        ellipse = Ellipse((np.mean(mat[:, 0]), np.mean(mat[:, 1])), width=w, height=h, angle=r, edgecolor=scomb[1], linestyle=scomb[2], fill=False)
        ax.add_patch(ellipse)
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    fig.suptitle('Confidence Ellipses')
    if bool_save:
        plt.savefig(dir_save)
    plt.show()
    return

'''
def plot_scatter(mat, bool_save, dir_save, choice):
    if choice=='FL2':
        params = ['Omega_m', 'h']
        tuples_posn = [(0, 1),  (0, 2), (1, 2)]
        title='Scatter'
        fig, axes=plt.subplots(2, 2)
        fig.delaxes(axes[1, 1])
        ax=axes.ravel()

        for snum, title in enumerate(params):
            tup=tuples_posn[snum] #eg. (0, 2)
            print (tup)
            ax[snum].scatter(x=mat[:, tup[0]], y=mat[:, tup[1]])
            ax[snum].set_xlabel(params[tup[0]])
            ax[snum].set_ylabel(params[tup[1]])
        fig.suptitle('Joint scatter plot of parameters')
        if bool_save:
            plt.savefig(dir_save)
        plt.show()
    return'''

'''def plot_confidence_ellipses_2d(mat, bool_save, dir_save, choice): #Plots sigma confidence intervals
    if choice=='FL2':
        params = ['Omega_m', 'h']
        print (params)
        fig, ax = plt.subplots()
        sigcolors=['red', 'green', 'black']
        sigstyles=['-', '--', ':']
        ax.scatter(x=mat[:, 0], y=mat[:, 1], s=0.25, color='blue')
        [confidence_ellipse_for_axis(x=mat[:, 0], y=mat[:, 1], ax=ax, n_std=sig, label=r'$'+str(sig)+'\sigma$', edgecolor=sigcolors[ind], linestyle=sigstyles[ind]) for ind, sig in enumerate(np.arange(1, 4))]
        ax.set_xlabel(params[0])
        ax.set_ylabel(params[1])
        fig.suptitle('Confidence Ellipses')
        if bool_save:
            plt.savefig(dir_save)
        plt.show()
    return
'''


def confidence_ellipse_for_axis (x, y, ax, n_std=3.0, facecolor='none', **kwargs): #Source for this function: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

'''def plot_confidence_ellipses(mat, bool_save, dir_save):
    tuples_posn = [(0, 1), (0, 2), (1, 2)]
    title = 'Scatter'
    fig, axes = plt.subplots(2, 2)
    fig.delaxes(axes[1, 1])
    ax = axes.ravel()
    sigcolors=['red', 'green', 'black']
    sigstyles=['-', '--', ':']
    for snum, title in enumerate(params):
        tup=tuples_posn[snum] #eg. (0, 2)
        ax[snum].scatter(x=mat[:, tup[0]], y=mat[:, tup[1]], s=0.25, color='blue')
        [confidence_ellipse_for_axis(x=mat[:, tup[0]], y=mat[:, tup[1]], ax=ax[snum], n_std=sig, label=r'$'+str(sig)+'\sigma$', edgecolor=sigcolors[ind], linestyle=sigstyles[ind]) for ind, sig in enumerate(np.arange(1, 4))]
        ax[snum].set_xlabel(params[tup[0]])
        ax[snum].set_ylabel(params[tup[1]])
    fig.suptitle('Confidence Ellipses')
    if bool_save:
        plt.savefig(dir_save)
    plt.show()
    return'''

def get_weights(matrix, sav_bool, sav_name, choice):
    if choice=='FL2':
        params = ['Omega_m', 'h']
    if choice=='OL2':
        params = ['Omega_m', 'Omega_lambda']
    if choice=='FW2':
        params = ['Omega_m', 'w']
    if choice=='FW3':
        params = ['Omega_m', 'w', 'h']
    if choice=='OL3':
        params = ['Omega_m', 'Omega_lambda', 'h']
    df=pd.DataFrame({params[0]: matrix[:, 0], params[1]: matrix[:, 1]})
    weight_df=df.groupby([params[0], params[1]]).size().sort_values(ascending=False).reset_index()
    weight_df.rename(columns={0:'Weight'}, inplace=True)
    if sav_bool:
        weight_df.to_csv(sav_name)
    return weight_df


def calc_efficiency(fnamelist, cut_bool, ll, ul):
    for i, f in enumerate(fnamelist):
        mat=np.loadtxt(f)
        if cut_bool:
            mat=mat[ll:ul, :]
        chain_len=mat.shape[0]
        mat_pred=mat[:(chain_len-1), :]
        mat_suc=mat[1:, :]
        update_bool=np.count_nonzero(np.all(mat_pred==mat_suc, axis=1)) #Number of times it didnt update
        print (i, (1-(update_bool/(chain_len-1)))*100)
    return

'''def get_mean_errbars(matrix, choice): #Sigma error bars
    if choice=='FL2':
        params = ['Omega_m', 'h']
    if choice=='OL2':
        params = ['Omega_m', 'Omega_lambda']
    if choice == 'FW2':
        params = ['Omega_m', 'w']
    if choice == 'FW3':
        params = ['Omega_m', 'w', 'h']
    means=np.mean(matrix, axis=0)
    sig=np.std(matrix, axis=0)
    for i, par in enumerate(params):
        print (par, means[i], sig[i])
    return
'''
def get_mean_cis(matrix, choice):
    if choice=='FL2':
        params = ['Omega_m', 'h']
    if choice=='OL2':
        params = ['Omega_m', 'Omega_lambda']
    if choice=='FW2':
        params = ['Omega_m', 'w']
    if choice=='FW3':
        params = ['Omega_m', 'w', 'h']
    if choice=='OL3':
        params = ['Omega_m', 'Omega_lambda', 'h']
    means, sigma=np.mean(matrix, axis=0), np.std(matrix, axis=0)
    N=np.sqrt(matrix.shape[0])
    for i, par in enumerate(params):
        print (par, means[i], norm.interval(0.683, loc=means[i], scale=sigma[i])-means[i], norm.interval(0.954, loc=means[i], scale=sigma[i])-means[i], norm.interval(0.997, loc=means[i], scale=sigma[i])-means[i])
    return

def test_burnins(fname_read, Num_ch):
    bfr=np.arange(0.05, 0.25, 0.05)
    print ('Burnin Candidates', bfr)
    for burn in bfr:
        print ('Burnin', burn)
        print (('Rhat: '+str(burn)), r_hat(fname_read, burn))
    return

def analysis_mid(dir_read, dir_saving, Num_chains, choice):
    fnamelist = []
    for n in range(Num_chains):
        fnamelist.append(dir_read + str(n) + '.txt')
    print('Efficiency: ')
    calc_efficiency(fnamelist, False, int(200000 * 0.1), 200000)
    return

def trace_2d (mat, fname_for_saving, choice):
    if (choice=='FW3') or (choice=='OL3'):
        if (choice == 'FW3'):
            params = ['Omega_m', 'w', 'h']
        if (choice == 'OL3'):
            params = ['Omega_m', 'Omega_lambda', 'h']
        fig, ax = plt.subplots(2, 2)
        fig.delaxes(ax[1, 1])
        ax = ax.ravel()

        for i in range(3):
            ax[i].plot(mat[:, i])
            ax[i].set_title(params[i])
        plt.savefig(fname_for_saving)
        plt.show()
        return
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


def analysis_2d(dir_read, dir_saving, Num_chains, choice):
    fnamelist = []
    print ('Burnin, R^: ')
    for n in range(Num_chains):
        fnamelist.append(dir_read + str(n) + '.txt')
    test_burnins(fnamelist, Num_chains)
    print('Efficiency: ')
    calc_efficiency(fnamelist, False, int(200000 * 0.1), 200000)
    burnin = 0.1
    matlist = []
    for ch in range(Num_chains):
        itns = np.loadtxt(fnamelist[ch])
        print (itns.shape)
        trace_2d(itns, dir_saving + 'trace' + str(ch) + '.png', choice)
        start = int(itns.shape[0] * burnin)
        matlist.append(itns[start:, :])
    mat = np.vstack(matlist)

    #weights_df=get_weights(mat, True, dir_saving+'results_weights.csv', choice)
    #get_mean_errbars(mat, choice)
    print ('Confidence Intervals')
    #get_mean_cis(mat, choice)
    #plot_confidence_ellipses_CL(mat, True, dir_saving + 'ce_cl.png', choice)
    return


if __name__=='__main__':
    '''dirname=r'/home/nayantara/Desktop/Cosmology_Dvorkin/Project_Cosmology/Cosmo212/Results_JLA/'
    modnames=['FL2/']

    df=pd.read_csv(dirname+modnames[0]+'results_weights.csv')
    df.sort_values('Weight', ascending=False, inplace=True)
    tot=sum(df['Weight'])'''


