# Show plots inline, and load main getdist plot module and samples class
from __future__ import print_function
import numpy as np
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, MCSamples
import getdist
# use this *after* importing getdist if you want to use interactive plots
# %matplotlib notebook
import matplotlib.pyplot as plt
import IPython
print('GetDist Version: %s, Matplotlib version: %s'%(getdist.__version__, plt.matplotlib.__version__))
# matplotlib 2 may not work very well without usetex on, can uncomment
# plt.rcParams['text.usetex']=True

#Getting conf intervals using gd
import numpy as np
from getdist import MCSamples, plots

#samples=MCSamples(root=r'/home/nayantara/Desktop/Cosmology_Dvorkin/Project_Cosmology/Datasets/Combined chains/chains_OL2/jla_h70')
Num_chains=4
matlist=[]
burnin=0.1
dirname=r'/home/nayantara/Desktop/Cosmology_Dvorkin/Project_Cosmology/Cosmo212/Results_JLA/OL2/'
names=['omegam', 'omegal']
labels=['\Omega_m', '\Omega_\Lambda']

for ch in range(Num_chains):
    itns = np.loadtxt(dirname+'chfinal_'+str(ch)+'.txt')
    print (itns.shape)
    start = int(itns.shape[0] * burnin)
    matlist.append(itns[start:, :])
mat=np.vstack(matlist)
#mat[:, 1] = mat[:, 1]*100 #for hubble
samp_sne=MCSamples(samples=mat, names = names, labels = labels)

#Planck plotting (alone)
g=plots.get_single_plotter(chain_dir=r'/home/nayantara/Desktop/Cosmology_Dvorkin/Project_Cosmology/Datasets/Combined chains/chains_FL2')
roots = ['base_plikHM_TTTEEE_lowl_lowE_lensing']
pairs = [['omegam', 'H0']]
g.plots_2d(roots, param_pairs=pairs, filled=True, shaded=False)
g.export()

#Planck
names=['omegam', 'H0']
labels=['\Omega_m', 'H_0']
mat_pl=np.loadtxt('base_plikHM_TTTEEE_lowl_lowE_lensing_1.txt', usecols=(29, 31)) #has 96 columns
names_pl=np.loadtxt('base_plikHM_TTTEEE_lowl_lowE_lensing.paramnames', dtype=str, delimiter=';') #has 94 columns
samp_pl=MCSamples(samples=mat_pl, names=names, labels=labels)

#JLA
Num_chains=4
matlist=[]
burnin=0.1
dirname=r'/home/nayantara/Desktop/Cosmology_Dvorkin/Project_Cosmology/Datasets/Combined chains/chains_FL2/'
for ch in range(1, (Num_chains+1)):
    itns = np.loadtxt(dirname+'chfinal_'+str(ch)+'.txt')
    print (itns.shape)
    start = int(itns.shape[0] * burnin)
    matlist.append(itns[start:, :])
mat=np.vstack(matlist)
mat[:, 1] = mat[:, 1]*100
samp_sne=MCSamples(samples=mat, names = names, labels = labels)


# Filled 2D comparison plot with legend
g = plots.get_single_plotter(width_inch=4, ratio=1)
g.plot_2d([samp_pl, samp_sne], 'omegam', 'H0', filled=True)
g.add_legend(['Planck', 'JLA'], colored_text=True)


#USE THE GETDIST WEBSITE DOCUMENTING FILE FORMATS