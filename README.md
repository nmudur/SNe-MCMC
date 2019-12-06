# SNe-MCMC

The code in this repo is organized as follows:
* **Cosmo212**: The main directory containing the code bank, results.
* GetDist Combined Final: Contains GetDist scripts for combined confidence ellipses.
* JLA Explore.ipynb: Examines binned SNe dataset and evaluates likelihood on a grid of points in the range, identifying the best fit. This serves as a test of the likelihood function before starting MCMC chains.

## Cosmo212:
* analysis.py: Contains functions for plotting and calculating convergence statistics on the chains.
* jla_mcmc_binned.py: Main routine running the MCMC chains.
* mcmc_helper_jla.py: Contains MCMC helper functions for evaluating the prior, likelihood and acceptance/rejection ratios.

### Plots
### Results_JLA:
MCMC runs for different cosmological paradigms and efficiency values. Each contains a Summary file containing analysis statistics for the 4 chains. The chains were not added to these folders due to size considerations. The folders are named as: modeln_etaxx, where n indicates the number of parameters being estimated in the model, and xx is the acceptance fraction for the run. The main runs, whose results are reported in the paper correspond to the following directories:
* FL2: Flat Lambda CDM, 2 parameter run
* FW3_Dunkley_eta17: Flat wCDM, 3 parameter run
* FW2_eta21_h73: Flat wCDM, 2 parameter run, H0=73.8 km (Mpc.s)-1
* FW2_eta29_h70: Flat wCDM, 2 parameter run, H0=70 km (Mpc.s)-1
* OL3_eta23: Open Lambda CDM, 3 parameter run
* OL2: Open Lambda CDM, 2 parameter run


