###########################################
##### Scan over simulation parameters #####
###########################################

import sys
sys.path.append('functions/')
import simulation
import nitrite_toxicity_model as ntm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint
import denitfit as dn
import glob
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import warnings
warnings.filterwarnings("ignore")

#define command line parameters
#to reproduce Fig. 7 use:
#python num_roots_k_I_tox_min_full.py 1.25 2.0 20 2 5 0.1 0.5 20 10 20 12 0.1 
print('usage is python num_roots_k_I_tox_min.py kmin kmax num_ks rA_thresh f Itox_min_min Itox_min_max num_Itox_mins Itox_max n_strains n_cycles rd_max')


#format plots
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.labelsize'] = 7
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 8
plt.rcParams['font.family'] = 'Arial'

#define array of curvatures k over which to scan based on command line input 
kmin = float(sys.argv[1])
kmax = float(sys.argv[2])
num_ks = int(sys.argv[3])
ks = np.logspace(np.log10(kmin), np.log10(kmax), num_ks)
ks = np.flip(ks)

#nitrate uptake rate at which strain becomes susceptible to nitrite toxicity
rA_thresh = float(sys.argv[4])

#scaling factor for uptake rates
f = float(sys.argv[5])

#define array of nitrite toxicity thresholds I_1/2 over which to scan based on command line input
I_tox_min_min = float(sys.argv[6])
I_tox_min_max = float(sys.argv[7])
num_I_tox_min = int(sys.argv[8])
I_tox_mins = np.linspace(I_tox_min_min, I_tox_min_max, num_I_tox_min)

#for sake of simulation set I_tox_max to 10 to XXX FIX  
I_tox_max = float(sys.argv[9])

#number of strains
n_strains = int(sys.argv[10])

#maximum death rate
rd_max = float(sys.argv[12])

#number of enrichment cycles 
n_cycles = int(sys.argv[11])


num_roots = np.zeros((len(ks), len(I_tox_mins))) #used to classify outcome of simulation 
breakthroughs = np.zeros((len(ks), len(I_tox_mins)))

#parameter space scan
for i in range(len(ks)):
    print(str(i+1)+'/'+str(len(ks)))
    for j in range(len(I_tox_mins)):
        #get outcome for given parameter values
        #get_phen_dist automatically generates relative abundance distributions and coculture enrichment simulations of the most abundant phenotypes for each condition
        num_roots[i,j] = simulation.get_phen_dist(ks[i], rA_thresh, f, I_tox_mins[j], I_tox_max, n_strains, 1e-2, rd_max, n_cycles)

#plot parameter scan
fig, ax = plt.subplots(1,1,figsize=(3,3), sharex = True, sharey = True)

ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

value = -1

masked_array = np.ma.masked_where(num_roots == value, num_roots)

#set up colormap
cmap = mpl.colors.ListedColormap(['purple', 'tab:green', 'tab:orange', 'tab:red', 'cyan', 'maroon','pink' ])
bounds = [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


im = ax.imshow(num_roots, cmap = cmap, alpha = 0.8, vmin = -3, vmax = 3)

# Show all ticks and label them with the respective list entries
y_labels = []
iterator = 0
nth_val = 5
y_positions = []
for k in ks:
    if iterator % nth_val == 0:
        y_labels.append(str(round(k,2)))
        y_positions.append(iterator)
    iterator = iterator + 1
x_labels = []
iterator = 0
x_positions = []
for Itoxmin in I_tox_mins:
    if iterator % nth_val == 0:
        x_labels.append(str(round(Itoxmin,2)))
        x_positions.append(iterator)
    iterator = iterator + 1

ax.set_yticks(y_positions)
ax.set_yticklabels(labels=y_labels)
ax.set_xticks(x_positions)
ax.set_xticklabels(labels=x_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
ax.set_xlabel('I_tox_min (mM)')
ax.set_ylabel('k')
ax.set_title(' n_strains = ' + str(n_strains) + ', rA_thresh = ' + str(rA_thresh) + ', n_cycles = ' + str(n_cycles))
fig.tight_layout()
fig.colorbar(im)

#save plots
plt.savefig('phase_plots/num_roots_k_I_tox_min_k='+str(kmin)+'-'+str(kmax)+'_num_ks='+str(num_ks)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_Itox_min='+str(I_tox_min_min)+'-'+str(I_tox_min_max)+'_num_Itox_mins='+str(num_I_tox_min)+'_rd_max='+str(rd_max)+'_Itox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_n_cycles='+str(n_cycles)+'.svg')
plt.savefig('phase_plots/num_roots_k_I_tox_min_k='+str(kmin)+'-'+str(kmax)+'_num_ks='+str(num_ks)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_Itox_min='+str(I_tox_min_min)+'-'+str(I_tox_min_max)+'_num_Itox_mins='+str(num_I_tox_min)+'_rd_max='+str(rd_max)+'_Itox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_n_cycles='+str(n_cycles)+'.png')
