import sys
sys.path.append('functions/')
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import bmgdata as bd
import glob
from matplotlib.pyplot import cm

pHs = [6, 7.3]

#choose initial conditions to show in bold
f0_pseudos_to_plot = [0, 0.03, 0.5, 0.97, 1]
alpha_small = 0.25

#load OD data, sorted by cycle
filenames = sorted(glob.glob('data/KC_OD600_*_FF_cycle*_OD_endpoint_*.csv'))
meta_filename = 'data/plate1_metadata.csv'
meta = pd.read_csv(meta_filename,index_col=0).dropna(how='all')  #import metadata

#generate array of initial relative abundance conditions
rhizo_f0s = np.setdiff1d(np.unique(meta["rhizo_f0"]),['O2_control', 'blank', 'presumed_blank'])

#set up colormap to indicate initial relative abundances
colors = ['mediumslateblue', 'tab:purple', 'hotpink'] 
cmap = LinearSegmentedColormap.from_list(
        "Custom", colors, N=20)

#define plot parameters
dot_size = 4
linewidth = 1.5

#start at cycle 1
cycle = 1

ODs = np.empty((len(filenames), len(pHs), len(rhizo_f0s)))
ODs[:] = np.nan
OD_stds = np.empty((len(filenames), len(pHs), len(rhizo_f0s)))
OD_stds[:] = np.nan

#generate OD matrices
for filename in filenames:
    OD_data_frame = bd.read_abs_endpoint(filename)
    OD_data_frame = OD_data_frame[OD_data_frame.index.isin(meta.index)]
    blank_idx = (meta["rhizo_f0"] == "blank")
    avg_blank_ODs = np.mean(OD_data_frame[blank_idx])
    for i in range(len(pHs)):
        for j in range(len(rhizo_f0s)):
            idx = (meta["pH"] == pHs[i]) & (meta["rhizo_f0"] == str(rhizo_f0s[j]))
            OD_val = np.mean(OD_data_frame[idx]) - avg_blank_ODs
            OD_std = np.std(OD_data_frame[idx])
            ODs[cycle-1,i,j] = OD_val
            OD_stds[cycle-1,i,j] = OD_std
    #iterate over cycles
    cycle = cycle + 1


plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12

fig, axs = plt.subplots(4,2,figsize=(10,8), sharex = True, sharey = False)

pHs = [6, 7.3]

cycle = 1
num_cycles = 4
last_t = 0
t_to_add = 0
endpoint_ts = [0]

#plot metabolite dynamics
for k in range(num_cycles):
    last_t = last_t + t_to_add
    met_file = 'data/cycle'+str(cycle)+'_metabolite_dynamics.pkl'
    all_experiments = pd.read_pickle(met_file)
    for experiments in all_experiments:
        for experiment in experiments:
            for i in range(len(pHs)):
                for j in range(len(rhizo_f0s)):
                    if (float(rhizo_f0s[j]) in f0_pseudos_to_plot):
                        label = True
                        alpha = 1
                    else:
                        label = False
                        alpha = alpha_small
                    if experiment.f0_pseudo == rhizo_f0s[j]:
                        if experiment.pH == pHs[i]:
                            if float(rhizo_f0s[j]) == 0:
                                color_val = 'tab:blue'
                            elif float(rhizo_f0s[j]) == 1:
                                color_val = 'tab:orange'
                            else:
                                color_val =  cmap(float(rhizo_f0s[j]))
                            
                            axs[2][i].errorbar(last_t+experiment.t,np.mean(experiment.A, axis=0), yerr = np.std(experiment.A, axis=0), marker = 'o', markersize = dot_size, ecolor = color_val, color = color_val, linewidth = linewidth, alpha = alpha)
                            axs[3][i].errorbar(last_t+experiment.t,np.mean(experiment.I, axis=0), yerr = np.std(experiment.I, axis=0), marker = 'o',markersize = dot_size,ecolor = color_val, color = color_val, linewidth = linewidth, alpha = alpha)

                            axs[2][0].set_ylabel('NO3 [mM]')
                            axs[3][0].set_ylabel('NO2 [mM]')
                            axs[2][i].set_ylim([-0.1, 2.2])
                            axs[3][i].set_ylim([-0.1, 2.2])
                            t_to_add = experiment.t[-1]
    #iterate cycle and time
    cycle=cycle+1
    endpoint_ts.append(last_t+ t_to_add)

#plot OD dynamics
for i in range(len(pHs)):
    for j in range(len(rhizo_f0s)):
        if (float(rhizo_f0s[j]) in f0_pseudos_to_plot):
            label = True
            alpha = 1
        else:
            label = False
            alpha = alpha_small
        if float(rhizo_f0s[j]) == 0:
            color_val = 'tab:blue'
        elif float(rhizo_f0s[j]) == 1:
            color_val = 'tab:orange'
        else:
            color_val =  cmap(float(rhizo_f0s[j]))
        if label:
            axs[0+1][i].errorbar(endpoint_ts[1:], ODs[:,i,j], yerr = OD_stds[:,i,j], marker = 'o', markersize = dot_size, ecolor = color_val, color = color_val, linewidth = linewidth, label = 'f0_Nar='+str(round(1 - float(rhizo_f0s[j]),2)), alpha = alpha)
        else:
            axs[0+1][i].errorbar(endpoint_ts[1:], ODs[:,i,j], yerr = OD_stds[:,i,j], marker = 'o', markersize = dot_size, ecolor = color_val, color = color_val, linewidth = linewidth, alpha = alpha)
            
        axs[0+1][0].legend(prop={'size': 7})
        axs[0+1][0].set_ylabel('OD600')
        axs[0+1][i].set_ylim([0, 0.17])


#load relative abundance dynamics
f = np.load('data/rel_ab_dynamics.npz')
nar_rel_abs = f['nar_rel_abs']
nar_rel_abs_stds = f['nar_rel_abs_stds']
pHs_vals = f['pHs_vals']
rhizo_f0s_vals = f['rhizo_f0s_vals']
endpt_vals = []

#plot relative abundance dynamics
for j in range(len(pHs_vals)):
    for k in range(len(rhizo_f0s_vals)):
        if (float(rhizo_f0s_vals[k]) in f0_pseudos_to_plot):
            label = True
            alpha = 1
        else:
            label = False
            alpha = alpha_small
        if rhizo_f0s_vals[k] == 0:
            color_val = 'tab:blue'
            avg=True
        elif rhizo_f0s_vals[k] == 1:
            color_val = 'tab:orange'
            avg=False
        else:
            color_val = cmap(rhizo_f0s_vals[k])
            avg = True
        if pHs_vals[j] == 7.3:
            if nar_rel_abs[j,k,-1] != 0:
                axs[0][j].errorbar([endpoint_ts[0],endpoint_ts[-1]], [nar_rel_abs[j,k,0],nar_rel_abs[j,k,-1]], yerr = [nar_rel_abs_stds[j,k,0],nar_rel_abs_stds[j,k,-1]], marker = 'o', markersize = dot_size, color = color_val, ecolor = color_val, linewidth = linewidth, alpha = alpha)
        else:
            axs[0][j].errorbar(endpoint_ts, nar_rel_abs[j,k,:], yerr = nar_rel_abs_stds[j,k,:], marker = 'o', markersize = dot_size, color = color_val,  ecolor = color_val, linewidth = linewidth, alpha = alpha)
            endpt_vals.append(nar_rel_abs_stds[j,k,-1])
        axs[0][j].set_title('pH = ' + str(pHs_vals[j]))#, fontsize = 8.0)
        axs[0][0].set_ylabel('PD Nar+ rel. ab.')#, fontsize = 8.0)


for i in range(4):
    axs[i][1].axes.yaxis.set_ticklabels([])
    
#plt.subplots_adjust(left=0.15, bottom=0.17, right=0.9, top=0.83, wspace=0.02, hspace=0.06)
fig.add_subplot(111, frameon=False)

# hide tick and tick label of the big axes                                                                                                                                                     
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)
plt.xlabel("t [h]")
fig.tight_layout()
plt.savefig('Fig5_v3.svg')
plt.savefig('Fig5_v3.png')
plt.show()
