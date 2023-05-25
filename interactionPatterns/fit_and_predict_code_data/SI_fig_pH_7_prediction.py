##########################
## construct SI figure ###
##########################


import sys
sys.path.append('functions/')
import pandas as pd
import numpy as np
import griess as gr
import bmgdata as bd
import matplotlib.pyplot as plt
import glob
import denitfit as dn
import pickle
import copy
from pprint import pprint
import nitrite_toxicity_model as ntm 




linewidth = 4.0
alpha = 1
strain_strs = ['NarG', 'NapA']
colors = ['tab:blue', 'tab:orange', 'tab:green']
conditions = [[2,0], [1,0], [0.5, 1.5], [1, 0.5]]

experiment_file = 'CRM_predict_cycle2_experiments.pkl'
all_experiments_10 = pd.read_pickle(experiment_file)

pfit_NarG = np.load('fits/NarG_fit_pH=7.3_no_offset.npz')['pfit']
pfit_NapA = np.load('fits/NapA_fit_pH=7.3_no_offset.npz')['pfit']
print('NarG parameters:')
print(pfit_NarG)
print('')
print('NapA parameters:')
print(pfit_NapA)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12


fig, axs = plt.subplots(4,2,figsize=(5,7), sharex = True, sharey = True)
fig.tight_layout()
markersize = 10.0
col_idx = 0
pH = 7.3
for strain_str in strain_strs:
    for iterator in range(0,4):
        for experiments in all_experiments_10:
            for experiment in experiments:
                if experiment.pH == pH and experiment.ID == strain_str and experiment.I0 == conditions[iterator][1] and experiment.A0 == conditions[iterator][0]:
                    if len(experiment.A) > 0:
                        A = conditions[iterator][0]
                        I = conditions[iterator][1]
                        color = colors[col_idx]                      
                        axs[iterator][col_idx].plot(experiment.t, np.mean(experiment.A, axis = 0), 'o', label = 'NO3',  alpha = alpha, color = color)
                        
                        axs[iterator][col_idx].vlines(experiment.t, np.mean(experiment.A, axis = 0) - np.std(experiment.A, axis = 0),np.mean(experiment.A, axis = 0) + np.std(experiment.A, axis = 0), color = color, alpha = alpha)
                        
                        ts = np.linspace(0,75, 1000)
                        if experiment.ID == 'NarG':
                            y0 = [experiment.N0, 0, np.mean(experiment.A, axis = 0)[0], np.mean(experiment.I, axis = 0)[0]]

                            yh = ntm.denitODE(y0,ts,pfit_NarG,1)
                            N_1 = yh[:,0]
                            N_1_dead = yh[:,1]

                            A_vals = yh[:,2]
                            I_vals = yh[:,3]

                            axs[iterator][col_idx].plot(ts+experiment.t[0], A_vals,  color = 'tab:blue', alpha = alpha)

                            axs[iterator][col_idx].plot(ts+experiment.t[0], I_vals,'-.',  color = 'tab:blue', alpha = alpha)
                            square = 0
                            num_vals = 0
                            for i in range(len(experiment.t)):
                                t_idx = np.argmin(np.abs(ts + experiment.t[0] - experiment.t[i]))
                                for j in range(len(experiment.A[:,i])):
                                    square = square + (experiment.A[j,i] - A_vals[t_idx])**2
                                    square = square + (experiment.I[j,i] - I_vals[t_idx])**2
                                    num_vals = num_vals + 2
                            mean = square / num_vals
                            rms = np.sqrt(mean)
                            axs[0][col_idx].set_title(strain_str)
                            if iterator > 0:
                                pass
                        elif experiment.ID == 'NapA':                 
                            y0 = [experiment.N0, 0, np.mean(experiment.A, axis = 0)[0], np.mean(experiment.I, axis = 0)[0]]

                            yh = ntm.denitODE(y0,ts,pfit_NapA,1)
                            N_1 = yh[:,0]                                                                                                                                                                            \

                            N_1_dead = yh[:,1]

                            A_vals = yh[:,2]
                            I_vals = yh[:,3]

                            axs[iterator][col_idx].plot(ts+experiment.t[0], A_vals, color = 'tab:orange', alpha = alpha)

                            axs[iterator][col_idx].plot(ts+experiment.t[0], I_vals,'-.', color = 'tab:orange', alpha = alpha)

                            square = 0
                            num_vals = 0
                            for i in range(len(experiment.t)):
                                t_idx = np.argmin(np.abs(ts + experiment.t[0] - experiment.t[i]))
                                for j in range(len(experiment.A[:,i])):
                                    square = square + (experiment.A[j,i] - A_vals[t_idx])**2
                                    square = square + (experiment.I[j,i] - I_vals[t_idx])**2
                                    num_vals = num_vals + 2
                            mean = square / num_vals
                            rms = np.sqrt(mean)
                            axs[0][col_idx].set_title(strain_str)
                        else:
                            y0 = [experiment.N0/2.0, 0,experiment.N0/2.0, 0, np.mean(experiment.A, axis = 0)[0], np.mean(experiment.I, axis = 0)[0]]
                            yh = ntm.denitODE(y0,ts,[pfit_NarG,pfit_NapA],2)
                            N_1 = yh[:,0]                                                                                                                    
                            N_1_dead = yh[:,1]

                            A_vals = yh[:,4]
                            I_vals = yh[:,5]

                            axs[iterator][col_idx].plot(ts+experiment.t[0], A_vals, linewidth = linewidth, color = 'tab:green', alpha = alpha)

                            axs[iterator][col_idx].plot(ts+experiment.t[0], I_vals,'-.', linewidth = linewidth, color = 'tab:green', alpha = alpha)

                            square = 0
                            num_vals = 0
                            for i in range(len(experiment.t)):
                                t_idx = np.argmin(np.abs(ts + experiment.t[0] - experiment.t[i]))
                                for j in range(len(experiment.A[:,i])):
                                    square = square + (experiment.A[j,i] - A_vals[t_idx])**2
                                    square = square + (experiment.I[j,i] - I_vals[t_idx])**2
                                    num_vals = num_vals + 2
                            mean = square / num_vals
                            rms = np.sqrt(mean)
                            axs[0][col_idx].set_title(strain_str)
                        axs[iterator][col_idx].plot(experiment.t, np.mean(experiment.I, axis = 0), '*',  color = color, label = 'NO2', alpha = alpha)
                        axs[iterator][col_idx].vlines(experiment.t, np.mean(experiment.I, axis = 0) - np.std(experiment.I, axis = 0),np.mean(experiment.I, axis = 0) + np.std(experiment.I, axis = 0), color = color, alpha = alpha)
                        axs[iterator][col_idx].legend()
                        axs[iterator][col_idx].set_ylim([-0.1, 2])
                        axs[iterator][col_idx].set_xlim([-1, 75])


    col_idx = col_idx+1
fig.add_subplot(111, frameon=False)
fig.tight_layout()
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(False)



plt.xlabel("t [h]")
plt.ylabel(r"Metabolite conc [mM]")
plt.ylim([0,2.2])
plt.legend()
fig.tight_layout()
plt.savefig('SI_pH7_fit.png')
plt.savefig('SI_pH7_fit.svg')
plt.show()
plt.cla()




#####################################################                                                                                                                                      
#simulate 4 cycles of serial enrichment for Nar+Nap                                                                                                                                        
####################################################                                                                                                                                       
f0_Naps = np.linspace(0.05,0.95, 11)
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 12


colors = ['tab:blue', 'tab:orange']


fig, axs = plt.subplots(1,1,figsize=(7,5), sharex = False, sharey = False)
for j in range(len(f0_Naps)):
    t_growth = np.linspace(0,72, num = 100)
    n_cycles = 5
    dilution_factor = 8;
    nk = np.zeros((n_cycles,2))
    for i in range(2):
        if colors[i] == 'tab:blue': #Nar strain                                                                                                                                            
            nk[0,i] = 0.01 - f0_Naps[j]*0.01
        else: #Nap strain                                                                                                                                                                  
            nk[0,i] = 0.01*f0_Naps[j]
    ndead = np.zeros((n_cycles,2))
    ndead[0,:] = 0
    A0 = 2.0
    I0 = 0.0
    A_vals = np.zeros((n_cycles-1, len(t_growth)))
    I_vals = np.zeros((n_cycles-1, len(t_growth)))

    A = np.zeros(n_cycles)
    I = np.zeros(n_cycles)
    t_vals = []
    for k in range(1,n_cycles):
        t_vals.append(t_growth + (k-1)*t_growth[-1])
        y0 = []
        for i in range(2):
            y0.append(nk[k-1,i])
            y0.append(ndead[k-1,i])
        y0.append(A0*(1.0 - 1.0/dilution_factor)+A[k-1]/dilution_factor)
        y0.append(I0*(1.0 - 1.0/dilution_factor)+I[k-1]/dilution_factor)
        yh = ntm.denitODE(y0,t_growth,[pfit_NarG, pfit_NapA],n=2)
        A_vals[k-1,:] = yh[:,-2]
        I_vals[k-1,:] = yh[:,-1]
        for i in range(2):
            nk[k,i] = yh[-1,i*2]/dilution_factor
            ndead[k,i] = yh[-1,i*2+1]/dilution_factor
            A[k] = yh[-1,-2]
            I[k] = yh[-1,-1]
    t_vals = np.asarray(t_vals)

    
    linewidth = 2.0
    Nar_co_OD = None
    Nap_co_OD = None
    for i in range(2):
        if colors[i] == 'tab:blue':
            Nar_co_OD = nk[-1,i]
            Nar_co_ODs = nk[:,i]
        else:
            Nap_co_OD = nk[-1,i]
            Nap_co_ODs = nk[:,i]
    axs.plot(range(0, n_cycles),Nap_co_ODs/(Nap_co_ODs+Nar_co_ODs), linewidth = linewidth, color = 'tab:orange')
axs.set_ylim([0,1])
plt.ylabel('PD Nar+ rel. ab.')
plt.xlabel('cycle')
fig.tight_layout()
plt.savefig('SI_pH7_predict.png')
plt.savefig('SI_pH7_predict.svg')
plt.show()

