import sys
sys.path.append('/home/kyle/microbial_ecology/custom_functions/')
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
import warnings
warnings.filterwarnings("ignore")


#define some parameters for plots
tick_label_size = 9
font_size = 11
alpha = 0.8

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['xtick.labelsize'] = 40
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = 42
plt.rcParams['font.family'] = 'Arial'

################################################
### define some useful ancillary functions #####
################################################
def neg_fitness(x):
    return -1.0*(x[0] + x[1])

def pareto_front(x,k,b,c,d):
    return ((d - b*x**k )/c)**(1.0/k)

def fitness(x):
    return x[0] + x[1]

def phys_constraint(x,y,k,b,c,d):
    return (b*x**k + c*y**k) - d;

def get_Itox_array(rAs, rA_thresh, Itox_min, Itox_max):
    Itox_vals = []
    for rA in rAs:
        if rA > rA_thresh:
            Itox_vals.append(Itox_min)
        else:
            Itox_vals.append(Itox_max)
    return Itox_vals

def get_Itox(rA, rA_thresh, Itox_min, Itox_max):
    if rA > rA_thresh:
        Itox_vals = Itox_min
    else:
        Itox_vals = Itox_max
    return Itox_vals

def cons_f(x):
    return phys_constraint(x[0],x[1],a,b,c,d)

#classify outcome of enrichment and calculate phenotypes of most abundant strains
def get_dr(rAs, rel_abs, ab_thresh):
    drs = []
    rAs_avg = []
    max_rel_ab = 0
    second_max_rel_ab = 0
    max_idx = None
    second_max_idx = None
    for i in range(len(rel_abs)):
        if rel_abs[i] > max_rel_ab:
            max_idx = i
            max_rel_ab = rel_abs[i]
        elif rel_abs[i] > second_max_rel_ab:
            second_max_idx = i
            second_max_rel_ab = rel_abs[i]
    if second_max_idx == None:
        second_max_idx = max_idx - 1

    # calculate some quantities to classify endpoint relative abundance distribution
    for i in range(len(rAs) - 1):    
        rel_abs_1 = rel_abs[i+1]
        rel_abs_0 = rel_abs[i]
        if np.abs(rel_abs_1) < ab_thresh:
            rel_abs_1 = 0
        if np.abs(rel_abs_0) < ab_thresh:
            rel_abs_0 = 0
        rAs_avg.append((rAs[i+1] + rAs[i])/2)  
        drs.append((rel_abs_1 - rel_abs_0))    
    
    # algorithm to classify outcome of enrichment
    # -3 => two generalist regime
    # -1 => one generalist regime
    # 1 => two specialist regime
    # 2 => NO3 specialist, one generalist
    # 3 => two specialists, one generalist 
    num_roots = 0
    drs_finite = []
    peak_indices = []
    orig_indices = []
    for i in range(len(drs)):
        if drs[i] != 0:
            drs_finite.append(drs[i])
            orig_indices.append(i)
    for i in range(len(drs_finite) - 1):
        if drs_finite[i+1]*drs_finite[i] < 0:
            num_roots = num_roots + 1
            peak_indices.append(orig_indices[i])
    if drs_finite[-1] <	0:
        num_roots = -1*num_roots

    #if there's only one peak, choose last rA generalist as second strain
    if len(peak_indices) ==1:
        max_idx = peak_indices[0]+1
        second_max_idx = len(rAs) - 2
    elif len(peak_indices) ==3:
        max_idx = peak_indices[1]
        second_max_idx = peak_indices[2]+1
        
    #return rAs_avg, drs, num_roots, max_idx, second_max_idx
    return num_roots, max_idx, second_max_idx


#######################################################################
### simulate growth of community subject to physiological tradeoffs ###
#######################################################################
def get_phen_dist(k, rA_thresh, f, I_tox_min, I_tox_max, n_strains, ab_thresh, rd_max, n_cycles_real=12, plot=True):
    r_lim = np.asarray([0,1.1])
    r_thresh = 0
    a = k #curvature parameter
    b = 1
    c = 1
    d = 1
    epsilon = 0.01 #how far points can be from the constraint curve
    kA = 0.01 #nitrate affinity parameter
    kI = 0.01 #nitrite affinity parameter
    N0 = 0.01 #initial cell density (in units of OD)
    N0_dead = 0 #initial number of dead cells
    tlag = 0 #lag time
    offset = 0 #observed OD offset

    #parameters to control nitrate toxicity. when set to 100, there is effectively no toxic effect of nitrate
    alphaA = 100.0 
    ItoxA = 100.0

    #parameters control nitrite toxicity. 
    alphaI = 100.0
    ItoxI = 100.0

    #parameter to control sharpness of transition between no death rate and max death rate
    alphad = 10000.0

    #nitrate reduction rate for each strain
    rAs_orig = np.linspace(0,1, n_strains)
    
    #generate phenotypes on the constraint curve
    p = []
    rs = np.zeros((n_strains,2))
    for i in range(0,n_strains):
        rA = rAs_orig[i]
        rI = pareto_front(rA,a,b,c,d) 
        gamA = 0.02
        gamI = 0.02
        Itoxd = get_Itox(rA*f, rA_thresh, I_tox_min, I_tox_max) 
        test_params = np.zeros(17)

        #rescale reduction rates
        test_params[0] = rA*f
        test_params[1] = rI*f

        test_params[2] = kA
        test_params[3] = kI
        test_params[4] = gamA
        test_params[5] = gamI
        test_params[6] = N0
        test_params[7] = tlag
        test_params[8] = offset
        test_params[9] = alphaA
        test_params[10] = ItoxA
        test_params[11] = alphaI
        test_params[12] = ItoxI
        test_params[13] = rd_max
        test_params[14] = alphad
        test_params[15] = Itoxd
        test_params[16] = N0_dead
        p.append(test_params)
        rs[i,:] = [rA*f,rI*f] #save nitrate and nitrite reduction rates
    parray = np.asarray(p)
    t_growth = np.linspace(0,72, num = 100)
    
    #simulate n_cycles_real cycles of serial enrichment
    n_cycles = n_cycles_real + 1
    dilution_factor = 8;
    nk = np.zeros((n_cycles,n_strains))
    nk[0,:] = 0.01
    ndead = np.zeros((n_cycles,n_strains))
    ndead[0,:] = 0
    A0 = 2.0
    I0 = 0.0
    A_vals = np.zeros((n_cycles-1, len(t_growth)))
    I_vals = np.zeros((n_cycles-1, len(t_growth)))
    A = np.zeros(n_cycles)
    I = np.zeros(n_cycles)
    t_vals = []
    
    #loop over cycles
    for k in range(1,n_cycles):
        t_vals.append(t_growth + (k-1)*t_growth[-1])
        y0 = []
        
        #set up initial conditions
        for i in range(n_strains):
            y0.append(nk[k-1,i])
            y0.append(ndead[k-1,i])
        y0.append(A0*(1.0 - 1.0/dilution_factor)+A[k-1]/dilution_factor)
        y0.append(I0*(1.0 - 1.0/dilution_factor)+I[k-1]/dilution_factor)

        #integrate CRM ODEs
        yh = ntm.denitODE(y0,t_growth,p,n_strains)
        A_vals[k-1,:] = yh[:,-2]
        I_vals[k-1,:] = yh[:,-1]
        for i in range(n_strains):
            nk[k,i] = yh[-1,i*2]/dilution_factor
            ndead[k,i] = yh[-1,i*2+1]/dilution_factor
            A[k] = yh[-1,-2]
            I[k] = yh[-1,-1]
    
    t_vals = np.asarray(t_vals)
    nk[-1,:] = nk[-1,:]/np.sum(nk[-1,:])
    #rAs_avg, drs, num_roots, max_idx, second_max_idx = get_dr(parray[:,0], nk[-1,:], ab_thresh) #classify outcome of enrichment, get most abundant strains for coculture experiment
    num_roots, max_idx, second_max_idx = get_dr(parray[:,0], nk[-1,:], ab_thresh) #classify outcome of enrichment, get most abundant strains for coculture experiment
    max_indices = [max_idx, second_max_idx]
    idx = np.argsort(nk[-1,:])
    color = 'maroon' #so scarlet it was maroon

    #if you want to plot
    if plot:
        #plot relative abundance distribution after n_cycles_real cycles
        plt.cla()    
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['xtick.labelsize'] = tick_label_size
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = font_size
        plt.rcParams['font.family'] = 'Arial'
        fig, axs = plt.subplots(1,1,figsize=(3.5,3), sharex = False, sharey = False)
        axs.scatter(parray[:,0],parray[:,1],c=(nk[-1,:]),cmap='Reds', edgecolors = 'black', label = 'phenotype')
        axs.legend(loc= 'upper right')
        axs.set_xlabel('rA (mM/OD/hr)')
        axs.set_ylabel('rI (mM/OD/hr)')
        ax2 = axs.twinx()
        ax2.bar(parray[:,0], nk[-1,:], width = 3.0/n_strains, color = color, alpha = .75)
        ax2.set_ylim([0,0.6])
        ax2.set_ylabel('rel. ab.', color = color)
        ax2.tick_params(axis='y', labelcolor=color)
        fig.tight_layout()
        plt.savefig('rel_ab_plots/k='+str(a)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_I_tox_min='+str(I_tox_min)+'_I_tox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_ab_thresh='+str(ab_thresh)+'_rd_max='+str(rd_max)+'_n_cycles_real='+str(n_cycles_real)+'.svg')  #k, rA_thresh, f, I_tox_min, I_tox_max, n_strains, ab_thresh, rd_max, n_cycles_real = 12
        plt.savefig('rel_ab_plots/k='+str(a)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_I_tox_min='+str(I_tox_min)+'_I_tox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_ab_thresh='+str(ab_thresh)+'_rd_max='+str(rd_max)+'_n_cycles_real='+str(n_cycles_real)+'.png')  #k, rA_thresh, f, I_tox_min, I_tox_max, n_strains, ab_thresh, rd_max, n_cycles_real = 12
        plt.cla()        
        
        if rAs_orig[max_idx] > rAs_orig[second_max_idx]:
            colors = ['tab:blue', 'tab:orange']
        else:
            colors = ['tab:orange', 'tab:blue']
        p_maxs = []

        #get phenotypes of enriched strains
        for i in max_indices:
            rA = rAs_orig[i]
            rI = pareto_front(rA,a,b,c,d)
            gamA = 0.02
            gamI = 0.02
            Itoxd = get_Itox(rA*f, rA_thresh, I_tox_min, I_tox_max)
            test_params = np.zeros(17)
            test_params[0] = rA*f
            test_params[1] = rI*f
            test_params[2] = kA
            test_params[3] = kI
            test_params[4] = gamA
            test_params[5] = gamI
            test_params[6] = N0
            test_params[7] = tlag
            test_params[8] = offset
            test_params[9] = alphaA
            test_params[10] = ItoxA
            test_params[11] = alphaI
            test_params[12] = ItoxI
            test_params[13] = rd_max
            test_params[14] = alphad
            test_params[15] = Itoxd
            test_params[16] = N0_dead
            p_maxs.append(test_params)

        ###########################################################
        ##### compare ODs in monoculture to ODs in coculture ######
        ###########################################################

        # simulate n_cycles - 1 cycles of serial enrichment for Nar+Nap
        t_growth = np.linspace(0,72, num = 100)
        n_cycles = 5
        dilution_factor = 8
        nk = np.zeros((n_cycles,2))
        nk[0,:] = 0.005
        ndead = np.zeros((n_cycles,2))
        ndead[0,:] = 0
        A0 = 2.0
        I0 = 0.0
        A_vals = np.zeros((n_cycles-1, len(t_growth)))
        I_vals = np.zeros((n_cycles-1, len(t_growth)))

        A = np.zeros(n_cycles)
        I = np.zeros(n_cycles)
        t_vals = []
        
        #loop over cycles 
        for k in range(1,n_cycles):
            #update times
            t_vals.append(t_growth + (k-1)*t_growth[-1])
            y0 = []
            #set up initial conditions
            for i in range(2):
                y0.append(nk[k-1,i])
                y0.append(ndead[k-1,i])

            #simulate dilution
            y0.append(A0*(1.0 - 1.0/dilution_factor)+A[k-1]/dilution_factor)
            y0.append(I0*(1.0 - 1.0/dilution_factor)+I[k-1]/dilution_factor)

            #integrate ODEs 
            yh = ntm.denitODE(y0,t_growth,p_maxs,n=2)
            A_vals[k-1,:] = yh[:,-2]
            I_vals[k-1,:] = yh[:,-1]

            for i in range(2):
                #save info across cycles
                nk[k,i] = yh[-1,i*2]/dilution_factor
                ndead[k,i] = yh[-1,i*2+1]/dilution_factor
                A[k] = yh[-1,-2]
                I[k] = yh[-1,-1]
        t_vals = np.asarray(t_vals)

        #set plot parameters 
        plt.cla()
        linewidth = 2.0
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['xtick.labelsize'] = tick_label_size
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = font_size
        plt.rcParams['font.family'] = 'Arial'
        
        fig, axs = plt.subplots(1,1,figsize=(3.25,3), sharex = False, sharey = False)
        Nar_co_OD = None
        Nap_co_OD = None
        for i in range(len(max_indices)):
            if colors[i] == 'tab:blue':
                
                #plot Nar+ coculture OD 
                axs.plot(range(0, n_cycles),nk[:, i], linewidth = linewidth, color = 'tab:purple', label = 'PD Nar+ co.', alpha = alpha)
                Nar_co_OD = nk[-1,i]
                Nar_co_ODs = nk[:,i]
            else:
                Nap_co_OD = nk[-1,i]
                Nap_co_ODs = nk[:,i]
        axs.set_ylabel('Endpt. biomass [OD600]')
        axs.set_xlabel('cycle')

        # now do monocultures 
        rAs_mono = []
        for i in range(2):
            rAs_mono.append(p_maxs[i][0])
            t_growth = np.linspace(0,72, num = 100)
            
            #simulate n_cycles - 1 cycles of serial enrichment
            n_cycles = 5
            dilution_factor = 8;
            nk = np.zeros((n_cycles,2))
            nk[0,:] = 0.01
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
                y0.append(nk[k-1,i])
                y0.append(ndead[k-1,i])
                y0.append(A0*(1.0 - 1.0/dilution_factor)+A[k-1]/dilution_factor)
                y0.append(I0*(1.0 - 1.0/dilution_factor)+I[k-1]/dilution_factor)
                yh = ntm.denitODE(y0,t_growth,p_maxs[i],n=1)
                A_vals[k-1,:] = yh[:,-2]
                I_vals[k-1,:] = yh[:,-1]
                nk[k,i] = yh[-1,0]/dilution_factor
                ndead[k,i] = yh[-1,1]/dilution_factor
                A[k] = yh[-1,-2]
                I[k] = yh[-1,-1]
            t_vals = np.asarray(t_vals)
            if colors[i] == 'tab:blue':
                Nar_mono_OD = nk[-1,i]
                label_val = 'PD Nar+ mono.'
            else:
                Nap_mono_OD = nk[-1,i]
                label_val = 'RH Nap+ mono.'

            #plot monoculture ODs
            axs.plot(range(0, n_cycles),nk[:, i], linewidth = linewidth, color = colors[i], label = label_val, alpha = alpha)
        axs.set_ylim([0, 0.0105])
        axs.legend()
        fig.tight_layout()
        plt.savefig('rel_ab_plots/monoculture_ODs_rAs='+str(rAs_mono)+'_k='+str(a)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_I_tox_min='+str(I_tox_min)+'_I_tox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_ab_thresh='+str(ab_thresh)+'_rd_max='+str(rd_max)+'_n_cycles_real='+str(n_cycles_real)+'.png')  
        plt.savefig('rel_ab_plots/monoculture_ODs_rAs='+str(rAs_mono)+'_k='+str(a)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_I_tox_min='+str(I_tox_min)+'_I_tox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_ab_thresh='+str(ab_thresh)+'_rd_max='+str(rd_max)+'_n_cycles_real='+str(n_cycles_real)+'.svg')  
        plt.cla()

        #############################################
        ##### plot coculture relative abundance #####
        #############################################
        f0_Naps = np.linspace(0.05,0.95, 11)
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['xtick.labelsize'] = tick_label_size
        plt.rcParams['text.usetex'] = False
        plt.rcParams['font.size'] = font_size
        plt.rcParams['font.family'] = 'Arial'

        fig, axs = plt.subplots(1,1,figsize=(3.25,3), sharex = False, sharey = False)
        # simulate 4 cycles of serial enrichment for Nar+Nap #####
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
                yh = ntm.denitODE(y0,t_growth,p_maxs,n=2)
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
            for i in range(len(max_indices)):
                if colors[i] == 'tab:blue':
                    Nar_co_OD = nk[-1,i]
                    Nar_co_ODs = nk[:,i]
                else:
                    Nap_co_OD = nk[-1,i]
                    Nap_co_ODs = nk[:,i]
            axs.set_xlabel('OD')
            axs.set_ylabel('cycle')
            
            # plot relative abundances
            if num_roots == -1:   # one generalist regime 
                axs.plot(range(0, n_cycles),Nar_co_ODs/(Nap_co_ODs+Nar_co_ODs), linewidth = linewidth, color = 'tab:orange', alpha = alpha)
            elif num_roots == -3: # two generalist regime
                axs.plot(range(0, n_cycles),Nar_co_ODs/(Nap_co_ODs+Nar_co_ODs), linewidth = linewidth, color = 'tab:purple', alpha = alpha)
            else:
                axs.plot(range(0, n_cycles),Nar_co_ODs/(Nap_co_ODs+Nar_co_ODs), linewidth = linewidth, color = 'tab:purple', alpha = alpha)
        axs.set_ylim([0,1])
        plt.ylabel('PD Nar+ rel. ab.')
        plt.xlabel('cycle')
        fig.tight_layout()
        plt.savefig('rel_ab_plots/coculture_rel_abs_rAs='+str(rAs_mono)+'_k='+str(a)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_I_tox_min='+str(I_tox_min)+'_I_tox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_ab_thresh='+str(ab_thresh)+'_rd_max='+str(rd_max)+'_n_cycles_real='+str(n_cycles_real)+'.png')  
        plt.savefig('rel_ab_plots/coculture_rel_abs_rAs='+str(rAs_mono)+'_k='+str(a)+'_rA_thresh='+str(rA_thresh)+'_f='+str(f)+'_I_tox_min='+str(I_tox_min)+'_I_tox_max='+str(I_tox_max)+'_n_strains='+str(n_strains)+'_ab_thresh='+str(ab_thresh)+'_rd_max='+str(rd_max)+'_n_cycles_real='+str(n_cycles_real)+'.svg')  
        plt.cla()
    
    p_dist = 0
    num_diffs = 0
    for i in range(len(nk[-1,:])):
        for j in range(len(nk[-1,:])):
            if i == j:
                pass
            else:
                p_dist = p_dist + 0.5*nk[-1,i] * nk[-1,j] * np.sqrt((parray[i,0] - parray[j,0])**2 + (parray[i,1] - parray[j,1])**2)
                num_diffs = num_diffs + 1
    return num_roots # return num_roots, which encodes classification
    # num_roots = -3 => two generalist regime
    # num_roots = -1 => one generalist regime
    # num_roots = 1 => two specialist regime
    # num_roots = 2 => NO3 specialist, one generalist
    # num_roots = 3 => two specialists, one generalist 
