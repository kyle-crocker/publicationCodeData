############################################
## define some functions for cfu analysis ##
############################################

import numpy as np

def get_avg_ste_shot(counts):
    N = len(counts)
    sum = 0
    for i in range(len(counts)):
        if np.isnan(counts[i]):
            sum = sum+0
        else:
            sum = sum + counts[i]
        
    avg = sum/N
    std = np.sqrt(sum)/N
    se = std/np.sqrt(N)
    return avg, se


def get_avg_std(counts):
    N = len(counts)
    #print(counts)
    sum_N = 0
    for i in range(len(counts)):
        if np.isnan(counts[i]):
            N = N - 1
        else:
            sum_N = sum_N + counts[i]
    #print(N)
    if N <= 0:
        return np.nan, np.nan
    elif N == 1:
        print('hit only one count')
        return sum_N/N, np.sqrt(sum_N) #use shot noise error
    else:
        avg = sum_N/N
        sq_residuals = 0
        for i in range(len(counts)):
            if np.isnan(counts[i]):
                pass
            else:
                sq_residuals = sq_residuals + (counts[i]-avg)**2 

        std = np.sqrt(sq_residuals/(N-1))
        #se = std/np.sqrt(N)
        return avg, std

def combine_avgs_unweighted(avgs, stds):
    for avg in avgs:
        if avg ==0:
            print('WARNING: AVERAGING WITH 0!')
    avg_avgs = np.mean(avgs)
    sq_stds = 0
    for i in range(len(stds)):
        sq_stds = sq_stds + stds[i]**2
    return avg_avgs, np.sqrt(sq_stds)/len(stds)

def combine_avgs_weighted(avgs, stds):
    ws = []
    for std in stds:
        ws.append(1.0/(std**2))
    avg_avgs = 0
    
    for i in range(len(avgs)):
        if avgs[i] ==0:
            print('WARNING: AVERAGING WITH 0!')
        elif np.isnan(avgs[i]):
            pass
        else:
            avg_avgs = avg_avgs + ws[i]*avgs[i]
    avg_avgs = avg_avgs/np.nansum(ws)
    error = np.sqrt(1/np.nansum(ws))
    return avg_avgs, error

def CFUs_per_OD(CFUs, CFUs_std, OD, OD_std):
    val = CFUs/OD
    std = np.sqrt((CFUs_std/OD)**2 + (OD_std*CFUs/OD**2)**2)
    return val, std


