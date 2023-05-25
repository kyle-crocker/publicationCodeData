import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lmfit import Minimizer, conf_interval, Parameters, report_fit
import statsmodels.api as sm
from sklearn.metrics import r2_score
import copy
import random as rd
import nitrite_toxicity_model as ntm 


class experiment:
    #This class contains the experimental data for a given condition
    def __init__(self, ID, phen, N0, Nend, A0, A, I0, I, t, N = None, N0_Nend_avg=None, blank_avg = None, Aend = None, Iend = None): #added N for continuous OD measurement
        self.ID = ID
        self.phen = phen
        self.N0 = N0
        self.Nend = Nend
        self.A0 = A0
        self.A = A
        self.I0 = I0
        self.I = I
        self.t = t
        self.N = N
        self.N0_Nend_avg = N0_Nend_avg
        self.blank_avg = blank_avg
        self.Aend = Aend
        self.Iend = Iend

class NNpY_experiment:
    #This class contains the experimental data for a given condition
    def __init__(self, ID, phen, N0, Nend, A0, A, I0, I, t, pH, N = None, N0_Nend_avg=None, blank_avg = None, Aend = None, Iend = None, genotype = None, overall_yield = None, A_consumed = None, I_consumed = None, met_consumed = None): #added N for continuous OD measurement
        self.ID = ID
        self.phen = phen
        self.N0 = N0
        self.Nend = Nend
        self.A0 = A0
        self.A = A
        self.I0 = I0
        self.I = I
        self.t = t
        self.N = N
        self.N0_Nend_avg = N0_Nend_avg
        self.blank_avg = blank_avg
        self.Aend = Aend
        self.Iend = Iend
        self.genotype = genotype
        self.overall_yield = overall_yield
        self.pH = pH
        self.A_consumed = A_consumed
        self.I_consumed = I_consumed
        self.met_consumed = met_consumed

class pseudo_rhizo_comp_experiment:
    #This class contains the experimental data for a given condition
    def __init__(self, pH, f0_pseudo, fend_pseudo, f0_rhizo, fend_rhizo, N0, Nend, A0, A, I0, I, t, N = None, N0_Nend_avg=None, blank_avg = None, Aend = None, Iend = None, A_rate = None, I_rate = None, A_mu = None, I_mu = None, A_C = None, I_C = None, A_t_switch = None, I_t_switch = None, T_c_A = None, T_c_I = None, met_rate_A = None, met_rate_I = None): #added N for continuous OD measurement
        self.pH = pH
        self.f0_pseudo = f0_pseudo
        self.f0_rhizo = f0_rhizo
        self.fend_pseudo = fend_pseudo
        self.fend_rhizo = fend_rhizo
        self.N0 = N0
        self.Nend = Nend
        self.A0 = A0
        self.A = A
        self.I0 = I0
        self.I = I
        self.t = t
        self.N = N
        self.N0_Nend_avg = N0_Nend_avg
        self.blank_avg = blank_avg
        self.Aend = Aend
        self.Iend = Iend
        self.A_rate = A_rate
        self.I_rate = I_rate
        self.A_mu = A_mu
        self.I_mu = I_mu
        self.A_C = A_C
        self.I_C = I_C
        self.A_t_switch = A_t_switch
        self.I_t_switch = I_t_switch
        self.T_c_A = T_c_A
        self.T_c_I = T_c_I
        self.met_rate_A = met_rate_A
        self.met_rate_I = met_rate_I

        
class enrichment_experiment:
    #This class contains the experimental data for a given condition
    def __init__(self, ID, phen, N0, Nend, A0, A, I0, I, t, succ_conc, pH, endpoint_pH = None, A_mu = None, I_mu = None, A_C = None, I_C = None, A_t_switch = None, I_t_switch = None, T_c_A = None, T_c_I = None, met_rate_A = None, met_rate_I = None, N = None, Nstd = None, N_ts = None): #added N for continuous OD measurement
        self.ID = ID
        self.phen = phen
        self.N0 = N0
        self.Nend = Nend
        self.N = N
        self.Nstd = Nstd
        self.A0 = A0
        self.A = A
        self.I0 = I0
        self.I = I
        self.t = t
        self.pH = pH
        self.succ_conc = succ_conc
        self.endpoint_pH = endpoint_pH
        self.A_mu = A_mu
        self.I_mu = I_mu
        self.A_C = A_C
        self.I_C = I_C
        self.A_t_switch = A_t_switch
        self.I_t_switch = I_t_switch
        self.T_c_A = T_c_A
        self.T_c_I = T_c_I
        self.met_rate_A = met_rate_A
        self.met_rate_I = met_rate_I


def fitYields_no_offset(experiments, gamA=None, gamI = None):               
    if experiments[0].phen == 'Nar/Nir':
        DelA = np.array([0])
        DelI = np.array([0])
        DelOD = np.array([0])
        for i in range(0,len(experiments)):
            DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
            current_DelA = experiments[i].A0-experiments[i].A[:,-1]
            DelI = np.append(DelI,experiments[i].A0-experiments[i].A[:,-1] + experiments[i].I0-experiments[i].I[:,-1])
            current_DelI = experiments[i].A0-experiments[i].A[:,-1] + experiments[i].I0-experiments[i].I[:,-1]

            if gamI != None:
                DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0 - gamI*current_DelI.reshape(-1,1))
            elif gamA != None:
                DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0 - gamA*current_DelA.reshape(-1,1))
            else:
                DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        if gamI != None:
            x = DelA.reshape(-1, 1)
        elif gamA != None:
            x = DelI.reshape(-1, 1)
        else:
            x = np.append(DelA.reshape(-1, 1), DelI.reshape(-1, 1),axis=1)
    if experiments[0].phen == 'Nar':
        DelA = np.array([0])
        DelOD = np.array([0])
        for i in range(0,len(experiments)):
            DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelA.reshape(-1, 1), np.ones((len(DelOD),1)),axis=1)
    if experiments[0].phen == 'Nir':
        DelI = np.array([0])
        DelOD = np.array([0])
        for i in range(0,len(experiments)):
            DelI = np.append(DelI,experiments[i].I0-experiments[i].I[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelI.reshape(-1, 1), np.ones((len(DelOD),1)),axis=1)

    mod = sm.OLS(DelOD, x)
    res = mod.fit()
    ci = res.conf_int(0.32)   # 68% confidence interval. comparable to 1 standard error
    DelOD_pred = res.predict()
    r2 = r2_score(DelOD,DelOD_pred)
    
    if experiments[0].phen == 'Nar/Nir':
        if gamI != None:
            gamA = res.params[0]
        elif gamA != None:
            gamI = res.params[0]
        else:
            gamA = res.params[0]
            gamI = res.params[1]
    elif experiments[0].phen == 'Nar':
        gamA = res.params[0]
        gamI = 0
    elif experiments[0].phen == 'Nir':
        gamA = 0
        gamI = res.params[0]

    return gamA, gamI, ci, r2
    
    
    
def fit_met(experiments_to_fit, rA = None, rI = None, gamA = None, gamI = None, gamA_std = None, gamI_std = None, phen = 'Nar/Nir', n=1):
    
    
    Itox_rI = 10.0
    alpha_rI = 100.0
    alphaA = 100.0
    ItoxA = 10.0                                                                                                          
    r_d = 0
    Itox_rd = 10.0
    alpha_rd = 100.0
    OD0 = 0.01
    params = Parameters()
    if rA != None:
        params.add('rA', value=rA, min=0, max=20,vary=False, brute_step=2)
    else:
        rA = 7.0
        params.add('rA', value=rA, min=0, max=20,vary=True, brute_step=2)

    if rI != None:
        params.add('rI', value=rI, min=0, max=20,vary=False, brute_step=2)
    else:
        rI = 7.0
        params.add('rI', value=rI, min=0, max=20,vary=True, brute_step=2)

    #params.add('rA', value=3.6, min=0, max=10,vary=False, brute_step=2)                                                                                                                                                                                                                                                                
    params.add('kA', value=1e-2, min=1e-3, max=1e1,vary=False)                 
    params.add('kI', value=1e-2, min=1e-3, max=2e-2,vary=False)
    params.add('OD0', value = OD0, min = 0, max = 0.03,vary=False,brute_step = 0.02)
    params.add('t_lag', value = 0, min=0, max = 10, vary =False) #True, brute_step=1)                                                                                                                     
    params.add('offset', value = 0, vary=False)
    params.add('Itox_rI', value = Itox_rI, min = 0, max = 2.0, vary = False, brute_step = 1.0)
    params.add('Itox_rd', value = Itox_rd, min = 0, max = 4.0, vary = False, brute_step = 2.0)
    #params.add('Itox_rA', value = 2.0, min = 0, max = 2.0, vary = False, brute_step = 0.5)                                                                                                               
    params.add('alphaA', value = alphaA, min = 1.0, max = 20.0, vary = False, brute_step = 10.0)
    params.add('ItoxA', value = ItoxA, min = 0, max = 3.0, vary = False, brute_step = 1.0)
    params.add('alpha_rd', value = alpha_rd, min = 1.0, max = 100.0, vary = False, brute_step = 50.0)
    params.add('alpha_rI', value = alpha_rI, min = 1.0, max = 20.0, vary = False, brute_step = 10.0)
    params.add('r_d', value = r_d, min = 0.0, max = 0.5, vary = False, brute_step = 0.25)
    
    if (gamA == None) or (gamI == None):
        gamA,gamI,gam_se,r2 = fitYields_no_offset(experiments_to_fit, gamI = gamI, gamA = gamA)
        #gamA,gamI,gam_se,r2, offset = dn.fitYields(experiments_to_fit, gamI = gamI, gamA = gamA)
        try:
            if phen == 'Nar/Nir':
                gamA_std = gamA - gam_se[0][0]
                gamI_std = gamI - gam_se[1][0]
            elif phen == 'Nar':
                gamA_std = gamA - gam_se[0][0]
                gamI_std = 0
            elif phen == 'Nir':
                gamA_std = gamA - gam_se[0][0]
                gamI_std = 0
        except:
            pass


    offset = 0                                                                                                                                                                                           
    params.add('gamA', value=gamA, min=0, max=0.05,vary=False, brute_step = 0.01)
    params.add('gamI', value=gamI, min=0, max=0.05,vary=False, brute_step = 0.01) 
    
    fitter = Minimizer(residualGlobLMFit, params, fcn_args=(experiments_to_fit,))
    result_brute = fitter.minimize(method='brute')#, keep = 20)                                                                                                                                   
    best_result = copy.deepcopy(result_brute)
    for candidate in result_brute.candidates:                                     
        trial = fitter.minimize(method='leastsq', params=candidate.params)
        if trial.chisqr < best_result.chisqr:
                best_result = trial

    pfit = convertPTableToMat(best_result.params)
    chisqr_val = best_result.chisqr
    
    return pfit, chisqr_val, best_result.params, gamA_std, gamI_std



def convertPTableToMat(params,n=1):
    if n == 1:
        p_out = np.zeros(17)
        p_out[0] = params['rA'].value
        p_out[1] = params['rI'].value
        p_out[2] = params['kA'].value
        p_out[3] = params['kI'].value
        p_out[4] = params['gamA'].value
        p_out[5] = params['gamI'].value
        p_out[6] = params['OD0'].value
        p_out[7] = params['t_lag'].value
        p_out[8] = params['offset'].value
        p_out[9] = params['alphaA'].value
        p_out[10] = params['ItoxA'].value
        p_out[11] = params['alpha_rI'].value
        p_out[12] = params['Itox_rI'].value
        p_out[13] = params['r_d'].value
        p_out[14] = params['alpha_rd'].value
        p_out[15] = params['Itox_rd'].value
        p_out[16] = 0
    else:
        p_out = np.zeros((n,17))
        for i in range(0,n):
            p_out[i,0] = params['rA'+str(i)].value
            p_out[i,1] = params['rI'+str(i)].value
            p_out[i,2] = params['kA'+str(i)].value
            p_out[i,3] = params['kI'+str(i)].value
            p_out[i,4] = params['gamA'+str(i)].value
            p_out[i,5] = params['gamI'+str(i)].value
            p_out[i,6] = params['OD0'+str(i)].value
            p_out[i,7] = params['t_lag'+str(i)].value
            p_out[i,8] = params['offset'+str(i)].value
            p_out[i,9] = params['alphaA'+str(i)].value
            p_out[i,10] = params['ItoxA'+str(i)].value
            p_out[i,11] = params['alpha_rI'+str(i)].value
            p_out[i,12] = params['Itox_rI'+str(i)].value
            p_out[i,13] = params['r_d'+str(i)].value
            p_out[i,14] = params['alpha_rd'+str(i)].value
            p_out[i,15] = params['Itox_rd'+str(i)].value
            p_out[i,16] = 0
    '''
    else:
        p_out = np.zeros((n,17))
        for i in range(0,n):
            p_out[i,0] = params[i]['rA'].value
            p_out[i,1] = params[i]['rI'].value
            p_out[i,2] = params[i]['kA'].value
            p_out[i,3] = params[i]['kI'].value
            p_out[i,4] = params[i]['gamA'].value
            p_out[i,5] = params[i]['gamI'].value
            p_out[i,6] = params[i]['OD0'].value
            p_out[i,7] = params[i]['t_lag'].value
            p_out[i,8] = params[i]['offset'].value
            p_out[i,9] = params[i]['alphaA'].value
            p_out[i,10] = params[i]['ItoxA'].value
            p_out[i,11] = params[i]['alpha_rI'].value
            p_out[i,12] = params[i]['Itox_rI'].value
            p_out[i,13] = params[i]['r_d'].value
            p_out[i,14] = params[i]['alpha_rd'].value
            p_out[i,15] = params[i]['Itox_rd'].value
            p_out[i,16] = 0
    '''


    return p_out

def residualGlobLMFit_cfu_dyn(params,experiments,OD_to_CFU,n=1):
    return residualGlob_cfu_dyn(convertPTableToMat(params,n),experiments,OD_to_CFU,n)

def residualGlobLMFit(params,experiments,n=1):
    return residualGlob(convertPTableToMat(params,n),experiments,n)

def residualGlob(p,experiments,n=1):
    #Computes the residual for all conditions                                                                                                                                                                      
    res_out = np.array([])
    for i in range(0,len(experiments)):
        '''
        print('')
        print('A0 is ' + str(experiments[i].A0))
        print('I0 is ' + str(experiments[i].I0))
        print('N0 is ' + str(experiments[i].N0))
        '''
        res_out = np.append(res_out,residual(p,experiments[i],n))
    return res_out

def residualGlob_cfu_dyn(p,experiments,OD_to_CFU,n=1):
    #Computes the residual for all conditions                                                                                                                                                                      
    res_out = np.array([])
    for i in range(0,len(experiments)):
        '''
        print('')
        print('A0 is ' + str(experiments[i].A0))
        print('I0 is ' + str(experiments[i].I0))
        print('N0 is ' + str(experiments[i].N0))
        '''
        res_out = np.append(res_out,residual_cfu_dyn(p,experiments[i],OD_to_CFU,n))
    return res_out

def RMSE(p,experiments,n=1):
    rmse_out = np.sqrt(np.mean((residualGlob(p,experiments,n))**2))
    return rmse_out
'''
def residual(p,experiment,n=1):
    #Compute the residual vector for the A and I variables using the replicate measurements taken in a given condition     
    try:
        #print('in try')
        y0 = [np.nanmedian(experiment.N0[:,0]), p[16], np.nanmedian(experiment.A[:,0]), np.nanmedian(experiment.I[:,0])]
    except:
        #print('in except') 
        y0 = [experiment.N0, p[16], np.nanmedian(experiment.A[:,0]), np.nanmedian(experiment.I[:,0])]
        #print(y0)
    yh = ntm.denitODE(y0,experiment.t,p,n)
    res = np.ravel([experiment.A-yh[:,-2],experiment.I-yh[:,-1]])
    print(res)
    res = res[~np.isnan(res)] #remove any nan elements                                                                                                                                               
    return res
'''

def residual(p,experiment,n=1):
    #Compute the residual vector for the A and I variables using the replicate measurements taken in a given condition     
    if n == 1:
        try:
            #print('in try')
            y0 = [np.nanmedian(experiment.N0[:,0]), p[16], np.nanmedian(experiment.A[:,0]), np.nanmedian(experiment.I[:,0])]
        except:
            #print('in except') 
            y0 = [experiment.N0, p[16], np.nanmedian(experiment.A[:,0]), np.nanmedian(experiment.I[:,0])]
            #print(y0)
        #print('initial conditions are ' + str(y0))
        yh = ntm.denitODE(y0,experiment.t,p,n)
        res = np.ravel([experiment.A-yh[:,-2],experiment.I-yh[:,-1]])
        res = res[~np.isnan(res)] #remove any nan elements                                                           
    else:
        y0 = []
        for i in range(n):
            try:
                y0.append(np.nanmedian(experiment.N0[i,:,0]))
                y0.append(p[i,16]) 
            except:
                y0.append(np.nanmedian(experiment.N0[i]))
                y0.append(p[i,16])
        y0.append(np.nanmedian(experiment.A[:,0]))
        y0.append(np.nanmedian(experiment.I[:,0]))
        yh = ntm.denitODE(y0,experiment.t,p,n)
        res = np.ravel([experiment.A-yh[:,-2],experiment.I-yh[:,-1]])        
        res = res[~np.isnan(res)] #remove any nan elements                                               
    return res

