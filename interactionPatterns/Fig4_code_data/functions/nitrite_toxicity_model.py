import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from lmfit import Minimizer, conf_interval
import statsmodels.api as sm
from sklearn.metrics import r2_score
import copy
import random as rd


def effective_flux_model_abs_dAdt(dAdt_1, dAdt_2, OD_1, OD_2, OD_c, f_1, f_2):
    return OD_c * (np.abs(dAdt_1) * f_1 / OD_1 + np.abs(dAdt_2) * f_2 / OD_2)

def effective_flux_model_dIdt(dIdt_1, dIdt_2, OD_1, OD_2, OD_c, f_1, f_2):
    return OD_c * (dIdt_1 * f_1 / OD_1 + dIdt_2 * f_2 / OD_2)


class experiment:
    #This class contains the experimental data for a given condition
    def __init__(self, ID, phen, N0, Nend, A0, A, I0, I, t, N = None, N0_Nend_avg=None, Aend = None, Iend = None): #added N for continuous OD measurement
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
        self.Aend = Aend
        self.Iend = Iend

class enrichment_experiment:
    #This class contains the experimental data for a given condition
    def __init__(self, ID, phen, N0, Nend, A0, A, I0, I, t, succ_conc, pH): #added N for continuous OD measurement
        self.ID = ID
        self.phen = phen
        self.N0 = N0
        self.Nend = Nend
        self.A0 = A0
        self.A = A
        self.I0 = I0
        self.I = I
        self.t = t
        self.pH = pH
        self.succ_conc = succ_conc

def fitYields(experiments):                   #kyle added offset 08/07/2021
    if experiments[0].phen == 'Nar/Nir':
        DelA = np.array([])
        DelI = np.array([])
        DelOD = np.array([])
        for i in range(0,len(experiments)):
            if (experiments[i].Aend is None) and (experiments[i].Iend is None):
                DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
                DelI = np.append(DelI,experiments[i].A0-experiments[i].A[:,-1] + experiments[i].I0-experiments[i].I[:,-1])
            else:
                DelA = np.append(DelA,experiments[i].A0-experiments[i].Aend)
                DelI = np.append(DelI,experiments[i].A0-experiments[i].Aend + experiments[i].I0-experiments[i].Iend)
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelA.reshape(-1, 1), DelI.reshape(-1, 1),axis=1)
        x = np.append(x,np.ones((len(DelOD),1)),axis=1)
    if experiments[0].phen == 'Nar':
        DelA = np.array([])
        DelOD = np.array([])
        for i in range(0,len(experiments)):
            DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelA.reshape(-1, 1), np.ones((len(DelOD),1)),axis=1)
    if experiments[0].phen == 'Nir':
        DelI = np.array([])
        DelOD = np.array([])
        for i in range(0,len(experiments)):
            DelI = np.append(DelI,experiments[i].I0-experiments[i].I[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelI.reshape(-1, 1), np.ones((len(DelOD),1)),axis=1)

    mod = sm.OLS(DelOD, x)
    res = mod.fit()
    ci = res.conf_int(0.32)   # 68% confidence interval. comparable to 1 standard error
    offset = res.params[2]
    #compute R2
    DelOD_pred = res.predict()
    r2 = r2_score(DelOD,DelOD_pred)
    
    if experiments[0].phen == 'Nar/Nir':
        gamA = res.params[0]
        gamI = res.params[1]
    elif experiments[0].phen == 'Nar':
        gamA = res.params[0]
        gamI = 0
        ci = np.array((ci[0],np.zeros(2),ci[1]))
    elif experiments[0].phen == 'Nir':
        gamA = 0
        gamI = res.params[0]
        ci = np.array((np.zeros(2),ci[0],ci[1]))

    return gamA, gamI, ci, r2, offset
   
def fitYields_no_offset(experiments):               
    if experiments[0].phen == 'Nar/Nir':
        DelA = np.array([0])
        DelI = np.array([0])
        DelOD = np.array([0])
        for i in range(0,len(experiments)):
            if (experiments[i].Aend is None) and (experiments[i].Iend is None):
                DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
                DelI = np.append(DelI,experiments[i].A0-experiments[i].A[:,-1] + experiments[i].I0-experiments[i].I[:,-1])
            else:
                DelA = np.append(DelA,experiments[i].A0-experiments[i].Aend)
                DelI = np.append(DelI,experiments[i].A0-experiments[i].Aend + experiments[i].I0-experiments[i].Iend)
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
        x = np.append(DelA.reshape(-1, 1), DelI.reshape(-1, 1),axis=1)
    if experiments[0].phen == 'Nar':
        DelA = np.array([0])
        DelOD = np.array([0])
        for i in range(0,len(experiments)):
            DelA = np.append(DelA,experiments[i].A0-experiments[i].A[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
    if experiments[0].phen == 'Nir':
        DelI = np.array([0])
        DelOD = np.array([0])
        for i in range(0,len(experiments)):
            DelI = np.append(DelI,experiments[i].I0-experiments[i].I[:,-1])
            DelOD = np.append(DelOD,experiments[i].Nend-experiments[i].N0)
    mod = sm.OLS(DelOD, x)
    res = mod.fit()
    ci = res.conf_int(0.32)   # 68% confidence interval. comparable to 1 standard error
    #compute R2
    DelOD_pred = res.predict()
    r2 = r2_score(DelOD,DelOD_pred)
    
    if experiments[0].phen == 'Nar/Nir':
        gamA = res.params[0]
        gamI = res.params[1]
    elif experiments[0].phen == 'Nar':
        gamA = res.params[0]
        gamI = 0
        ci = np.array((ci[0],np.zeros(2),ci[1]))
    elif experiments[0].phen == 'Nir':
        gamA = 0
        gamI = res.params[0]
        ci = np.array((np.zeros(2),ci[0],ci[1]))

    return gamA, gamI, ci, r2

def fitRates(params,experiments,initial_conditions,n=1):  
    ##print(initial_conditions)
    fitter = Minimizer(residualGlobLMFit, params, fcn_args=(experiments,initial_conditions,))
    best_result = fitter.minimize(method = 'differential_evolution')
    return best_result

def convertPTableToMat(params, initial_conditions,n=1):
    if n == 1:
        p_out = [params['rA'].value, params['rI'].value, params['kA'].value, params['kI'].value, params['gamA'].value, params['gamI'].value, params['OD0'].value, params['t_lag'].value, params['offset'].value, params['alphaA'].value, params['ItoxA'],params['alphaI'].value, params['ItoxI']]
    else:
        p_out = np.zeros((n,6))
        for i in range(0,n):
            p_out[i,0] = params[i]['rA'].value
            p_out[i,1] = params[i]['rI'].value
            p_out[i,2] = params[i]['kA'].value
            p_out[i,3] = params[i]['kI'].value
            p_out[i,4] = params[i]['gamA'].value
            p_out[i,5] = params[i]['gamI'].value
    return p_out


def residualGlobLMFit(params,experiments,initial_conditions,n=1):
    #print(initial_conditions)
    return residualGlob(convertPTableToMat(params,initial_conditions,n),experiments,initial_conditions,n)

def residualGlob(p,experiments,initial_conditions,n=1):
    #Computes the residual for all conditions
    ##print(initial_conditions)
    res_out = np.array([])
    for i in range(0,n):
        res_out = np.append(res_out,residual(p,experiments,initial_conditions,n))
    return res_out


def residualGlob_global(p,experiments,initial_conditions,n=1):
    #Computes the residual for all conditions
    ##print(initial_conditions)
    res_out = np.array([])
    for i in range(0,len(experiments)):
        res_out = np.append(res_out,residual(p[i],experiments[i],initial_conditions,n))
    return res_out

def residualGlobLMFit_global(params,experiments,initial_conditions,n=1):
    #print(initial_conditions)
    return residualGlob_global(convertPTableToMat_global(params,initial_conditions,experiments,n),experiments,initial_conditions,n)

def RMSE(p,experiments,n=1):
    rmse_out = np.sqrt(np.mean((residualGlob(p,experiments,n))**2))
    return rmse_out

def residual(p,experiment,initial_conditions,offset, n=1):
    #Compute the residual vector for the A and I variables using the replicate measurements taken in a given condition
    N0 = p[6]
    y0 = np.append(N0, [experiment.A0, experiment.I0])


    all_t = experiment.t
    growth_t = experiment.t[experiment.t>p[7]] #only fit times after the lag phase
    yh = denitODE(y0,growth_t,p,n)
    OD = np.insert(yh[:,0], 0, np.ones((len(all_t)-len(growth_t)))*p[6])
    offset = p[8]
    res = np.ravel([np.log(experiment.N)-np.log(OD+offset)])
    res = res[~np.isnan(res)] #remove any nan elements
    return res

def denitODE(y0,t,p,n=1, tol=1e-6):
    sol = odeint(F, y0, t, args=(p,n), rtol=tol)
    return sol

def F(y,t,p,n):
    #ODE RHS
    #takes p as an ndarray where each row contains the parameters for a given strain
    Fout = np.zeros(2*n+2)
    if n > 1:
        for i in range(0,n):
            Fout[ i*2]  = f(p[i],[y[i*2],y[i*2+1],y[-2],y[-1]])
            Fout[ i*2+1]  = f_death(p[i],[y[i*2],y[i*2+1],y[-2],y[-1]])
            Fout[-2] += g(p[i],[y[i*2],y[i*2+1],y[-2],y[-1]])
            Fout[-1] += h(p[i],[y[i*2],y[i*2+1],y[-2],y[-1]])
    elif n==1:
            Fout[0]  = f(p,[y[0],y[1],y[-2],y[-1]])
            Fout[1]  = f_death(p,[y[0],y[1],y[-2],y[-1]])
            Fout[-2] += g(p,[y[0],y[1],y[-2],y[-1]])
            Fout[-1] += h(p,[y[0],y[1],y[-2],y[-1]])  
    else:
        raise NameError('check this!')
    return Fout

def f(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rd_max = p[13]
    I_tox_d = p[15]
    alpha_d = p[14]
    r_death = rd_max * 1/(1 + np.exp(-alpha_d*(y[3] - I_tox_d)))

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))
    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))

    if y[0] >= 0:
        return gamA*rA*y[0]*y[2]/(kA + y[2]) + gamI*rI*y[0]*y[3]/(kI + y[3]) - r_death * y[0]
    else:
        return 0
    
def dNdt(p, N, A, I):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rd_max = p[13]
    I_tox_d = p[15]
    alpha_d = p[14]
    r_death = rd_max * 1/(1 + np.exp(-alpha_d*(I - I_tox_d)))

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(I - I_tox_A))))
    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(I - I_tox_I))))
    if N >= 0:
        return gamA*rA*N*A/(kA + A) + gamI*rI*N*I/(kI + I) - r_death * N
    else:
        return 0

def f_death(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rd_max = p[13]
    I_tox_d = p[15]
    alpha_d = p[14]
    r_death = rd_max * 1/(1 + np.exp(-alpha_d*(y[3] - I_tox_d)))
    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))
    return r_death * y[0]

def g(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    kA     = p[2]
    alpha_A  = p[9]
    I_tox_A  = p[10]
    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))


    return -1*rA*y[0]*y[2]/(kA + y[2])

def dAdt(p,N,A,I, n=1):
    if n > 1:
        dAdt = 0
        for i in range(n):
            N_strain = N[i]
            p_strain = p[i]
            rA_max     = p_strain[0]
            kA     = p_strain[2]
            alpha_A  = p_strain[9]
            I_tox_A  = p_strain[10]
            rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(I - I_tox_A))))
            dAdt = dAdt - 1*rA*N_strain*A/(kA + A)
        return dAdt
    elif n == 1:
        rA_max     = p[0]
        kA     = p[2]
        alpha_A  = p[9]
        I_tox_A  = p[10]
        rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(I - I_tox_A))))
        return -1*rA*N*A/(kA + A)
    else:
        raise NameError('whats going on here')

def h(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))

    return rA*y[0]*y[2]/(kA + y[2]) - rI*y[0]*y[3]/(kI + y[3])

def dIdt(p,N, A, I, n = 1):
    if n > 1:
        dIdt = 0
        for i in range(n):
            N_strain = N[i]
            p_strain = p[i]
            rA_max     = p_strain[0]
            rI_max     = p_strain[1]
            kA     = p_strain[2]
            kI = p_strain[3]
            alpha_A  = p_strain[9]
            I_tox_A  = p_strain[10]
        
            alpha_I  = p_strain[11]
            I_tox_I  = p_strain[12]

            rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(I - I_tox_A))))
        
            rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(I - I_tox_I))))
            dIdt = dIdt + 1*rA*N_strain*A/(kA + A) - rI*N_strain*I/(kI + I)
        return dIdt
    elif n == 1:
        rA_max     = p[0]
        rI_max     = p[1]
        kA     = p[2]
        kI     = p[3]
        alpha_A  = p[9]
        I_tox_A  = p[10]
        
        alpha_I  = p[11]
        I_tox_I  = p[12]

        rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(I - I_tox_A))))
        
        rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(I - I_tox_I))))
        
        return rA*N*A/(kA + A) - rI*N*I/(kI + I)
        
def dfdN(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))
    return gamA*rA*y[2]/(kA + y[2]) + gamI*rI*y[3]/(kI + y[3])

def dfdA(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    kA     = p[2]
    gamA   = p[4]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))
    return gamA*rA*kA*y[0]/(kA + y[2])**2

def dfdI(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]
    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))
    return gamI*rI*kI*y[0]/((kI + y[3])**2)+gamI*drIdI(p,y)*y[3]*y[0] / (kI + y[3])+gamA*drAdI(p,y)*y[2]*y[0] / (kA + y[2])

def dgdN(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    kA     = p[2]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))
    return -1*rA*y[2]/(kA + y[2])

def dgdA(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    kA     = p[2]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))
    return -1*rA*kA*y[0]/(kA + y[2])**2

def dgdI(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    kA     = p[2]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    return -1*drAdI(p,y)*y[0]*y[2]/(kA + y[2])

def dhdN(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))
    return rA*y[2]/(kA + y[2]) - rI*y[3]/(kI + y[3])

def dhdA(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    kA     = p[2]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    return rA*kA*y[0]/(kA + y[2])**2

def dhdI(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]
    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))
    return -1*rI*kI*y[0]/((kI + y[3])**2)  - drIdI(p,y)* y[3]*y[0] / (kI + y[3]) + drAdI(p,y)*y[2]*y[0]/(kA+y[2])

def drAdI(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))
    
    return -1*rA_max*alpha_A*np.exp(-1*alpha_A*(y[3] - I_tox_A))/((1 + np.exp(-1*alpha_A*(y[3] - I_tox_A)))**2)
    
def drIdI(p,y):
    #takes p as a 1darray
    rA_max     = p[0]
    rI_max     = p[1]
    kA     = p[2]
    kI     = p[3]
    gamA   = p[4]
    gamI   = p[5]
    alpha_A  = p[9]
    I_tox_A  = p[10]

    alpha_I  = p[11]
    I_tox_I  = p[12]

    rA = rA_max * (1 - 1/(1 + np.exp(-alpha_A*(y[3] - I_tox_A))))

    rI = rI_max * (1 - 1/(1 + np.exp(-alpha_I*(y[3] - I_tox_I))))
    

    
    return -1*rI_max*alpha_I*np.exp(-1*alpha_I*(y[3] - I_tox_I))/((1 + np.exp(-1*alpha_I*(y[3] - I_tox_I)))**2)

def rd_val(rd_max, alpha_d, I, I_tox_d):
    return rd_max * 1/(1 + np.exp(-alpha_d*(I - I_tox_d)))
    
def rA_val(rA_max, alpha_A, I, I_tox_A):
    return rA_max * (1 - 1/(1 + np.exp(-alpha_A*(I - I_tox_A))))
    

def rI_val(rI_max, alpha_I, I, I_tox_I):
    return rI_max * (1 - 1/(1 + np.exp(-alpha_I*(I - I_tox_I))))
    
def observed_OD(N_cells, N_cells_dead, alpha_cells, alpha_cells_dead):
    return N_cells * alpha_cells + N_cells_dead * alpha_cells_dead
