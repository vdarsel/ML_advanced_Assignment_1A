import numpy as np
import scipy.stats as scs
from math import *
import matplotlib.pyplot as plt

def initiate(k: int, X_1:np.ndarray,X_2: np.ndarray,S: np.ndarray):
    '''Initiailisation by taking points as far as possible the one from the other'''
    ids = []
    ids.append(np.argmax(X_1))
    for j in range(k-1):
        ids.append(np.argmax([np.sum(np.array([(X_1[i]-X_1[id])**2+(X_2[i]-X_2[id])**2+(S[i]-S[id])**2 for id in ids])) for i in range(len(X_1)) if i not in ids]))
    mu_1 = np.array([X_1[id] for id in ids])
    mu_2 = np.array([X_2[id] for id in ids])
    tau_1 = np.ones(k)
    tau_2 = np.ones(k)
    lambd = np.array([S[id]+1 for id in ids])
    pi = np.ones(k)/k
    return mu_1,mu_2,tau_1,tau_2,lambd,pi


def E_step(K: int,X_1: np.ndarray,X_2: np.ndarray, S: np.ndarray, previous_mu_1: np.ndarray,previous_mu_2: np.ndarray,previous_tau_1: np.ndarray,previous_tau_2: np.ndarray,previous_lambda: np.ndarray,previous_pi: np.ndarray):
    '''
    E-step
    '''
    E = np.zeros((len(X_1),K))
    log_likelihood = 1
    for n in range(len(X_1)):
        for k in range(K):
            p_X_1 = scs.norm(previous_mu_1[k],np.sqrt(1/previous_tau_1[k])).pdf(X_1[n])
            p_X_2 = scs.norm(previous_mu_2[k],np.sqrt(1/previous_tau_2[k])).pdf(X_2[n])
            p_S = scs.poisson(previous_lambda[k]).pmf(S[n])
            p_Z = previous_pi[k]
            E[n][k]=p_X_1*p_X_2*p_S*p_Z
        log_likelihood += np.log(np.sum(E[n]))
        E[n] = E[n]/np.sum(E[n]) #normalize
    return E,log_likelihood
    
def M_step(K: int,X_1: np.ndarray,X_2: np.ndarray, S: np.ndarray, A : np.ndarray):
    mu_1 = np.zeros(K)
    mu_2 = np.zeros(K)
    tau_1 = np.zeros(K)
    tau_2 = np.zeros(K)
    lambd = np.zeros(K)
    pi = np.zeros(K)
    for k in range(K): #maximization / begin M-step
        if(np.sum(A[k])<1e-5):
            A[k] = A[k]/np.sum(A[k])
        pi[k]=np.sum(A[k])
        mu_1[k] = np.sum(A[k]*X_1)/np.sum(A[k])
        mu_2[k] = np.sum(A[k]*X_2)/np.sum(A[k])
        tau_1[k] = min(1e4,np.sum(A[k])/max(1e-8,np.sum(A[k]*np.power(X_1-mu_1[k],2))))
        tau_2[k] = min(1e4,np.sum(A[k])/max(1e-8,np.sum(A[k]*np.power(X_2-mu_2[k],2))))
        lambd[k] = np.sum(A[k]*S)/(np.sum(A[k]))
    pi = pi/np.sum(pi)
    return mu_1,mu_2,tau_1,tau_2,lambd,pi


def EM(threshold: float, K: int, X_1: np.ndarray,X_2: np.ndarray, S: np.ndarray):
    '''Full EM algorithm'''
    N= len(X_1)
    mu_1,mu_2,tau_1,tau_2,lambd,pi = initiate(K,X_1,X_2,S)
    # previous_mu_1,previous_mu_2,previous_tau_1,previous_tau_2,previous_lambda,previous_pi = mu_1+1,mu_2,tau_1*3,tau_2*3,lambd,pi*3
    # loop=1
    log_like = 0
    change = True
    while(change):
        # loop+=1
        # previous_mu_1,previous_mu_2,previous_tau_1,previous_tau_2,previous_lambda,previous_pi = mu_1.copy(),mu_2.copy(),tau_1.copy(),tau_2.copy(),lambd.copy(),pi.copy()
        A,new_log_like = E_step(K,X_1,X_2,S,mu_1,mu_2,tau_1,tau_2,lambd,pi)
        if (np.abs(new_log_like-log_like)<threshold):
            change=False
        log_like = new_log_like
        A = A.transpose()
        mu_1,mu_2,tau_1,tau_2,lambd,pi = M_step(K,X_1,X_2,S,A)
    return mu_1,mu_2,tau_1,tau_2,lambd,pi
  


def read_files(X_file_name: str, S_file_name: str):
    X_file ="data\\"+ X_file_name
    S_file ="data\\"+ S_file_name
    with open(X_file) as file:
        X = file.readlines()
    X_1 = np.zeros(len(X))
    X_2 = np.zeros(len(X))
    for i,line in enumerate(X):
        X_1[i]=float(line.split(" ")[0])
        X_2[i]=float(line.split("\n")[0].split(" ")[1])
    with open(S_file) as file:
        S_temp = file.readlines()
    S = np.zeros(len(S_temp))
    for i,line in enumerate(S_temp):
        S[i]=float(line.split("\n")[0])
    return X_1, X_2, S


def proba_point(x_1,x_2,s,mu_1,mu_2,tau_1,tau_2,lambd,pi):
    k = len(mu_1)
    res = np.zeros(k)
    for i in range(k):
        res[i] = pi[i] * scs.norm(loc=mu_1[i],scale=np.sqrt(1/tau_1[i])).pdf(x_1) * scs.norm(loc=mu_2[i],scale=1/tau_2[i]).pdf(x_2) * scs.poisson(mu=lambd[i]).pmf(s)
    return res

def affectation (X_1 : np.ndarray, X_2 : np.ndarray, S : np.ndarray, 
                    mu_1 : np.ndarray, mu_2 : np.ndarray, tau_1 : np.ndarray, tau_2 : np.ndarray,
                    lambd : np.ndarray, pi : np.ndarray):
    res = np.zeros(len(X_1))-1
    proba = np.zeros(len(X_1))-1
    for i in range(len(X_1)):
        temp = proba_point(X_1[i],X_2[i],S[i],mu_1,mu_2,tau_1,tau_2,lambd,pi)
        proba[i] = np.sum(temp)
        res[i] = np.argmax(temp)
    return res