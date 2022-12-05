import numpy as np

def generator_coin_proba_uniform(K : int):
    return 0.5*np.ones(2*K)

def generator_coin_biais(K : int, pa: float):
    return pa*np.ones(2*K)

def generator_coin_biais_group0(K : int, pa: float):
    return np.concatenate([pa*np.ones(K),0.5*np.ones(K)])

def generator_dice_proba_uniform(N : int):
    return 1/6*np.ones((N,2,6))

def generator_dice_proba_uniform_bias6(N : int, k : int):
    a = 1/6*np.ones((N,1,6))
    b = np.array([[[1/(6+k),1/(6+k),1/(6+k),1/(6+k),1/(6+k),(1+k)/(6+k)]]]*N)
    return np.concatenate([a,b],axis=1)

def generator_dice_proba_bias6_biasN(N : int, k : int):
    b=[]
    a = np.array([[[1/(6+k),1/(6+k),1/(6+k),1/(6+k),1/(6+k),(1+k)/(6+k)]]]*N)
    for i in range(N):
        b.append([[1/(6+i),1/(6+i),1/(6+i),1/(6+i),1/(6+i),(1+i)/(6+i)]])
    b = np.array(b)
    return np.concatenate([a,b],axis=1)