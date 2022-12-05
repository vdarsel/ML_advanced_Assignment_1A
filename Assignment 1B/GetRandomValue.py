import plotly.graph_objects as go
import numpy as np
from itertools import product
from ComputeLikelihood import *
from GenerateParameter import *

def generate_R_i(r_previous: int, i: int, K: int, tab_alpha:np.ndarray, s: np.ndarray, player: int, tab_dice : np.ndarray, tab_coin: np.ndarray):
    tab_alpha, a_1 = alpha(1,r_previous,i,K,tab_alpha,player,tab_dice,tab_coin,s)
    tab_alpha, a_0 = alpha(0,r_previous,i,K,tab_alpha,player,tab_dice,tab_coin,s)
    p = a_1/(a_1+a_0)
    res = np.random.binomial(1,p)
    return tab_alpha,res

def generate_random_value(s: np.ndarray, player: int, tab_dice : np.ndarray, tab_coin: np.ndarray, tab_alpha: np.ndarray=None):
    K = int(len(tab_coin)/2)
    r=np.empty(K,int)
    if(type(tab_alpha)==type(None)): # initiate if needed
        tab_alpha = np.empty((K,2,2))
        tab_alpha[:]=np.nan
    # suppose r0=1
    tab_alpha,a = generate_R_i(1,0,K,tab_alpha,s,player,tab_dice,tab_coin)
    r[0] = int(a)
    for i in range(1,K):
        tab_alpha,a = generate_R_i(r[i-1],i,K,tab_alpha,s,player,tab_dice,tab_coin)
        #update tab_alpha all the time in order to compute it only once
        r[i] = int(a)
    return r,tab_alpha

def decode(trans: np.ndarray):
    res=0
    pow =1
    for i,a in enumerate(trans):
        res= res + a*pow
        pow=pow*2
    return res

def Verify_Model(attempt:int, s: np.ndarray, player: int, tab_dice : np.ndarray, tab_coin: np.ndarray, tab_alpha: np.ndarray=None, printage:bool=False):
    K=int(len(tab_coin)/2)
    possible = [np.array(i) for i in product(range(2), repeat=K)]
    proba = np.empty(len(possible))
    count = np.zeros(len(possible))
    for i in range(attempt):
        a,tab_alpha = generate_random_value(s,player,tab_dice,tab_coin,tab_alpha)
        count[decode(a)]+=1
    for poss in possible:
        a, tab_alpha=p_all_r(poss,s,player,tab_dice,tab_coin,tab_alpha)
        proba[decode(poss)] = a
    print("Gap on "+str(attempt)+" attempts:\t",np.sum(np.abs(proba-count/attempt)))
    if (printage):
        fig = go.Figure(data=[go.Scatter(x=[decode(poss) for poss in possible],y=proba, customdata=possible, hovertemplate='Tables=%{customdata}<br>Probability=%{y}<extra></extra>', mode="markers", name="Real probability"),
                        go.Scatter(x=[decode(poss) for poss in possible],y=count/attempt, customdata=possible, hovertemplate='Tables=%{customdata}<br>Probability=%{y}<extra></extra>', mode="markers", name="Observed reality on "+str(attempt)+" tests")],
                        layout={'xaxis_title':"Possible draws",
                                'yaxis_title':"Proportion/Probability",
                                'title':"Comparison between random draws and probability"})
        fig.write_html('example.html', auto_open=True)
