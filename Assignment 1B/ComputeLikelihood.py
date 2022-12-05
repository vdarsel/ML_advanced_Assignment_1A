import numpy as np

def p_one_dice_r(k: int, K: int, ri: int, player: int, tab_coin: np.ndarray, tab_dice: np.ndarray):
    res = np.zeros(6)
    pA = tab_coin[ri*K+k]
    #ri*K+k=K+k or k
    for i in range(6):
        res[i] = pA*tab_dice[player][0][i]+(1-pA)*tab_dice[player][1][i]
    return res

def p_s_r(k: int, K: int, s: int, ri: int, player: int, tab_coin: np.ndarray, tab_dice: np.ndarray):
    p=p_one_dice_r(k,K,ri,player,tab_coin,tab_dice)
    res=0
    sum_val = s[k]
    for i in range(max(0,sum_val-7),min(6,sum_val-1)):
        res+=p[i]*p[sum_val-i-2]
    return res

def p_r_r(ri: int, ri_previous: int):
    if (ri==ri_previous):
        return 0.25
    else:
        return 0.75
    
def alpha(ri : int, ri_previous : int, i: int, K: int, 
          tab_alpha: np.ndarray, player: int, tab_dice: np.ndarray,
          tab_coin: np.ndarray, s : np.ndarray):
          #tab_alpha[i][ri][ri_previous]=alpha_i(ri,ri_previous)
    if(type(tab_alpha)==type(None)): # initiate if needed
        tab_alpha = np.empty((K,2,2))
        tab_alpha[:]=np.nan
    if(np.isnan(tab_alpha[i][ri][ri_previous])):
        p_s = p_s_r(i, K, s, ri, player, tab_coin,tab_dice)
        p_r = p_r_r(ri,ri_previous)
        if(i==(K-1)):
            bonus = 1/(np.power(0.25,int(K/2))*1/6*1/6) 
            #to avoid to small value. Since we compute a ratio of alpha at the end, it is ok
            tab_alpha[i][ri][ri_previous] = p_s*p_r*bonus
        else:
            tab_alpha, r_next_0 = alpha(0,ri,i+1,K,tab_alpha,player,tab_dice,tab_coin,s)
            tab_alpha, r_next_1 = alpha(1,ri,i+1,K,tab_alpha,player,tab_dice,tab_coin,s)
            tab_alpha[i][ri][ri_previous] = p_s*p_r*(r_next_0+r_next_1)
    if(np.isnan(tab_alpha[i][ri][ri_previous])):
        print(i,ri,ri_previous)
    return tab_alpha,tab_alpha[i][ri][ri_previous]

def p_ri(ri: int, ri_previous: int, i: int, s: np.ndarray, K: int, 
          tab_alpha: np.ndarray, player: int, tab_dice: np.ndarray,
          tab_coin: np.ndarray):
    tab_alpha,r = alpha(ri,ri_previous,i, K, tab_alpha, player,tab_dice,tab_coin,s)
    tab_alpha,r_minus = alpha(1-ri,ri_previous,i, K, tab_alpha, player,tab_dice,tab_coin,s)
    return tab_alpha,r/(r+r_minus)


def p_all_r(r: np.ndarray, s: np.ndarray, player: int, tab_dice : np.ndarray, tab_coin: np.ndarray,  tab_alpha: np.ndarray=None):
    K = int(len(tab_coin)/2)
    if(type(tab_alpha)==type(None)): # initiate if needed
        tab_alpha = np.empty((K,2,2))
        tab_alpha[:]=np.nan
    # suppose r0=1
    tab_alpha,res = p_ri(r[0],1,0,s,K,tab_alpha,player,tab_dice,tab_coin)
    for i in range(1,K):
        tab_alpha,temp = p_ri(r[i],r[i-1],i,s,K,tab_alpha,player,tab_dice,tab_coin)
        #update tab_alpha all the time in order to compute it only once
        res=temp*res
    return res,tab_alpha