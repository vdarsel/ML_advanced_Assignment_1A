import numpy as np

def next_table_group(current_table_group: int):
    #current_table= 0 or 1
    r = np.random.binomial(1,0.25)               #proba 0.25 to stay in the table => r = 0 => same table group
    return (1-r)*current_table_group + r*(1-current_table_group) #  r=0 ? current_table_group : 1-current_table_group

def next_table(current_table_group: int, K : int, k : int):
    return current_table_group*K + k

def coin_value(current_table : int, coin_tab_prob: np.ndarray):
    return np.random.binomial(n=1,p=coin_tab_prob[current_table],size=2)

def dice_value(coin_values: np.ndarray, player: int, dice_proba_tab: int):
    res=np.zeros(2,int)
    for i,coin in enumerate(coin_values):
        res[i] = 1+np.random.choice(6,p=dice_proba_tab[player][coin])
    return np.sum(res)

def sum_values(dice_results: np.ndarray):
    return np.sum(dice_results)

def experiment_player(player: int, tab_dice : np.ndarray,tab_coin: np.ndarray):
    r=1
    K= int(len(tab_coin)/2)
    res = np.zeros(K)
    for k in range(K):
        r=next_table_group(r)
        table=next_table(r,K,k)
        c = coin_value(table,tab_coin)
        res[k] = dice_value(c,player,tab_dice)
    return res

def experiment(tab_dice : np.ndarray,tab_coin: np.ndarray):
    K= int(len(tab_coin)/2)
    N = len(tab_dice)
    res = np.zeros((N,K),int)
    for n in range(N):
        res[n] = experiment_player(n,tab_dice,tab_coin)
    return res