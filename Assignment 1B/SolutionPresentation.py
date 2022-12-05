from ComputeLikelihood import *
from GenerateParameter import *
from GenerateModel import *
from GetRandomValue import *

def complete_set_event(c : np.ndarray, d: np.ndarray):
    c= generator_coin_proba_uniform(3)
    d= generator_dice_proba_uniform(2)
    res=0
    tab_alpha=None
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a,tab_alpha =p_all_r(np.array([i,j,k]),np.array([3,5,8]),1,d,c,tab_alpha)
                res+=a
    return res

def test_Likelihood():
    i=0
    c = generator_coin_proba_uniform(3)
    d = generator_dice_proba_uniform(2)
    i+=1
    print("Test "+str(i)+":\t",abs(complete_set_event(c,d)-1)<1e-5)
    c = generator_coin_biais(3,0.3)
    d = generator_dice_proba_uniform_bias6(2,5)
    i+=1
    print("Test "+str(i)+":\t",abs(complete_set_event(c,d)-1)<1e-5)
    c = generator_coin_biais(3,0.3)
    d = generator_dice_proba_bias6_biasN(2,0)
    i+=1
    print("Test "+str(i)+":\t",abs(complete_set_event(c,d)-1)<1e-5)

def test_Random_variable():
    i=0
    K=5
    bias_dice = 1
    player = 5
    attempt = 3000
    tab_dice = generator_dice_proba_bias6_biasN(player+1,bias_dice)
    tab_dice_2 = generator_dice_proba_uniform_bias6(player+1,bias_dice)
    tab_coin = generator_coin_biais_group0(K,0.1)
    s_n= experiment(tab_dice,tab_coin)
    s_2= np.random.randint(2,13,K)
    s_3= np.ones(K,int)*12
    i+=1
    print("Test "+str(i)+":")
    Verify_Model(attempt,s_n[player],player,tab_dice,tab_coin,None,True)
    i+=1
    print("Test "+str(i)+":")
    Verify_Model(attempt,s_2,player,tab_dice,tab_coin,None)
    i+=1
    print("Test "+str(i)+":")
    Verify_Model(attempt,s_3,player,tab_dice,tab_coin,None)
    i+=1
    print("Test "+str(i)+":")
    Verify_Model(attempt,s_3,player,tab_dice_2,tab_coin,None)
    attempts = [1000,3000,10000]
    i+=1
    print("Test "+str(i)+": Evolution with sample size")
    for att in attempts:
        Verify_Model(att,s_2,player,tab_dice,tab_coin)


def main():
    print("Test of ComputeLikelihood...")
    test_Likelihood()
    print("\n\n\nTest of GetRandomValue...")
    test_Random_variable()

if __name__ == "__main__":
    main()