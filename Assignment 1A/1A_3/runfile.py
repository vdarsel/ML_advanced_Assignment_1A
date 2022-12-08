import numpy as np
from SupportFunction import *


def Model(N: int, lambd: float, prior:bool):
    threshold = 0.1
    mu_prime,a,b = 5, 60, 4
    T,m,X = generate_sample(N,mu_prime,lambd,a,b)
    print(m)
    v = VI(X,mu_prime,lambd,a,b,threshold)
    p = exactPosterior(X,mu_prime,lambd,a,b)
    print_comparison(v,p,mu_prime,lambd,a,b,m,T,N,prior)



def main():
    print("Case 1 : lambda>>N")
    N=3
    lambd = 30
    Model(N,lambd,True)
    print("Case 2 : lambda=N")
    N=10
    lambd = 10
    Model(N,lambd,True)
    print("Case 3 : lambda<<N")
    N=1000
    lambd = 2
    Model(N,lambd,False)
    

if __name__ == "__main__":
    main()
