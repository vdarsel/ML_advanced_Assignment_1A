import numpy as np
from math import *
import matplotlib.pyplot as plt
import scipy.stats as scs


def update_mu(mu_prime: float, lambd: float, a_0: float, b_0 : float, X: np.ndarray):
    mu = (lambd*mu_prime+np.sum(X))/(lambd+len(X))
    exeptation_tau = a_0/b_0
    tau = (lambd+len(X))*exeptation_tau
    return mu,tau

def update_tau(mu_prime: float, lambd: float , tau_0: float, mu_0: float, a: float, b : float, X: np.ndarray):
    N=len(X)
    a_1 = a + (N+1)/2
    b_1 = b + 1/2*(np.sum(np.power(X,2))+(np.power(mu_prime,2)*lambd)-2*mu_0*(np.sum(X)+lambd*mu_prime)+(len(X)+lambd)*(1/tau_0+mu_0**2))
    return a_1, b_1


def VI(X: np.ndarray,mu_prime : float,lambd : float,a : float,b : float, threshold: float):
    mu_0, tau_0, a_0, b_0 = 5, 5, 5, 5
    mu_1, tau_1, a_1, b_1 = 0, 1, 1, 1
    while(max(np.power(mu_0-mu_1,2),np.power(tau_0-tau_1,2),np.power(a_0-a_1,2),np.power(b_0-b_1,2))>threshold):
        mu_0, tau_0, a_0, b_0 = mu_1, tau_1, a_1, b_1
        mu_1,tau_1=update_mu(mu_prime,lambd,a_1,b_1,X)
        a_1,b_1 = update_tau(mu_prime,lambd,tau_1,mu_1,a,b,X)
    return mu_0, tau_0, a_0, b_0

def exactPosterior(X: np.ndarray,mu_prime : float,lambd : float,a : float,b : float):
    N=len(X)
    mu_star = (np.sum(X)+lambd*mu_prime)/(N+lambd)
    lambd_star = N+lambd
    a_star = N/2+a
    b_star = np.sum(np.power(X,2))/2+b+lambd*np.power(mu_prime,2)/2-lambd_star*np.power(mu_star,2)/2
    return mu_star,lambd_star,a_star,b_star

def generate_sample(N,mu_prime,lambd,a,b):
    T = np.random.gamma(a,1/b)
    m = np.random.normal(mu_prime,1/np.sqrt(lambd*T))
    X= np.random.normal(m,1/np.sqrt(T),N)
    return T,m,X


def print_comparison(val_VI: list, val_post: list, mu_prime : float,lambd : float,a : float,b : float,mu_val:float, tau_val:float, N: int, prior=False):
    x_mu = np.linspace(val_post[0]-2*np.sqrt(val_post[3]/(val_post[1]*val_post[2])),val_post[0]+2*np.sqrt(val_post[3]/(val_post[1]*val_post[2])),100)
    y_tau = np.linspace(max(val_post[2]/val_post[3]*(1-2/(np.sqrt(lambd*N))),0.01),val_post[2]/val_post[3]*(1+2/(np.sqrt(lambd*N))),100)
    if (prior):
        x_mu= np.linspace(min(val_post[0],mu_prime,mu_val)-2*np.sqrt(val_post[3]/(val_post[1]*val_post[2])),max(val_post[0],mu_prime,mu_val)+2*np.sqrt(val_post[3]/(val_post[1]*val_post[2])),100)
        y_tau = np.linspace(max(min(a/b/10,tau_val,val_post[2]/val_post[3]/5),0.05),max(a/b,val_post[2]/val_post[3]*2),100)
    XX, YY = np.meshgrid(x_mu,y_tau)

    q_tau = scs.gamma.pdf(x=y_tau,a=val_VI[2],scale=1/val_VI[3])
    q_mu  = scs.norm(val_VI[0],1/np.sqrt(val_VI[1])).pdf(x_mu)
    M = q_mu*np.transpose([q_tau])

    p_tau = scs.gamma.pdf(YY,a=val_post[2],scale=1/val_post[3])
    p_mu = scs.norm(val_post[0],np.sqrt(1/(val_post[1]*YY))).pdf(XX)
    M2 = p_tau*p_mu

    pp_tau = scs.gamma.pdf(YY,a=a,scale=1/b)
    pp_mu = scs.norm(mu_prime,1/np.sqrt(lambd*YY)).pdf(XX)
    M3 = pp_tau*pp_mu

    plt.figure(figsize=(10,8))
    plt.contour(XX,YY,M,colors='b',alpha=0.4)
    plt.contour(XX,YY,M2,colors='r',alpha=0.4)
    plt.scatter(val_VI[0],val_VI[2]/val_VI[3], marker="+",s=100, label="Variational Inference")
    plt.scatter(val_post[0],val_post[2]/val_post[3],s=50, label="Exact Posterior")
    if (prior):
        plt.contour(XX,YY,M3,colors='g',alpha=0.4)
        plt.scatter(mu_prime,a/b, label="Prior distribution")
        plt.scatter(mu_val,tau_val, label="Exact value")
    plt.legend(fontsize=10)
    plt.xlabel(r"Mean value $\mu$", fontsize=20)
    plt.ylabel(r"Precision value $\tau$", fontsize=20)
    plt.title(r"Probabilities for N = "+str(N)+" and $\lambda$="+str(round(lambd,2)))
    plt.xlim(x_mu[0],x_mu[-1])
    plt.ylim(y_tau[0],y_tau[-1])
    plt.show()