# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 21:39:35 2017

@author: Hajime
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections, numpy
import pandas as pd
import copy
import os
import csv
import sys
import scipy.optimize as opt
import scipy.stats as stats
from scipy import interpolate
import itertools
from sklearn.linear_model import LinearRegression

#from theano import tensor as tt
#from theano import function

def SumByGroup(groupid,x,shrink=0):
    nobs = groupid.size
    id_list = np.unique(groupid)
    id_num = id_list.size
    if x.ndim==1:
        x = np.array([x]).T
    nchar = x.shape[1]
    if shrink==0:    
        sums = np.zeros([nobs,nchar])
        for i in range(id_num):
            a = np.sum(x[groupid==id_list[i],:],axis=0)
            sums[groupid==id_list[i],:] =a
        return sums
    if shrink==1:
        sums = np.zeros([id_num,nchar])
        for i in range(id_num):
            a = np.sum(x[groupid==id_list[i],:],axis=0)
            sums[i] = a
        return sums

#Dynamic Demand Simulation
#Assuming one time purchase, therefore the state takes delta only, not f.
#Assuming homogeneous consumers.
#Assuming the number of prducts don't change.
#%%Consumer Problem

#make grids
dgrid_n=20
dgrid_max=10.
dgrid_min=-10.
 
dgrid=np.linspace(dgrid_min,dgrid_max,num=dgrid_n)

N_draw_E=7 # draw for Expectation on delta'
#---temp for coding---
N_prod=3 
N_period=10 
N_obs=N_prod*N_period
N_prod_t=np.ones(N_period)*N_prod #number of produtcts in each t        
t = np.repeat(np.arange(N_period),N_prod)
prodid=np.tile(np.arange(N_prod),N_period)

#%%
alpha_j = np.array([5.,3.,10.])
alpha_p = -5.
P = np.random.randn(N_obs)
lam = np.random.randn(N_obs)

beta = .8        
tol_value_iter=.1
delta_draw=np.random.randn(N_draw_E)
#delta_draw=Halton_draw.halton_randn(1,N_draw_E)[:,0]       


#%%
def Calc_delta(P,lam):
    f = alpha_j[prodid]  +lam    
    deltait = np.log( SumByGroup(t,np.exp(f+alpha_p*P),shrink=1) ).flatten()
    return deltait
    

#%%
#Calculate AR1 process
def Calc_AR1(deltait):
    lr = LinearRegression(fit_intercept=True)
    lr.fit(deltait[:-1].reshape([-1,1]),deltait[1:].reshape([-1,1]))
    gamma0 = lr.intercept_
    gamma1 = lr.coef_        
    res = deltait[1:] - lr.predict(deltait[:-1].reshape([-1,1])).flatten()
    var_nu = np.var(res)
    return gamma0,gamma1,var_nu

#%%
def gen_deltait_next(deltait_seq): #input a sequence of delta [delta1 delta2 ...], return [delta's for delta1, delta's for delta2,...]
    deltait_n=deltait_seq.size
    deltait_next_stuck=(gamma0+gamma1*deltait_seq).repeat(N_draw_E)+np.tile(delta_draw*np.sqrt(var_nu),deltait_n) #as delta_n*N_draw_E vector.
    deltait_next=deltait_next_stuck.reshape(deltait_n,N_draw_E) # as delta_n by N_draw_E matrix.        
    return deltait_next,deltait_next_stuck

def Calc_EV(V,deltait_seq):
    deltait_next,_ = gen_deltait_next(deltait_seq)
    #V_next = interpolate.griddata(dgrid, V, delta_next  )
    grid_interp = interpolate.UnivariateSpline(deltait_seq,V)
    V_next = grid_interp(deltait_next  )
    ev = np.mean(V_next,axis=1).flatten() #return from delta_seq to EV for each delta
    return ev

#%%
def loop_val(V_init):
    V_old = copy.deepcopy(V_init)
    norm = tol_value_iter+1000.
    while norm>tol_value_iter:
        EV_on_grid = Calc_EV(V_old, dgrid)
        V_new = np.log( np.exp(dgrid)+np.exp(beta*EV_on_grid) )
        norm = np.max(np.abs(V_new-V_old))  
        V_old = copy.deepcopy(V_new)
    return V_new

#%%
def V_delta_poly(dgrid,V):
    coef=np.polyfit(dgrid,V,deg=2)
    return coef

def Calc_share_given_delta(deltait,delta_ijt,deltait,V):
    ev = Calc_EV(V,deltait)
    share_denom = (np.exp(deltait)+np.exp(beta*ev) )[t]
    share_num = np.exp(delta_ijt)
    share = share_num/share_denom
    return share

def Calc_FOC_error(P,V_init=copy.deepcopy(dgrid)):
    deltait=Calc_delta(P,lam)
    gamma0,gamma1,var_nu=Calc_AR1(deltait)
    V = loop_val(V_init)
    V_coef=V_delta_poly(dgrid,V)
    foc_resid = 
    return foc
    
    


if __name__=="__main__":
    deltait=Calc_delta(P,lam)
    gamma0,gamma1,var_nu=Calc_AR1(deltait)
    V_init=copy.deepcopy(dgrid)
    V = loop_val(V_init)
    