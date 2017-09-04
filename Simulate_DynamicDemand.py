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
dd_interp = interpolate.interp1d(dgrid, dgrid, kind='nearest')

N_draw_E=7 # draw for Expectation on delta and lambda
#---temp for coding---
N_prod=1 
N_period=10 
N_obs=N_prod*N_period
N_prod_t=np.ones(N_period)*N_prod #number of produtcts in each t        
periodid = np.repeat(np.arange(N_period),N_prod)
prodid=np.tile(np.arange(N_prod),N_period)

#%%Pricing Problem
#make grids
mgrid_n=20
mgrid_max=1.
mgrid_min=0. 
mgrid=np.linspace(mgrid_min,mgrid_max,num=mgrid_n)
mm_interp = interpolate.interp1d(mgrid, mgrid, kind='nearest')

lgrid_n=20
lgrid_max=3.
lgrid_min=-3. 
lgrid=np.linspace(lgrid_min,lgrid_max,num=lgrid_n)
ll_interp = interpolate.interp1d(lgrid, lgrid, kind='nearest')

xgrid_n=20
xgrid_max=3.
xgrid_min=-3. 
xgrid=np.linspace(xgrid_min,xgrid_max,num=xgrid_n)
xx_interp = interpolate.interp1d(xgrid, xgrid, kind='nearest')

pgrid_n=20
pgrid_max=5.
pgrid_min=0. 
pgrid=np.linspace(lgrid_min,lgrid_max,num=lgrid_n)


#%%
#alpha_j = np.array([5.,3.,10.])
alpha_j = np.array([10.]) #For now assume the quality is constant
alpha_p = -5.
alpha_cost = 3.
p = np.random.randn(N_obs)
xi = np.random.randn(N_obs)
lam = np.random.randn(N_obs)
mc = alpha_cost+lam

beta = .8        
tol_value_iter=.1
tol_price_iter=.1
delta_draw=np.random.randn(N_draw_E)
lam_draw=np.random.randn(N_draw_E)
xi_draw=np.random.randn(N_draw_E)
#delta_draw=Halton_draw.halton_randn(1,N_draw_E)[:,0]       


#%% Demand Side Functions (assuming ologopoly)
def Calc_deltait(p,xi,Monopoly=True):
    #f = alpha_j[prodid]  +xi
    f = alpha_j  +xi    
    delta_ijt = f+alpha_p*p
    if Monopoly==False:
        deltait = np.log( SumByGroup(periodid,np.exp(delta_ijt),shrink=1) ).flatten()
    elif Monopoly==True:
        deltait = delta_ijt
    return deltait
    
#Calculate AR1 process
def Calc_AR1(deltait):
    lr = LinearRegression(fit_intercept=True)
    if len(deltait)==N_obs:
        x = deltait[np.where(periodid!=N_period)].reshape([-1,1])
        y = deltait[np.where(periodid!=0)].reshape([-1,1])
    elif len(deltait)==N_period:
        x = deltait[:-1].reshape([-1,1])
        y = deltait[1:].reshape([-1,1])
    lr.fit(x, y)
    gamma0 = lr.intercept_
    gamma1 = lr.coef_        
    res = x.flatten() - lr.predict(y).flatten()
    var_nu = np.var(res)
    return gamma0,gamma1,var_nu

def gen_deltait_next(deltait_seq): #input a sequence of delta [delta1 delta2 ...], return [delta's for delta1, delta's for delta2,...]
    deltait_n=deltait_seq.size
    deltait_next_stuck=(gamma0+gamma1*deltait_seq).repeat(N_draw_E)+np.tile(delta_draw*np.sqrt(var_nu),deltait_n) #as delta_n*N_draw_E vector.
    deltait_next=deltait_next_stuck.reshape(deltait_n,N_draw_E) # as delta_n by N_draw_E matrix.        
    return deltait_next,deltait_next_stuck

def Calc_EV(V,deltait_seq):
    deltait_next,_ = gen_deltait_next(deltait_seq)
    #V_next = interpolate.griddata(dgrid, V, delta_next  )
    grid_interp = interpolate.UnivariateSpline(dgrid,V)
    V_next = grid_interp(deltait_next  )
    ev = np.mean(V_next,axis=1).flatten() #return from delta_seq to EV for each delta
    return ev

def Calc_loop_V(V_init):
    V_old = copy.deepcopy(V_init)
    norm = tol_value_iter+1000.
    while norm>tol_value_iter:
        EV_on_grid = Calc_EV(V_old, dgrid)
        V_new = np.log( np.exp(dgrid)+np.exp(beta*EV_on_grid) )
        norm = np.max(np.abs(V_new-V_old))  
        V_old = copy.deepcopy(V_new)
    return V_new



#%% Supply Side Function    

def Calc_V_delta_poly(dgrid,V):
    coef=np.polyfit(dgrid,V,deg=2)
    return coef

def Calc_share_given_delta_and_V(deltait,delta_ijt,deltait,V):
    ev = Calc_EV(V,deltait)
    share_denom = (np.exp(deltait)+np.exp(beta*ev) )[t]
    share_num = np.exp(delta_ijt)
    share = share_num/share_denom
    return share

def Calc_share_from_pxgrid_given_V(pgrid,xgrid,V,Monopoly=True):
    if Monopoly==True:
        pp,xx = np.meshgrid(pgrid,xgrid)        
        f = alpha_j +xx        
        delta_ijt = f+alpha_p*pp
        deltait = np.log( np.exp( delta_ijt) ) 
        deltait_flat = deltait.flatten()
        ev_flat = Calc_EV(V,deltait_flat)
        ev = ev_flat.reshape(deltait.shape)
        share_denom = (np.exp(deltait)+np.exp(beta*ev) )
        share_num = np.exp(delta_ijt)
        share = share_num/share_denom
        return share #pgrid_n by xgrid_n


def Calc_EW(W,Mnext_seq):
    lam_next = lam_draw
    xi_next = xi_draw
    #grid_interp = interpolate.interp2d(mgrid,lgrid,W,kind="cubic")
    W_interp = interpolate.RegularGridInterpolator( (mgrid,lgrid,xgrid), W, method='linear' )
    mm,ll,xx = np.meshgrid( Mnext_seq, lam_next, xi_next )
    W_next = W_interp( (mm,ll,xx) )
    ew = np.mean(W_next,axis=(1,2) ).flatten() #return from Mnext_seq to EV for each M
    return ew

def Calc_loop_W(W_init,V):
    share_on_pxgrid = Calc_share_from_pxgrid_given_V(pgrid,xgrid,V)
    share_on_pmxgrid = np.repeat(share_on_pxgrid[:,np.newaxis,:], mgrid_n, axis=1)
    share_on_pmxlgrid = np.repeat(share_on_pmxgrid[:,:,:,np.newaxis], lgrid_n, axis=3)
    p_on_pmxlgrid = np.repeat( np.repeat( np.repeat( pgrid[:,np.newaxis],mgrid_n,axis=1 )[:,:,np.newaxis], xgrid_n, axis=2)[:,:,:,np.newaxis], lgrid_n,axis=3)    
    l_on_pmxlgrid = np.repeat( np.repeat( np.repeat( lgrid[np.newaxis,:],mgrid_n,axis=0 )[np.newaxis,:,:], xgrid_n, axis=0)[:,:,np.newaxis,:], xgrid_n,axis=2)
    m_on_pmxgrid = np.repeat( np.repeat( mgrid[np.newaxis,:],pgrid_n,axis=0 )[:,:,np.newaxis], xgrid_n, axis=2)
    m_on_pmxlgrid = np.repeat( m_on_pmxgrid[:,:,:,np.newaxis], lgrid_n, axis=3 )
    
    prof_on_pmxlgrid = m_on_pmxlgrid * share_on_pmxlgrid * ( p_on_pmxlgrid - alpha_cost - l_on_pmxlgrid )

    mnext_on_pmxgrid = m_on_pmxgrid*(1. - share_on_pmxgrid)    
    mnext_on_pmxgrid = mm_interp(mnext_on_pmxgrid) #move mnext on mgrid

    W_old = copy.deepcopy(W_init)
    norm = tol_value_iter+1000.    
    while norm>tol_value_iter:
        EW_on_mnextgrid = Calc_EW(W_old,mgrid)
        EW_on_pmxgrid = EW_on_mnextgrid[mnext_on_pmxgrid]
        EW_on_pmxlgrid = np.repeat( EW_on_pmxgrid[:,:,:,np.newaxis],lgrid_n,axis=3 )
        value = prof_on_pmxlgrid+EW_on_pmxlgrid
        
        W_new = np.max(value, axis=0)
        price_opt_on_mxlgrid = pgrid[ np.argmax(value, axis=0) ]
        norm = np.max( np.abs(W_new-W_old) )        
        W_old = copy.deepcopy(W_new)
                
    return W_new, price_opt_on_mxlgrid, mnext_on_pmxgrid
    

def Calc_optprice_given_V(V,W_init):
    W, price_opt_on_mxlgrid, mnext_on_pmxgrid = Calc_loop_W(W_init,V)
    price_opt_t = np.zeros(N_period)
    M_t = np.zeros(N_period)
    M_t[0] = mgrid_max
    for i in range(N_period-1):
        price_opt_t[i] = price_opt_on_mxlgrid[M_t[i], xi[i], lam[i]]
        M_t[i+1] = mnext_on_pmxgrid[price_opt_t[i], M_t[i], xi[i]]
    price_opt_t[N_period-1] = price_opt_on_mxlgrid[M_t[N_period-2], xi[N_period-2], lam[N_period-2]]
    
    return price_opt_t,W
    

def Sim_Price_Sales():
    xi = np.random.randn(N_obs)
    lam = np.random.randn(N_obs)
    mc = alpha_cost+lam

    p_init = np.random.randn(N_obs)
    V_init = np.zeros(dgrid_n)
    W_init = np.zeros([mgrid_n, xgrid_n, lgrid_n])    
    p_old = copy.deepcopy(p_init)
    V_old = copy.deepcopy(V_init)
    W_old = copy.deepcopy(W_init)
    norm = 1000.
    #Loop
    while norm>tol_price_iter:
        deltait = Calc_deltait(p_old,xi)
        gamma0,gamma1,var_nu = Calc_AR1(deltait)
        V_new = Calc_loop_V(V_init=V_old)
        p_new = np.zeros(N_obs)
        for i in range(N_prod):
            p_new[prodid==i],W_new = Calc_optprice_given_V(V_new,W_old)
        norm = np.max(np.abs(p_new-p_old ))
        p_old=copy.deepcopy(p_new)
        V_old=copy.deepcopy(V_new)
        W_old=copy.deepcopy(W_new)
    
    
    


if __name__=="__main__":
    pass
    