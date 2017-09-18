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


#Dynamic Demand Simulation
#Assuming one time purchase, therefore the state takes delta only, not f.
#Assuming homogeneous consumers.
#Assuming the number of prducts doesn't change.
#%%
class DynamicDemandPricing_Simulate:
    def __init__(self,alpha_j,alpha_p,alpha_cost_j,flag_RC=False):
        
        self.N_prod=3 
        self.N_period=10 
        self.N_obs=self.N_prod*self.N_period
        self.N_prod_t=np.ones(self.N_period)*self.N_prod #number of produtcts in each t        
        self.periodid = np.repeat(np.arange(self.N_period),self.N_prod)
        self.prodid=np.tile(np.arange(self.N_prod),self.N_period)
        #Draw random values
        self.xi = np.random.randn(self.N_obs)
        self.lam = np.random.randn(self.N_obs)
        
        #make grids
        self.mgrid_n=22
        self.mgrid_max=1.
        self.mgrid_min=0. 
        self.mgrid=np.linspace(self.mgrid_min,self.mgrid_max,num=self.mgrid_n)
        self.mm_interp = interpolate.interp1d(self.mgrid, self.mgrid, kind='nearest', bounds_error=False, fill_value=(self.mgrid_min,self.mgrid_max))
        
        self.lgrid_n=26
        self.lgrid_max=np.max(self.lam)
        self.lgrid_min=np.min(self.lam) 
        self.lgrid=np.linspace(self.lgrid_min,self.lgrid_max,num=self.lgrid_n)
        self.ll_interp = interpolate.interp1d(self.lgrid, self.lgrid, kind='nearest', bounds_error=False, fill_value=(self.lgrid_min,self.lgrid_max))
        
        self.xgrid_n=24
        self.xgrid_max=np.max(self.xi)
        self.xgrid_min=np.min(self.lam)
        self.xgrid=np.linspace(self.xgrid_min,self.xgrid_max,num=self.xgrid_n)
        self.xx_interp = interpolate.interp1d(self.xgrid, self.xgrid, kind='nearest', bounds_error=False, fill_value=(self.xgrid_min,self.xgrid_max))
        
        ##price and delta grid will change dynamically.
        self.pgrid_n=20
        self.pgrid_max=5.
        self.pgrid_min=0. 
        self.pgrid=np.linspace(self.pgrid_min,self.pgrid_max,num=self.pgrid_n)
        self.pp_interp = interpolate.interp1d(self.pgrid, self.pgrid, kind='nearest', bounds_error=False, fill_value=(self.pgrid_min,self.pgrid_max))

        self.dgrid_n=28
        self.dgrid_max=10.
        self.dgrid_min=-10. 
        self.dgrid=np.linspace(self.dgrid_min,self.dgrid_max,num=self.dgrid_n)
        self.dd_interp = interpolate.interp1d(self.dgrid, self.dgrid, kind='nearest', bounds_error=False, fill_value=(self.dgrid_min,self.dgrid_max) )
                
        #Set true parameters
        self.alpha_j = alpha_j
        self.alpha_p = alpha_p
        self.alpha_cost_j = alpha_cost_j
        
        #Set hyperparameters
        self.N_draw_E=7 # draw for Expectation on delta and lambda
        self.beta = .8        
        self.tol_value_iter=.1
        self.tol_price_iter=.1
        self.value_maxiter = 500
        self.price_maxiter = 500
        self.delta_draw=np.random.randn(self.N_draw_E)
        self.lam_draw=np.random.randn(self.N_draw_E)
        self.xi_draw=np.random.randn(self.N_draw_E)
        #delta_draw=Halton_draw.halton_randn(1,N_draw_E)[:,0]
        self.Monopoly = True #only True is allowed in current version.
        
        self.flag_RC = flag_RC
        self.N_Cons = 2
        self.flag_warning = 0
        self.warning = []

    def SumByGroup(self,groupid,x,shrink=0):
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

    #----------------- Demand Side Functions (assuming ologopoly)------------------
    def Calc_deltait(self,p,xi,alpha_j,alpha_p=None):
        if alpha_p is None:
            alpha_p = self.alpha_p
        f = alpha_j[self.prodid]  +xi
        #f = alpha_j  +xi    
        delta_ijt = f+alpha_p*p
        if self.Monopoly is False:
            deltait = np.log( self.SumByGroup(self.periodid,np.exp(delta_ijt),shrink=1) ).flatten()
        elif self.Monopoly is True:
            deltait = delta_ijt
        if np.any(np.isinf(np.exp(deltait))):
            self.flag_warning=1
            self.warning.append("exp(delta) infinity in Calc_deltait.")        
        return deltait
    
    #Calculate AR1 process
    def Calc_AR1(self,deltait):
        lr = LinearRegression(fit_intercept=True)
        if len(deltait)==self.N_obs:
            x = deltait[np.where(self.periodid!=(self.N_period-1) )].reshape([-1,1])
            y = deltait[np.where(self.periodid!=0)].reshape([-1,1])
        elif len(deltait)==self.N_period:
            x = deltait[:-1].reshape([-1,1])
            y = deltait[1:].reshape([-1,1])
        lr.fit(x, y)
        gamma0 = lr.intercept_
        gamma1 = lr.coef_        
        res = x.flatten() - lr.predict(y).flatten()
        var_nu = np.var(res)
        return gamma0,gamma1,var_nu

    def gen_deltait_next(self,deltait_seq,gamma0,gamma1,var_nu): #input a sequence of delta [delta1 delta2 ...], return [delta's for delta1, delta's for delta2,...]
        deltait_n=deltait_seq.size
        deltait_next_stuck=(gamma0+gamma1*deltait_seq).repeat(self.N_draw_E)+np.tile(self.delta_draw*np.sqrt(var_nu),deltait_n) #as delta_n*N_draw_E vector.
        deltait_next=deltait_next_stuck.reshape(deltait_n,self.N_draw_E) # as delta_n by N_draw_E matrix.        
        return deltait_next, deltait_next_stuck

    def Calc_EV(self,V,deltait_seq,gamma0,gamma1,var_nu):
        deltait_next,_ = self.gen_deltait_next(deltait_seq,gamma0,gamma1,var_nu)
        #V_next = interpolate.griddata(dgrid, V, delta_next  )
        self.Calc_EV_V = V
        grid_interp = interpolate.UnivariateSpline(self.dgrid,V)
    
        V_next = grid_interp(deltait_next  )
        ev = np.mean(V_next,axis=1).flatten() #return from delta_seq to EV for each delta
        return ev

    def Calc_loop_V(self,V_init,gamma0,gamma1,var_nu):
        V_old = copy.deepcopy(V_init)
        norm = self.tol_value_iter+1000.
        i=0
        while norm>self.tol_value_iter and i<self.value_maxiter:
            EV_on_grid = self.Calc_EV(V_old, self.dgrid,gamma0,gamma1,var_nu)
            V_new = np.log( np.exp(self.dgrid)+np.exp(self.beta*EV_on_grid) )
            norm = np.max(np.abs(V_new-V_old))  
            V_old = copy.deepcopy(V_new)
            i=i+1
        if norm>self.tol_value_iter:
            print('V loop not converged after %i interations'%i)
            self.flag_warning=1
            self.warning.append("V not converged")
            sys.exit()
            
        #For debug
        if True:
            self.loop_V_EV_on_grid=EV_on_grid
            self.loop_V_V_new=V_new
            self.loop_V_V_old=V_old
            self.loop_V_i=i
            self.loop_V_norm=norm
    
        return V_new

    #------------Supply Side Function-------------------    
    def Calc_V_delta_poly(self,V):
        coef=np.polyfit(self.dgrid,V,deg=2)
        return coef

    def Calc_share_given_delta_and_V(self,deltait,delta_ijt,V):
        ev = self.Calc_EV(V,deltait)
        share_denom = (np.exp(deltait)+np.exp(self.beta*ev) )[self.periodid]
        share_num = np.exp(delta_ijt)
        share = share_num/share_denom
        return share

    def Calc_share_from_pxgrid_given_V(self,V,alpha_j,alpha_p,gamma0,gamma1,var_nu):
        if self.Monopoly==True:
            pp,xx = np.meshgrid(self.pgrid,self.xgrid,indexing="ij")        
            f = alpha_j +xx     
            delta_ijt = f+alpha_p*pp
            deltait =  delta_ijt 
            deltait_flat = deltait.flatten()
            ev_flat = self.Calc_EV(V,deltait_flat,gamma0,gamma1,var_nu)
            ev = ev_flat.reshape(deltait.shape)
            share_denom = (np.exp(deltait)+np.exp(self.beta*ev) )
            share_num = np.exp(delta_ijt)
            share = share_num/share_denom
            
            #For debug
            if True:
                self.Calc_share_delta_ijt=delta_ijt
                self.Calc_share_deltait=deltait
                self.Calc_share_deltait_flat=deltait_flat
                self.Calc_share_ev_flat=ev_flat
                self.Calc_share_ev=ev
                self.Calc_share_share_denom=share_denom
                self.Calc_share_share_num=share_num
                self.Calc_share_share=share
            
            return share #pgrid_n by xgrid_n
        

    def Calc_EW(self,W,Mnext_seq):
        lam_next = self.lam_draw
        xi_next = self.xi_draw
        #grid_interp = interpolate.interp2d(mgrid,lgrid,W,kind="cubic")
        if True:
            self.Calc_EW_W = W
        
        if self.flag_RC is False:
            W_interp = interpolate.RegularGridInterpolator( (self.mgrid,self.xgrid,self.lgrid), W, method='linear',bounds_error=False, fill_value=None)
            mm,ll,xx = np.meshgrid( Mnext_seq, lam_next, xi_next, indexing="ij" )
            W_next = W_interp( (mm,xx,ll) )
            ew = np.mean(W_next,axis=(1,2) ).flatten() #return from Mnext_seq to EV for each M
        if self.flag_RC is True:
            #Mnext_seq is assumed to be n by N_Cons matrix
            #Temporary: N_Cons=2 assumed.
            W_interp = interpolate.RegularGridInterpolator( (self.mgrid,self.mgrid,self.xgrid,self.lgrid), W, method='linear',bounds_error=False, fill_value=None )
            mm0,mm1,ll,xx = np.meshgrid( Mnext_seq[:,0].flatten(),Mnext_seq[:,1].flatten(), lam_next, xi_next, indexing="ij" )
            W_next = W_interp( (mm0,mm1,xx,ll) )
            ew = np.mean(W_next,axis=(2,3) ) #return from Mnext_seq to EV for each M
            mm = np.c_[mm0,mm1]
            
        #For debug
        if True:
            self.Calc_EW_W = W
            self.Calc_EW_Mnext_seq = Mnext_seq
            self.Calc_EW_lam_next = lam_next
            self.Calc_EW_xi_next = lam_next
            self.Calc_EW_W_interp =W_interp
            self.Calc_EW_mm,self.Calc_EW_ll,self.Calc_EW_xx = mm,ll,xx
            self.Calc_EW_W_next = W_next
            self.Calc_EW_ew = ew
        return ew



    def Calc_loop_W(self,W_init,V,alpha_j,alpha_cost_j,gamma0,gamma1,var_nu):
        if self.flag_RC is False:
            p_on_pmxlgrid = np.repeat( np.repeat( np.repeat( self.pgrid[:,np.newaxis],self.mgrid_n,axis=1 )[:,:,np.newaxis], self.xgrid_n, axis=2)[:,:,:,np.newaxis], self.lgrid_n,axis=3)    
            l_on_pmxlgrid = np.repeat( np.repeat( np.repeat( self.lgrid[np.newaxis,:],self.xgrid_n,axis=0 )[np.newaxis,:,:], self.mgrid_n, axis=0)[np.newaxis,:,:,:], self.pgrid_n,axis=0)
            m_on_pmxgrid = np.repeat( np.repeat( self.mgrid[np.newaxis,:],self.pgrid_n,axis=0 )[:,:,np.newaxis], self.xgrid_n, axis=2)
            m_on_pmxlgrid = np.repeat( m_on_pmxgrid[:,:,:,np.newaxis], self.lgrid_n, axis=3 )
            share_on_pxgrid = self.Calc_share_from_pxgrid_given_V(V,alpha_j,self.alpha_p,gamma0,gamma1,var_nu)
            share_on_pmxgrid = np.repeat(share_on_pxgrid[:,np.newaxis,:], self.mgrid_n, axis=1)
            share_on_pmxlgrid = np.repeat(share_on_pmxgrid[:,:,:,np.newaxis], self.lgrid_n, axis=3)
            prof_on_pmxlgrid = m_on_pmxlgrid * share_on_pmxlgrid * ( p_on_pmxlgrid - alpha_cost_j - l_on_pmxlgrid )
        if self.flag_RC is True:
            p_on_pmxlgrid = np.repeat( np.repeat( np.repeat( self.pgrid[:,np.newaxis],self.mgrid_n,axis=1 )[:,:,np.newaxis], self.xgrid_n, axis=2)[:,:,:,np.newaxis], self.lgrid_n,axis=3)    
            l_on_pmxlgrid = np.repeat( np.repeat( np.repeat( self.lgrid[np.newaxis,:],self.xgrid_n,axis=0 )[np.newaxis,:,:], self.mgrid_n, axis=0)[np.newaxis,:,:,:], self.pgrid_n,axis=0)
            p_on_pmmxlgrid =  np.repeat(p_on_pmxlgrid[:,np.newaxis,:,:,:],self.mgrid_n,axis=1)  
            l_on_pmmxlgrid =  np.repeat(l_on_pmxlgrid[:,np.newaxis,:,:,:],self.mgrid_n,axis=1) 
            m_on_pmxgrid = np.repeat( np.repeat( self.mgrid[np.newaxis,:],self.pgrid_n,axis=0 )[:,:,np.newaxis], self.xgrid_n, axis=2)
            m0_on_pmmxgrid = np.repeat( m_on_pmxgrid[:,np.newaxis,:,:], self.mgrid_n, axis=1 )
            m0_on_pmmxlgrid = np.repeat( m0_on_pmmxgrid[:,:,:,:,np.newaxis], self.lgrid_n, axis=4 )
            m1_on_pmmxgrid = np.repeat( m_on_pmxgrid[:,:,np.newaxis,:], self.mgrid_n, axis=2 )
            m1_on_pmmxlgrid = np.repeat( m1_on_pmmxgrid[:,:,:,:,np.newaxis], self.lgrid_n, axis=4 )
            V0 = V[:,0]
            V1 = V[:,1]
            share0_on_pxgrid = self.Calc_share_from_pxgrid_given_V(V0,alpha_j,self.alpha_p[0],gamma0[0],gamma1[0],var_nu[0])
            share0_on_pmxgrid = np.repeat(share0_on_pxgrid[:,np.newaxis,:], self.mgrid_n, axis=1)
            share0_on_pmmxgrid = np.repeat(share0_on_pmxgrid[:,np.newaxis,:,:], self.mgrid_n, axis=1) #p-m0-m1-x grid
            share0_on_pmmxlgrid = np.repeat(share0_on_pmmxgrid[:,:,:,:,np.newaxis], self.lgrid_n, axis=4) #p-m0-m1-x-l grid
            share1_on_pxgrid = self.Calc_share_from_pxgrid_given_V(V1,alpha_j,alpha_p[1],gamma0[1],gamma1[1],var_nu[1])
            share1_on_pmxgrid = np.repeat(share1_on_pxgrid[:,np.newaxis,:], self.mgrid_n, axis=1)
            share1_on_pmmxgrid = np.repeat(share1_on_pmxgrid[:,:,np.newaxis,:], self.mgrid_n, axis=2)
            share1_on_pmmxlgrid = np.repeat(share1_on_pmmxgrid[:,:,:,:,np.newaxis], self.lgrid_n, axis=4) #p-m0-m1-x-l grid
            
            prof_on_pmmxlgrid = (m0_on_pmmxlgrid * share0_on_pmmxlgrid + m1_on_pmmxlgrid * share1_on_pmmxlgrid ) * ( p_on_pmmxlgrid - alpha_cost_j - l_on_pmmxlgrid )
            self.prof_on_pmmxlgrid=prof_on_pmmxlgrid

        if True:#for debug
            #self.loop_W_share_on_pxgrid=share_on_pxgrid
            #self.loop_W_share_on_pmxgrid=share_on_pmxgrid
            #self.loop_W_share_on_pmxlgrid=share_on_pmxlgrid
            self.loop_W_p_on_pmxlgrid=p_on_pmxlgrid
            self.loop_W_l_on_pmxlgrid=l_on_pmxlgrid
            #self.loop_W_m_on_pmxlgrid=m_on_pmxlgrid
            #self.loop_W_prof_on_pmxlgrid=prof_on_pmxlgrid

        if self.flag_RC is False:
            mnext_on_pmxgrid = m_on_pmxgrid*(1. - share_on_pmxgrid)    
            mnext_on_pmxgrid = self.mm_interp(mnext_on_pmxgrid) #move mnext on mgrid
            mnext_index_on_pmxgrid = np.searchsorted(self.mgrid, mnext_on_pmxgrid)
        if self.flag_RC is True:
            m0next_on_pmxgrid = m_on_pmxgrid*(1. - share0_on_pmxgrid)    
            m0next_on_pmxgrid = self.mm_interp(m0next_on_pmxgrid) #move mnext on mgrid
            m0next_index_on_pmxgrid = np.searchsorted(self.mgrid, m0next_on_pmxgrid)
            m0next_index_on_pmmxgrid = np.repeat(m0next_index_on_pmxgrid[:,np.newaxis,:,:],self.mgrid_n,axis=1)
            m1next_on_pmxgrid = m_on_pmxgrid*(1. - share1_on_pmxgrid)    
            m1next_on_pmxgrid = self.mm_interp(m1next_on_pmxgrid) #move mnext on mgrid
            m1next_index_on_pmxgrid = np.searchsorted(self.mgrid, m1next_on_pmxgrid)
            m1next_index_on_pmmxgrid = np.repeat(m1next_index_on_pmxgrid[:,:,np.newaxis,:],self.mgrid_n,axis=2)

            mnext_on_pmxgrid = np.c_[m0next_on_pmxgrid,m1next_on_pmxgrid]
            mnext_index_on_pmxgrid = np.c_[m0next_index_on_pmxgrid,m1next_index_on_pmxgrid]
        W_old = copy.deepcopy(W_init)
        norm = self.tol_value_iter+1000.    
        i=0
        while norm>self.tol_value_iter and i<self.value_maxiter:
            if self.flag_RC is False:
                EW_on_mnextgrid = self.Calc_EW(W_old,self.mgrid)
                self.EW_on_mnextgrid=EW_on_mnextgrid
                if np.any(np.isnan( EW_on_mnextgrid )):
                    self.W_old = W_old
                    self.flag_warning=1
                    self.warning.append("EW has nan")
                    sys.exit('EW has nan')
                EW_on_pmxgrid = EW_on_mnextgrid[mnext_index_on_pmxgrid]
                EW_on_pmxlgrid = np.repeat( EW_on_pmxgrid[:,:,:,np.newaxis],self.lgrid_n,axis=3 )
                value = prof_on_pmxlgrid+EW_on_pmxlgrid
                W_new = np.max(value, axis=0)
                price_opt_on_mxlgrid = self.pgrid[ np.argmax(value, axis=0) ]
                
            if self.flag_RC is True:
                mnext_grid = np.c_[self.mgrid,self.mgrid]
                EW_on_mmnextgrid = self.Calc_EW(W_old,mnext_grid) #mgrid by mgrid
                self.EW_on_mmnextgrid=EW_on_mmnextgrid
                self.m0next_index_on_pmmxgrid=m0next_index_on_pmmxgrid
                self.m1next_index_on_pmmxgrid=m1next_index_on_pmmxgrid
                EW_on_pmmxgrid = EW_on_mmnextgrid[m0next_index_on_pmmxgrid,m1next_index_on_pmmxgrid]
                self.EW_on_pmmxgrid=EW_on_pmmxgrid
                EW_on_pmmxlgrid = np.repeat( EW_on_pmmxgrid[:,:,:,:,np.newaxis],self.lgrid_n,axis=4 )
                value = prof_on_pmmxlgrid+EW_on_pmmxlgrid
                W_new = np.max(value, axis=0)
                price_opt_on_mmxlgrid = self.pgrid[ np.argmax(value, axis=0) ]
            
            norm = np.max( np.abs(W_new-W_old) )        
            W_old = copy.deepcopy(W_new)
            i=i+1
                
        if norm>self.tol_value_iter:
            print('W loop not converged after %i interations'%i)
            self.flag_warning=1
            self.warning.append("W not converged")
            sys.exit()

        #for debug
        if False:
            self.loop_W_prof_on_pmxlgrid=prof_on_pmxlgrid
            self.loop_W_mnext_on_pmxgrid=mnext_on_pmxgrid
            self.loop_W_mnext_index_on_pmxgrid=mnext_index_on_pmxgrid
            self.loop_W_EW_on_pmxlgrid=EW_on_pmxlgrid
            self.loop_W_price_opt_on_mxlgrid=price_opt_on_mxlgrid
            self.loop_W_price_opt_on_mmxlgrid=price_opt_on_mmxlgrid
            self.loop_W_norm=norm
            self.loop_W_i=i
            self.loop_W_W=W_new

        if self.flag_RC is False:            
            return W_new, price_opt_on_mxlgrid, mnext_on_pmxgrid
        if self.flag_RC is True:            
            return W_new, price_opt_on_mmxlgrid, m0next_on_pmxgrid, m1next_on_pmxgrid
    

    def Calc_optprice_given_V(self,V,W_init,alpha_j,alpha_cost_j,gamma0,gamma1,var_nu):
        price_opt_t = np.zeros(self.N_period)
        if self.flag_RC is False:
            W, price_opt_on_mxlgrid, mnext_on_pmxgrid = self.Calc_loop_W(W_init,V,alpha_j,alpha_cost_j,gamma0,gamma1,var_nu)
            M_t = np.zeros(self.N_period)
            M_t[0] = self.mgrid_max        
            M_t_index =np.zeros(self.N_period).astype(int)
            M_t_index[0] = self.mgrid_n-1
            price_opt_t_index = np.zeros(self.N_period).astype(int)        
            for t in range(self.N_period-1):
                price_opt_t[t] = price_opt_on_mxlgrid[M_t_index[t], self.xi_index[t], self.lam_index[t]]
                price_opt_t_index[t] = np.searchsorted(self.pgrid, price_opt_t[t])
                M_t[t+1] = mnext_on_pmxgrid[price_opt_t_index[t], M_t_index[t], self.xi_index[t]]
                M_t_index[t+1] = np.searchsorted(self.mgrid, M_t[t+1])
            price_opt_t[self.N_period-1] = price_opt_on_mxlgrid[M_t_index[self.N_period-2], self.xi_index[self.N_period-2], self.lam_index[self.N_period-2]]        

        if self.flag_RC is True:
            W, price_opt_on_mxlgrid, m0next_on_pmxgrid, m1next_on_pmxgrid = self.Calc_loop_W(W_init,V,alpha_j,alpha_cost_j,gamma0,gamma1,var_nu)
            M_t = np.zeros([self.N_period,2])
            M_t[0,:] = self.mgrid_max        
            M_t_index =np.zeros([self.N_period,2]).astype(int)
            M_t_index[0,:] = self.mgrid_n-1
            price_opt_t_index = np.zeros(self.N_period).astype(int)        
            for t in range(self.N_period-1):
                price_opt_t[t] = price_opt_on_mxlgrid[M_t_index[t,0],M_t_index[t,1], self.xi_index[t], self.lam_index[t]]
                price_opt_t_index[t] = np.searchsorted(self.pgrid, price_opt_t[t])
                
                self.M_t=M_t
                self.M_t_index=M_t_index
                self.price_opt_t=price_opt_t
                self.price_opt_t_index=price_opt_t_index
                self.m0next_on_pmxgrid=m0next_on_pmxgrid
                self.m0next_on_pmxgrid=m0next_on_pmxgrid
                
                M_t[t+1,0] = m0next_on_pmxgrid[price_opt_t_index[t], M_t_index[t,0], self.xi_index[t]]
                M_t[t+1,1] = m1next_on_pmxgrid[price_opt_t_index[t], M_t_index[t,1], self.xi_index[t]]
                M_t_index[t+1,0] = np.searchsorted(self.mgrid, M_t[t+1,0])
                M_t_index[t+1,1] = np.searchsorted(self.mgrid, M_t[t+1,1])
            price_opt_t[self.N_period-1] = price_opt_on_mxlgrid[M_t_index[self.N_period-2,0],M_t_index[self.N_period-2,1], self.xi_index[self.N_period-2], self.lam_index[self.N_period-2]]        

        return price_opt_t,W


    
    def Sim_Price_Sales(self):    
        self.xi_on_xgrid = self.xx_interp(self.xi)
        self.lam_on_lgrid = self.ll_interp(self.lam)
        self.xi_index = np.searchsorted(self.xgrid, self.xi_on_xgrid)
        self.lam_index = np.searchsorted(self.lgrid, self.lam_on_lgrid) 
    
        p_init = self.pgrid_max*.7#+np.abs( np.random.randn(self.N_obs) )
        if self.flag_RC is False:
            V_init = np.zeros(self.dgrid_n)
            W_init = np.zeros([self.mgrid_n, self.xgrid_n, self.lgrid_n])    
        if self.flag_RC is True:
            V_init = np.zeros([self.dgrid_n,2])
            W_init = np.zeros([self.mgrid_n, self.mgrid_n, self.xgrid_n, self.lgrid_n])    
        p_old = copy.deepcopy(p_init)
        V_old = copy.deepcopy(V_init)
        W_old = copy.deepcopy(W_init)
        norm = self.tol_price_iter+1000.
        #Loop
        i = 0
        while norm>self.tol_price_iter and i<self.price_maxiter:
            if self.flag_RC is False: 
                deltait = self.Calc_deltait(p_old,self.xi,self.alpha_j)
                gamma0,gamma1,var_nu = self.Calc_AR1(deltait)
            if self.flag_RC is True: 
                delta0t = self.Calc_deltait(p_old,self.xi,self.alpha_j,self.alpha_p[0])
                gamma0_cons0,gamma1_cons0,var_nu_cons0 = self.Calc_AR1(delta0t)
                delta1t = self.Calc_deltait(p_old,self.xi,self.alpha_j,self.alpha_p[1])
                gamma0_cons1,gamma1_cons1,var_nu_cons1 = self.Calc_AR1(delta1t)
                deltait = np.c_[delta0t,delta1t]
                gamma0,gamma1,var_nu = np.array([gamma0_cons0,gamma0_cons1]),np.array([gamma1_cons0,gamma1_cons1]),np.array([var_nu_cons0,var_nu_cons1])
            #Redefine delta and price grid
            self.pgrid_max= np.max( [np.max(p_old)*1.2, 20.])
            self.pgrid=np.linspace(self.pgrid_min,self.pgrid_max,num=self.pgrid_n)
            self.pp_interp = interpolate.interp1d(self.pgrid, self.pgrid, kind='nearest', bounds_error=False, fill_value=(self.pgrid_min,self.pgrid_max))
            self.dgrid_max=np.max(deltait)*( (np.max(deltait)>=0)*1.2  +  (np.max(deltait)<0)*.8 )
            self.dgrid_min=np.min(deltait) * ( (np.min(deltait)>=0)*.8  +  (np.min(deltait)<0)*1.2 )
            self.dgrid=np.linspace(self.dgrid_min,self.dgrid_max,num=self.dgrid_n)
            self.dd_interp = interpolate.interp1d(self.dgrid, self.dgrid, kind='nearest', bounds_error=False, fill_value=(self.dgrid_min,self.dgrid_max) )
            
            if self.flag_RC is False:
                V_new = self.Calc_loop_V(V_init=V_old,gamma0=gamma0,gamma1=gamma1,var_nu=var_nu)
            if self.flag_RC is True:
                V_new = np.zeros([self.dgrid_n,2])
                V_new[:,0] = self.Calc_loop_V(V_init=V_old[:,0],gamma0=gamma0[0],gamma1=gamma1[0],var_nu=var_nu[0])
                V_new[:,1] = self.Calc_loop_V(V_init=V_old[:,1],gamma0=gamma0[1],gamma1=gamma1[1],var_nu=var_nu[1])
            p_new = np.zeros(self.N_obs)
            for j in range(self.N_prod):
                p_new[self.prodid==j],W_new = self.Calc_optprice_given_V(V_new,W_old,self.alpha_j[j],self.alpha_cost_j[j],gamma0,gamma1,var_nu)
            norm = np.max(np.abs(p_new-p_old ))
            p_old=copy.deepcopy(p_new)
            V_old=copy.deepcopy(V_new)
            W_old=copy.deepcopy(W_new)
            i=i+1
            if i%10==0:
                print('Outside loop %i th iteration '%i)
        if norm>self.tol_price_iter:
            print('Outside loop not converged after %i interations'%i)
            sys.exit()
            print('Gammas:',gamma0,',',gamma1)
            self.warning=1
            
            
        #Price
        self.price_simulated = p_new
        price_simulated_index = np.searchsorted(self.pgrid, p_new)
        
        #for debug
        if True:
            self.i_outloop = i
            self.gamma0=gamma0
            self.gamma1=gamma1
            self.var_nu=var_nu
            self.Sim_price_price_simulated_index=price_simulated_index        
            self.Sim_price_V_new=V_new
            self.Sim_price_norm=norm
            self.Sim_price_W_new=W_new

        #Share and sales        
        if self.flag_RC is False:
            self.share_simulated = np.zeros(self.N_obs)
            self.M_t_simulated = np.zeros(self.N_obs)
            self.M_t_simulated[self.periodid==0] = 1.
        if self.flag_RC is True:
            self.share_simulated = np.zeros([self.N_obs,2])
            self.M_t_simulated = np.zeros([self.N_obs,2])
            self.M_t_simulated[self.periodid==0,:] = 1./self.N_Cons
        
        self.sales_simulated = np.zeros(self.N_obs)
        for j in range(self.N_prod):
            if self.flag_RC is False:
                share_on_pxgrid_j = self.Calc_share_from_pxgrid_given_V(V_new,self.alpha_j[j],self.alpha_p,gamma0,gamma1,var_nu)
                self.Sim_price_share_on_pxgrid_j=share_on_pxgrid_j
                self.share_simulated[self.prodid==j] = share_on_pxgrid_j[price_simulated_index[self.prodid==j], self.xi_index[self.prodid==j]]
            if self.flag_RC is True:
                share0_on_pxgrid_j = self.Calc_share_from_pxgrid_given_V(V_new[:,0],self.alpha_j[j],self.alpha_p[0],gamma0[0],gamma1[0],var_nu[0])
                share1_on_pxgrid_j = self.Calc_share_from_pxgrid_given_V(V_new[:,1],self.alpha_j[j],self.alpha_p[1],gamma0[1],gamma1[1],var_nu[1])
                self.share_simulated[self.prodid==j,0] = share0_on_pxgrid_j[price_simulated_index[self.prodid==j], self.xi_index[self.prodid==j]]
                self.share_simulated[self.prodid==j,1] = share1_on_pxgrid_j[price_simulated_index[self.prodid==j], self.xi_index[self.prodid==j]]
                for t in range(self.N_period-1):
                    self.M_t_simulated[( (self.prodid==j)*(self.periodid==t+1) ),0] = self.M_t_simulated[( (self.prodid==j)*(self.periodid==t) ),0] * (1.-self.share_simulated[( (self.prodid==j)*(self.periodid==t) ),0])
                    self.M_t_simulated[( (self.prodid==j)*(self.periodid==t+1) ),1] = self.M_t_simulated[( (self.prodid==j)*(self.periodid==t) ),1] * (1.-self.share_simulated[( (self.prodid==j)*(self.periodid==t) ),1])

        if self.flag_RC is False:        
            self.sales_simulated = self.M_t_simulated * self.share_simulated
        if self.flag_RC is True:        
            self.sales_simulated_temp = self.M_t_simulated * self.share_simulated
            self.sales_simulated = np.sum(self.sales_simulated_temp, axis=1).flatten()
    
    


if __name__=="__main__": 
    alpha_j = np.array([5.,3.,10.])
    alpha_p = np.array([-2.,-7.])
    alpha_cost_j = np.array([4.,4.,4.])
    flag_RC=True
    sim = DynamicDemandPricing_Simulate(alpha_j = alpha_j,alpha_p = alpha_p,alpha_cost_j = alpha_cost_j,flag_RC=flag_RC)
    sim.Sim_Price_Sales()

    print("price_simulated:",sim.price_simulated)
    print("share_simulated:",sim.share_simulated)









    