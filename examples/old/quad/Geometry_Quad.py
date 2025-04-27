#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:55:48 2020

@author: ion
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
mu0 = 4*np.pi*1e-7;                                                           
I = 35;                                                                    
mux = 3000*mu0;                                                            
muy = 300*mu0;                                                             
use_nonlinear_material = 0;                                               




def Line(t,ta,tb,A,B):
    P = np.outer((t-ta)/(tb-ta),B-A) + A 
    T = np.outer(t*0+1,(B-A)/(tb-ta)) 
    N = T @ np.array([[0,1],[-1,0]])
    return P, T, N

def Lines(t,Pts):
    
    ls = np.sqrt(np.sum((Pts[1:,:] - Pts[:-1,:])**2,1))
    ls = np.cumsum(ls) / np.cumsum(ls)[-1]
    ls = np.concatenate((np.array([0]),ls))
    
    P = np.zeros((t.size,2))
    T = np.zeros((t.size,2))
    N = np.zeros((t.size,2))
    
    for i in range(Pts.shape[0]-1):
        idx = np.logical_and(t>=ls[i],t<=ls[i+1])
        P[idx,:], T[idx,:], N[idx,:] = Line(t[idx],ls[i],ls[i+1],Pts[i,:],Pts[i+1,:])
        
    return P, T, N
        
class Geometry():
    
    def __init__(self,scale=1.0):
        self.scale = scale
        self.hmesh = 10e-3
        self.Nt = 24                                                                
        self.lz = 40e-3                                                             
        self.Do = 72e-3                                                            
        self.Di = 51e-3                                                            
        self.hi = 13e-3                                                             
        self.bli = 3e-3                                                             
        self.Dc = 3.27640e-2                                                           
        self.hc = 7.55176e-3                                                           
        self.ri = 20e-3                                                           
        self.ra = 18e-3                                                           
        self.blc = self.hi-self.hc                                                           
        self.rm = (self.Dc*self.Dc+self.hc*self.hc-self.ri*self.ri)/(self.Dc*np.sqrt(2)+self.hc*np.sqrt(2)-2*self.ri)                 
        self.R = self.rm-self.ri
        
    
    def L1(self,t):
    
        Pts = np.array([[self.Dc,self.hc],[self.Dc+self.blc,self.hi],[self.Di-self.bli,self.hi],[self.Di,self.hi-self.bli],[self.Di,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        
        return Lines(t,Pts)
    
    def L2(self,t):
    
        Pts = np.array([[self.Do,0],[self.Do,self.Do * np.tan(np.pi/8)],[self.Do/np.sqrt(2),self.Do/np.sqrt(2)],[self.ri/np.sqrt(2),self.ri/np.sqrt(2)]])
        Pts = Pts @ (np.eye(2) * self.scale)
        
        return Lines(t,Pts)
    
    def L3(self,t):
    
        Pts = np.array([[self.Dc,0],[self.Di,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        
        return Lines(t,Pts)
    
    def L4(self,t):
    
        Pts = np.array([[0,0],[self.ri/np.sqrt(2),self.ri/np.sqrt(2)]])
        Pts = Pts @ (np.eye(2) * self.scale)
        
        return Lines(t,Pts)
    
    def L5(self,t):
    
        Pts = np.array([[0,0],[self.Dc,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        
        return Lines(t,Pts)
    
    def L6(self,t):
    
        
        O = np.array([self.rm/np.sqrt(2),self.rm/np.sqrt(2)])
        alpha1 = -np.pi*3/4       
        alpha2 = np.math.asin((self.hc-self.rm/np.sqrt(2))/self.R)
        
        t = t * (alpha1-alpha2) + alpha2
        Pts = np.vstack((self.R*np.cos(t)+O[0],self.R*np.sin(t)+O[1])).transpose() @ (np.eye(2) * self.scale)
        T = np.vstack((-self.R*np.sin(t)*(alpha1-alpha2),self.R*np.cos(t)*(alpha1-alpha2))).transpose()
        N = T @ np.array([[0,1],[-1,0]])
        
        return Pts, T, N
    
    def L7(self,t):
    
        Pts = np.array([[self.Dc,0],[self.Dc,self.hc]])
        Pts = Pts @ (np.eye(2) * self.scale)
        
        return Lines(t,Pts)
    
    def L8(self,t):
    
        Pts = np.array([[self.Di,0],[self.Do,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        
        return Lines(t,Pts)
    
    def is_D(self,P):
        tmp1 = np.logical_and(P[:,0]>=0,P[:,0]/self.scale<=self.Do/np.sqrt(2))
        tmp2 = np.logical_and(P[:,1]<=P[:,0],P[:,1]>=0)
        tmp3 = np.logical_and(P[:,0]/self.scale>=self.Do/np.sqrt(2),P[:,0]/self.scale<=self.Do)
        tmp4 = np.logical_and(P[:,1]/self.scale<=self.Do/np.sqrt(2)+self.Do*(np.tan(np.pi/8)-1/np.sqrt(2))*(P[:,0]/self.scale-self.Do/np.sqrt(2))/(self.Do-self.Do/np.sqrt(2)),P[:,1]>=0)
        return np.logical_or(np.logical_and(tmp1,tmp2), np.logical_and(tmp3,tmp4))

    def is_D1(self,P):
        return np.logical_and(self.is_D(P),np.logical_not(np.logical_or(self.is_D2(P),self.is_D3(P)))) 
    
    def is_D2(self,P):
        tmp1 = np.logical_and(P[:,0]/self.scale>=self.Dc,P[:,0]/self.scale<=self.Dc+self.blc)
        tmp2 = np.logical_and(P[:,1]/self.scale<=self.hc+(self.hi-self.hc)*(P[:,0]/self.scale-self.Dc)/self.blc,P[:,1]>=0)
        tmp3 = np.logical_and(P[:,0]/self.scale>=self.Dc+self.blc,P[:,0]/self.scale<=self.Di-self.bli)
        tmp4 = np.logical_and(P[:,1]/self.scale<=self.hi,P[:,1]>=0)
        tmp5 = np.logical_and(P[:,0]/self.scale>=self.Di-self.bli,P[:,0]/self.scale<=self.Di)
        tmp6 = np.logical_and(P[:,1]/self.scale<=self.hi+(-self.bli)*(P[:,0]/self.scale-self.Di+self.bli)/self.bli,P[:,1]>=0)
        return np.logical_or(np.logical_or(np.logical_and(tmp1,tmp2), np.logical_and(tmp3,tmp4)),np.logical_and(tmp5,tmp6))

    def is_D3(self,P):
        tmp1 = np.logical_and(P[:,0]>=0,P[:,0]<=self.ri/np.sqrt(2)*self.scale)
        tmp2 = np.logical_and(P[:,1]<=P[:,0],P[:,1]>=0)
        tmp3 = np.logical_and(P[:,0]>=self.ri/np.sqrt(2)*self.scale,P[:,0]<=self.Dc*self.scale)
        tmp4 = np.logical_and(P[:,1]/self.scale<=self.rm/np.sqrt(2)-np.sqrt(self.R**2-(P[:,0]/self.scale-self.rm/np.sqrt(2))**2),P[:,1]>=0)
        return np.logical_or(np.logical_and(tmp1,tmp2), np.logical_and(tmp3,tmp4))
        
    def goe_file(self):
        
        s = 'h_fine=0.0125;\nh_coarse=0.05;\n\n'
        # L1
        Pts = np.array([[self.Dc,self.hc],[self.Dc+self.blc,self.hi],[self.Di-self.bli,self.hi],[self.Di,self.hi-self.bli],[self.Di,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i in range(Pts.shape[0]):
            s += 'Point(1%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n'
        
        # L2
        Pts = np.array([[self.Do,0],[self.Do,self.Do * np.tan(np.pi/8)],[self.Do/np.sqrt(2),self.Do/np.sqrt(2)],[self.ri/np.sqrt(2),self.ri/np.sqrt(2)]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i in range(Pts.shape[0]):
            s += 'Point(2%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n'
        
        # L3
        Pts = np.array([[self.Dc,0],[self.Di,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i in range(Pts.shape[0]):
            s += 'Point(3%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n'
        
        # L4
        Pts = np.array([[0,0],[self.ri/np.sqrt(2),self.ri/np.sqrt(2)]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i in range(Pts.shape[0]):
            s += 'Point(4%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n'
        
        # L5
        Pts = np.array([[0,0],[self.Dc,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i in range(Pts.shape[0]):
            s += 'Point(5%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n'
        
        # L6
        Pts = np.array([[self.rm/np.sqrt(2),self.rm/np.sqrt(2)]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i  in range(Pts.shape[0]):
            s += 'Point(6%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n'
        
        # L7
        Pts = np.array([[self.Dc,0],[self.Dc,self.hc]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i in range(Pts.shape[0]):
            s += 'Point(7%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n'
        
        # L8
        Pts = np.array([[self.Di,0],[self.Do,0]])
        Pts = Pts @ (np.eye(2) * self.scale)
        for i in range(Pts.shape[0]):
            s += 'Point(8%d) = {%4.20f,%4.20f, 0, 1.0};\n' % (i,Pts[i,0],Pts[i,1])
        s+= '\n\n'
        
        s+= 'Line(100) = {10,11,12,13,14};\n'
        s+= 'Line(200) = {20,21,22,23};\n'
        s+= 'Line(300) = {30,31};\n'
        s+= 'Line(400) = {40,41};\n'
        s+= 'Line(500) = {50,51};\n'
        s+= 'Circle(600) = {23, 60, 10};\n'
        s+= 'Line(700) = {70,71};\n'
        s+= 'Line(800) = {80,81};\n'
        return s
    
    def get_area_cu(self):
        area = 0.5*self.blc*(self.hc+self.hi)+self.hi*(self.Di-self.bli-self.Dc-self.blc)+0.5*self.bli*(self.hi*2-self.bli)
        return area*self.scale**2
    
if __name__ == "__main__":
    geo = Geometry(10)
    
    plt.figure()
    
    P, T, N = geo.L1(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])   
    
    P, T, N = geo.L2(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])   
    
    P, T, N = geo.L3(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])      
        
    P, T, N = geo.L4(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])      
        
    P, T, N = geo.L5(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])  
    
    P, T, N = geo.L6(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])  


    P, T, N = geo.L7(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])  
    
    P, T, N = geo.L8(np.linspace(0,1,100))
    plt.plot(P[:,0],P[:,1])  
    plt.quiver(P[:,0],P[:,1],T[:,0],T[:,1])  
        
    
    plt.legend([r'$\Gamma_'+str(i)+'$' for i in range(1,9)])
    
    pts = np.random.rand(100000,2) 
   
    idx = geo.is_D3(pts)
    plt.scatter(pts[idx,0],pts[idx,1],s=1,c='b')
    
    idx = geo.is_D2(pts)
    plt.scatter(pts[idx,0],pts[idx,1],s=1,c='red')
    
    idx = geo.is_D1(pts)
    plt.scatter(pts[idx,0],pts[idx,1],s=1,c='g')
    
    print('Inside ', np.sum(geo.is_D(pts)))
    
    print(geo.goe_file())
    
    pts = np.random.rand(1000000,2) 
    idx = geo.is_D2(pts)
    
    print('Analytical ' ,geo.get_area_cu(),' MC ',np.sum(idx)/pts.shape[0])

    