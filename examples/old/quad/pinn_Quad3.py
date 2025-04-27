#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 10:45:00 2020

@author: yonnss
"""


# import modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import sys
sys.path.insert(1, '..')
from ANN import ANN,Dx,Dy
import timeit
# sys.path.insert(1, '../fem/')
from Geometry_Quad import Geometry
from quad1 import FEM
import random

# set seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)



# materials
mu0 = 1 # 4*np.pi*1e-7
mur_Fe = 1000 # 3000
J0_ext = 3 # 551150
def f1(H):
    mur = mur_Fe
    B= mur * mu0 * H
   # B = mu*tf.tanh(H)+0.15*H
    return B

def nu_iron(Bx,By):
    k1 = 0.001
    k2 = 1.65
    k3 = 0.05
    x = Bx**2+By**2
    return k1*tf.exp(k2*x)+k3

#define material
def f2(H):
    mur = 1
    B= mur * mu0 * H
    # B = mu*tf.tanh(H)+0.15*H
    return B


#define material
def f3(H):
    mur = 1
    B= mur * mu0 * H
    # B = mu*tf.tanh(H)+0.15*H
    return B


# display or hide details
show_details = True

# choose optimization method {'bfgs','backpropagation','lbfgs'}
opti_method = 'bfgs'

basic_type = tf.float64
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
tf.keras.backend.set_floatx('float64')

def relu_cube(x):
    return K.relu(x)**3


    
    
# define our model
class PINN(ANN):

    def __init__(self,training_data):
        super(PINN, self).__init__()
        
        # define layers
        nl = 20
        self.denseH1_11 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1_12 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1_2 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1_3 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1_4 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1_5 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1x_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1x_7 =tf.keras.layers.Dense(1)
        self.denseH1y_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH1y_7 =tf.keras.layers.Dense(1)
        
        self.denseA1_11 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA1_12 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA1_2 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA1_3 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA1_4 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA1_5 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA1_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA1_7 =tf.keras.layers.Dense(1)
        
        nl = 16
        self.denseH2_11 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2_12 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2_2 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2_3 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2_4 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2_5 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2x_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2x_7 =tf.keras.layers.Dense(1)
        self.denseH2y_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH2y_7 =tf.keras.layers.Dense(1)
        
        self.denseA2_11 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA2_12 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA2_2 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA2_3 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA2_4 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA2_5 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA2_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA2_7 =tf.keras.layers.Dense(1)
        
        self.denseH3_11 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3_12 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3_2 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3_3 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3_4 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3_5 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3x_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3x_7 =tf.keras.layers.Dense(1)
        self.denseH3y_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseH3y_7 =tf.keras.layers.Dense(1)
        
        self.denseA3_11 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA3_12 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA3_2 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA3_3 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA3_4 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA3_5 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA3_6 =tf.keras.layers.Dense(nl,activation=tf.nn.tanh)
        self.denseA3_7 =tf.keras.layers.Dense(1)
        
        
        
        
        # init networks
        test =tf.random.uniform([10,1],dtype=basic_type)
        self.Az1(test,test)
        self.Hx1(test,test)
        self.Hy1(test,test)
        self.Az2(test,test)
        self.Hx2(test,test)
        self.Hy2(test,test)
        self.Az3(test,test)
        self.Hx3(test,test)
        self.Hy3(test,test)

        # interior points
        self.P1x = tf.reshape(training_data['P1'][:,0],[-1,1])
        self.P1y = tf.reshape(training_data['P1'][:,1],[-1,1])
        self.P2x = tf.reshape(training_data['P2'][:,0],[-1,1])
        self.P2y = tf.reshape(training_data['P2'][:,1],[-1,1])
        self.P3x = tf.reshape(training_data['P3'][:,0],[-1,1])
        self.P3y = tf.reshape(training_data['P3'][:,1],[-1,1])
        # D1-D2 interface
        self.P12x = tf.reshape(training_data['L1'][0][:,0],[-1,1])
        self.P12y = tf.reshape(training_data['L1'][0][:,1],[-1,1])
        self.T12x = tf.reshape(training_data['L1'][1][:,0],[-1,1])
        self.T12y = tf.reshape(training_data['L1'][1][:,1],[-1,1])
        # D1-D3 interface
        self.P13x = tf.reshape(training_data['L6'][0][:,0],[-1,1])
        self.P13y = tf.reshape(training_data['L6'][0][:,1],[-1,1])
        self.T13x = tf.reshape(training_data['L6'][1][:,0],[-1,1])
        self.T13y = tf.reshape(training_data['L6'][1][:,1],[-1,1])
        # D2-D3 interface
        self.P23x = tf.reshape(training_data['L7'][0][:,0],[-1,1])
        self.P23y = tf.reshape(training_data['L7'][0][:,1],[-1,1])
        self.T23x = tf.reshape(training_data['L7'][1][:,0],[-1,1])
        self.T23y = tf.reshape(training_data['L7'][1][:,1],[-1,1])
        # L2 boundary
        self.PL2x = tf.reshape(training_data['L2'][0][:,0],[-1,1])
        self.PL2y = tf.reshape(training_data['L2'][0][:,1],[-1,1])
        # L4 boundary
        self.PL4x = tf.reshape(training_data['L4'][0][:,0],[-1,1])
        self.PL4y = tf.reshape(training_data['L4'][0][:,1],[-1,1])
        # L8 boundary
        self.PL8x = tf.reshape(training_data['L8'][0][:,0],[-1,1])
        self.PL8y = tf.reshape(training_data['L8'][0][:,1],[-1,1])
        self.T8x = tf.reshape(training_data['L8'][1][:,0],[-1,1])
        self.T8y = tf.reshape(training_data['L8'][1][:,1],[-1,1])
        # L3 boundary
        self.PL3x = tf.reshape(training_data['L3'][0][:,0],[-1,1])
        self.PL3y = tf.reshape(training_data['L3'][0][:,1],[-1,1])
        self.T3x = tf.reshape(training_data['L3'][1][:,0],[-1,1])
        self.T3y = tf.reshape(training_data['L3'][1][:,1],[-1,1])
        # L5 boundary
        self.PL5x = tf.reshape(training_data['L5'][0][:,0],[-1,1])
        self.PL5y = tf.reshape(training_data['L5'][0][:,1],[-1,1])
        self.T5x = tf.reshape(training_data['L5'][1][:,0],[-1,1])
        self.T5y = tf.reshape(training_data['L5'][1][:,1],[-1,1])
    
    def Hx1(self,x,y):
        L = self.denseH1_11(x)+self.denseH1_12(y)
        L = self.denseH1_2(L)
        L = self.denseH1_3(L)
        L = self.denseH1_4(L)
        L = self.denseH1_5(L)
        L = self.denseH1x_6(L)
        L = self.denseH1x_7(L)
        return L
    
    def Hy1(self,x,y):
        L = self.denseH1_11(x)+self.denseH1_12(y)
        L = self.denseH1_2(L)
        L = self.denseH1_3(L)
        L = self.denseH1_4(L)
        L = self.denseH1_5(L)
        L = self.denseH1y_6(L)
        L = self.denseH1y_7(L)
        return L
    
    def Az1(self,x,y):
        L = self.denseA1_11(x)+self.denseA1_12(y)
        L = self.denseA1_2(L)
        L = self.denseA1_3(L)
        L = self.denseA1_4(L)
        L = self.denseA1_5(L)
        L = self.denseA1_6(L)
        L = self.denseA1_7(L)
        return L
    
    def Hx2(self,x,y):
        L = self.denseH2_11(x)+self.denseH2_12(y)
        L = self.denseH2_2(L)
        L = self.denseH2_3(L)
        L = self.denseH2_4(L)
        L = self.denseH2_5(L)
        L = self.denseH2x_6(L)
        L = self.denseH2x_7(L)
        return L
    
    def Hy2(self,x,y):
        L = self.denseH2_11(x)+self.denseH2_12(y)
        L = self.denseH2_2(L)
        L = self.denseH2_3(L)
        L = self.denseH2_4(L)
        L = self.denseH2_5(L)
        L = self.denseH2y_6(L)
        L = self.denseH2y_7(L)
        return L
    
    def Az2(self,x,y):
        L = self.denseA2_11(x)+self.denseA2_12(y)
        L = self.denseA2_2(L)
        L = self.denseA2_3(L)
        L = self.denseA2_4(L)
        L = self.denseA2_5(L)
        L = self.denseA2_6(L)
        L = self.denseA2_7(L)
        return L
    
    def Hx3(self,x,y):
        L = self.denseH3_11(x)+self.denseH3_12(y)
        L = self.denseH3_2(L)
        L = self.denseH3_3(L)
        L = self.denseH3_4(L)
        L = self.denseH3_5(L)
        L = self.denseH3x_6(L)
        L = self.denseH3x_7(L)
        return L
    
    def Hy3(self,x,y):
        L = self.denseH3_11(x)+self.denseH3_12(y)
        L = self.denseH3_2(L)
        L = self.denseH3_3(L)
        L = self.denseH3_4(L)
        L = self.denseH3_5(L)
        L = self.denseH3y_6(L)
        L = self.denseH3y_7(L)
        return L
    
    def Az3(self,x,y):
        L = self.denseA3_11(x)+self.denseA3_12(y)
        L = self.denseA3_2(L)
        L = self.denseA3_3(L)
        L = self.denseA3_4(L)
        L = self.denseA3_5(L)
        L = self.denseA3_6(L)
        L = self.denseA3_7(L)
        return L
    
    
    def call(self, x , y):
        return self.Az1(x,y)*tf.cast((x**2+y**2 < 0.2**2),dtype=basic_type)+self.Az2(x,y)*tf.cast((x**2+y**2 >= 0.2**2),dtype=basic_type)
        # return 0
    # @tf.function
    def call_loss(self): 
        Hmult = 10
        J0 = tf.constant(J0_ext,dtype = basic_type)
        
        
        wpde = tf.constant(1.0,dtype = basic_type)
        wbd = tf.constant(10.0,dtype = basic_type)
        wmat = tf.constant(10.0,dtype = basic_type)

        # boundary conditions
        Lbd1 = ( tf.reduce_mean(tf.square(self.Az1(self.PL2x,self.PL2y))) )
        Lbd2 = ( tf.reduce_mean(tf.square(self.Az3(self.PL4x,self.PL4y))) )
        # BD of tangential 0
        # employed on H        
        Lbd3 = Hmult**2*( tf.reduce_mean(tf.square( self.Hx1(self.PL8x,self.PL8y)*self.T8x+self.Hy1(self.PL8x,self.PL8y)*self.T8y )) )
        Lbd4 = ( tf.reduce_mean(tf.square( self.Hx2(self.PL3x,self.PL3y)*self.T3x+self.Hy2(self.PL3x,self.PL3y)*self.T3y )) )
        Lbd5 = ( tf.reduce_mean(tf.square( self.Hx3(self.PL5x,self.PL5y)*self.T5x+self.Hy3(self.PL5x,self.PL5y)*self.T5y )) )
      
        Lbd = Lbd1 + Lbd2 + Lbd3 + Lbd4 + Lbd5
        
        #interfaces
        Li12 = ( tf.reduce_mean(tf.square(self.Az1(self.P12x,self.P12y)-self.Az2(self.P12x,self.P12y))) )
        print('\t',Li12.numpy())
        Li12 += Hmult**2*( tf.reduce_mean(tf.square((self.Hx1(self.P12x,self.P12y)*self.T12x+self.Hy1(self.P12x,self.P12y)*self.T12y) - (self.Hx2(self.P12x,self.P12y)*self.T12x+self.Hy2(self.P12x,self.P12y)*self.T12y) )) )
        print('\t',Li12.numpy())
        
        Li13 = ( tf.reduce_mean(tf.square(self.Az1(self.P13x,self.P13y)-self.Az3(self.P13x,self.P13y))) )
        print('\t',Li13.numpy())
        Li13 += Hmult**2*( tf.reduce_mean(tf.square((self.Hx1(self.P13x,self.P13y)*self.T13x+self.Hy1(self.P13x,self.P13y)*self.T13y) - (self.Hx3(self.P13x,self.P13y)*self.T13x+self.Hy3(self.P13x,self.P13y)*self.T13y) )) )
        print('\t',Li13.numpy())
        
        Li23 = ( tf.reduce_mean(tf.square(self.Az2(self.P23x,self.P23y)-self.Az3(self.P23x,self.P23y))) )
        print('\t',Li23.numpy())
        Li23 += ( tf.reduce_mean(tf.square(self.Hx2(self.P23x,self.P23y)*self.T23x+self.Hy2(self.P23x,self.P23y)*self.T23y   - (self.Hx3(self.P23x,self.P23y)*self.T23x+self.Hy3(self.P23x,self.P23y)*self.T23y) )) )
        print('\t',Li23.numpy()) 
    

        # L material 1
        # L_material1 = ( tf.reduce_mean(tf.square(Dy(self.Az1,self.P1x,self.P1y)-f1(self.Hx1(self.P1x,self.P1y)))+tf.square(-Dx(self.Az1,self.P1x,self.P1y)-f1(self.Hy1(self.P1x,self.P1y)))) )
        Bx1 = Dy(self.Az1,self.P1x,self.P1y)
        By1 = -Dx(self.Az1,self.P1x,self.P1y)
        Hx1 = self.Hx1(self.P1x,self.P1y)
        Hy1 = self.Hy1(self.P1x,self.P1y)
        L_material1 = Hmult**2*( tf.reduce_mean( tf.square( Hx1 - nu_iron(Bx1, By1)*Bx1 ) + tf.square( Hy1 - nu_iron(Bx1, By1)*By1 ) ) )
    
        L_material2 = ( tf.reduce_mean(tf.square(Dy(self.Az2,self.P2x,self.P2y)-f2(self.Hx2(self.P2x,self.P2y)))+tf.square(-Dx(self.Az2,self.P2x,self.P2y)-f2(self.Hy2(self.P2x,self.P2y)))) )
        L_material3 = ( tf.reduce_mean(tf.square(Dy(self.Az3,self.P3x,self.P3y)-f3(self.Hx3(self.P3x,self.P3y)))+tf.square(-Dx(self.Az3,self.P3x,self.P3y)-f3(self.Hy3(self.P3x,self.P3y)))) )
           
        L_material = L_material1 + L_material2 + L_material3
        
        # PDE 1
        L_ampere1 = Hmult**2*( tf.reduce_mean(tf.square(Dx(self.Hy1,self.P1x,self.P1y)-Dy(self.Hx1,self.P1x,self.P1y))))
        # PDE 2
        L_ampere2 = ( tf.reduce_mean(tf.square(Dx(self.Hy2,self.P2x,self.P2y)-Dy(self.Hx2,self.P2x,self.P2y)-J0)))
        # PDE 3
        L_ampere3 = ( tf.reduce_mean(tf.square(Dx(self.Hy3,self.P3x,self.P3y)-Dy(self.Hx3,self.P3x,self.P3y))))
        
        print('\t\tMAT %5.3e %5.3e %5.3e'%(L_material1.numpy(),L_material2.numpy(),L_material3.numpy()))
        Lpde = L_ampere1+L_ampere2+L_ampere3
        print('\t\tPDE %5.3e %5.3e %5.3e'%(L_ampere1.numpy(),L_ampere2.numpy(),L_ampere3.numpy()))
        Li = Li12+Li13+Li23
        print('\t\tBD  %5.3e %5.3e %5.3e %5.3e %5.3e'%(Lbd1,Lbd2,Lbd3,Lbd4,Lbd5))
        Ltot = wpde*Lpde+wbd*(Lbd+Li)+wmat*L_material
        # print('000')
        print('loss %5.3e, pde %5.3e, int %5.3e bd %5.3e, mat %5.3e'%(Ltot.numpy(),Lpde.numpy(),Li.numpy(),Lbd.numpy(),L_material.numpy()))
        # print('\t%e %e %e'%(Li12,Li13,Li23))
        return Ltot

# parameters 
Nin = 80000
Nbd = 5000


# points inside the domain
geo = Geometry(10)
Pts = np.random.rand(Nin,2)
P_D1 = Pts[geo.is_D1(Pts),:]
P_D2 = Pts[geo.is_D2(Pts),:]
P_D3 = Pts[geo.is_D3(Pts),:]
Nin_actual = P_D1.shape[0]+P_D2.shape[0]+P_D3.shape[0]

inputs = dict()
inputs['P1'] = tf.constant(P_D1,dtype=basic_type)
inputs['P2'] = tf.constant(P_D2,dtype=basic_type)
inputs['P3'] = tf.constant(P_D3,dtype=basic_type)

P_L1 = geo.L1(np.random.rand(Nbd))
P_L2 = geo.L2(np.random.rand(Nbd))
P_L3 = geo.L3(np.random.rand(Nbd))
P_L4 = geo.L4(np.random.rand(Nbd))
P_L5 = geo.L5(np.random.rand(Nbd))  
P_L6 = geo.L6(np.random.rand(Nbd))
P_L7 = geo.L7(np.random.rand(Nbd))
P_L8 = geo.L8(np.random.rand(Nbd))

inputs['L1'] = tf.constant(P_L1,dtype=basic_type)
inputs['L2'] = tf.constant(P_L2,dtype=basic_type)
inputs['L3'] = tf.constant(P_L3,dtype=basic_type)
inputs['L4'] = tf.constant(P_L4,dtype=basic_type)
inputs['L5'] = tf.constant(P_L5,dtype=basic_type)
inputs['L6'] = tf.constant(P_L6,dtype=basic_type)
inputs['L7'] = tf.constant(P_L7,dtype=basic_type)
inputs['L8'] = tf.constant(P_L8,dtype=basic_type)



# Draw the geometry
plt.figure()
plt.scatter(P_D1[:,0],P_D1[:,1],s=0.1,c='blue')
plt.scatter(P_D2[:,0],P_D2[:,1],s=0.1,c='red')
plt.scatter(P_D3[:,0],P_D3[:,1],s=0.1,c='green')
# plt.legend([r'$\Omega_1$',r'$\Omega_2$',r'$\Omega_3$'])
plt.scatter(P_L1[0][:,0],P_L1[0][:,1],s=1,c='k')
plt.scatter(P_L2[0][:,0],P_L2[0][:,1],s=1,c='k')
plt.scatter(P_L3[0][:,0],P_L3[0][:,1],s=1,c='k')
plt.scatter(P_L4[0][:,0],P_L4[0][:,1],s=1,c='k')
plt.scatter(P_L5[0][:,0],P_L5[0][:,1],s=1,c='k')
plt.scatter(P_L6[0][:,0],P_L6[0][:,1],s=1,c='k')
plt.scatter(P_L7[0][:,0],P_L7[0][:,1],s=1,c='k')
plt.scatter(P_L8[0][:,0],P_L8[0][:,1],s=1,c='k')
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')

model = PINN(inputs)


t_training = timeit.time.time()
opti_results = model.train_bfgs(num_iter=8)
# opti_results = model.train_lbfgs(num_iter=5000)
# opti_results = model.train_backpropagation(50000)
# model.train_backpropagation(50000)
t_training = timeit.time.time() - t_training


#%% Visualization
show_title = False 
x_mesh, y_mesh = np.meshgrid(np.linspace(0,0.75,400),np.linspace(0,0.55,400))
A1_comp = model.Az1(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
A2_comp = model.Az2(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
A3_comp = model.Az3(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()


idx = geo.is_D1(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A1_comp[np.logical_not(idx)] = 0

idx = geo.is_D2(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A2_comp[np.logical_not(idx)] = 0

idx = geo.is_D3(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A3_comp[np.logical_not(idx)] = 0


A_comp = A1_comp + A2_comp + A3_comp
idx = geo.is_D(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A_comp[np.logical_not(idx)] = np.nan


A_comp = A_comp.reshape(x_mesh.shape)

# np.save('../fem/quad_xmesh.dat',x_mesh)
# np.save('../fem/quad_ymesh.dat',y_mesh)
# np.save('../fem/quad_A_comp.dat',A_comp)
# A_ref = np.load('./quad_A_ref.dat.npy')
fem = FEM(mur=mur_Fe,J0=J0_ext,mu0=mu0,k1 = 0.001,k3 = 0.05)
A_ref = fem.call_A(x_mesh.flatten(),y_mesh.flatten()).reshape(x_mesh.shape)


plt.figure()
plt.contourf(x_mesh,y_mesh,A_comp,levels=20)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Computed A')

plt.figure()
plt.contourf(x_mesh,y_mesh,A_ref,levels=20)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Reference A')


plt.figure()
plt.contourf(x_mesh,y_mesh,np.abs(A_comp-A_ref),levels=12)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Error abs')

plt.figure()
plt.contourf(x_mesh,y_mesh,np.log10(np.abs(A_comp-A_ref)),levels=12)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Error log')

plt.figure()
plt.contour(x_mesh,y_mesh,A_comp,levels=30)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Computed A')

# plot the B-field
x_mesh, y_mesh = np.meshgrid(np.linspace(0,0.75,40),np.linspace(0,0.55,40))
Bx_1 = Dy(model.Az1,tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
Bx_2 = Dy(model.Az2,tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
Bx_3 = Dy(model.Az3,tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
By_1 = -Dx(model.Az1,tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
By_2 = -Dx(model.Az2,tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
By_3 = -Dx(model.Az3,tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()

idx = geo.is_D1(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Bx_1[np.logical_not(idx)] = 0
By_1[np.logical_not(idx)] = 0

idx = geo.is_D2(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Bx_2[np.logical_not(idx)] = 0
By_2[np.logical_not(idx)] = 0

idx = geo.is_D3(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Bx_3[np.logical_not(idx)] = 0
By_3[np.logical_not(idx)] = 0

Bx_comp = Bx_1 + Bx_2 + Bx_3
By_comp = By_1 + By_2 + By_3
idx = geo.is_D(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Bx_comp[np.logical_not(idx)] = np.nan
By_comp[np.logical_not(idx)] = np.nan
Bx_comp = Bx_comp.reshape(x_mesh.shape)
By_comp = By_comp.reshape(x_mesh.shape)
B_ref = fem.call_B(x_mesh.flatten(),y_mesh.flatten())

Hx_1 = model.Hx1(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
Hx_2 = model.Hx2(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
Hx_3 = model.Hx3(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
Hy_1 = model.Hy1(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
Hy_2 = model.Hy2(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
Hy_3 = model.Hy3(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()

idx = geo.is_D1(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Hx_1[np.logical_not(idx)] = 0
Hy_1[np.logical_not(idx)] = 0

idx = geo.is_D2(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Hx_2[np.logical_not(idx)] = 0
Hy_2[np.logical_not(idx)] = 0

idx = geo.is_D3(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Hx_3[np.logical_not(idx)] = 0
Hy_3[np.logical_not(idx)] = 0

Hx_comp = Hx_1 + Hx_2 + Hx_3
Hy_comp = Hy_1 + Hy_2 + Hy_3
idx = geo.is_D(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
Hx_comp[np.logical_not(idx)] = np.nan
Hy_comp[np.logical_not(idx)] = np.nan
Hx_comp = Hx_comp.reshape(x_mesh.shape)
Hy_comp = Hy_comp.reshape(x_mesh.shape)

plt.figure()
plt.quiver(x_mesh,y_mesh,Bx_comp,By_comp,np.sqrt(Bx_comp**2+By_comp**2))
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Computed B')

plt.figure()
plt.quiver(x_mesh,y_mesh,Hx_comp,Hy_comp,np.sqrt(Bx_comp**2+By_comp**2))
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Computed H')

plt.figure()
plt.quiver(x_mesh,y_mesh,B_ref[:,0],B_ref[:,1],np.sqrt(B_ref[:,0]**2+B_ref[:,1]**2))
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
if show_title: plt.title('Reference B')

err_max = np.nanmax(np.abs(A_comp-A_ref).flatten())
err_max_rel = np.nanmax(np.abs(A_comp-A_ref).flatten()) / np.nanmax(np.abs(A_ref).flatten())
err_rms = np.sqrt(np.nanmean((A_comp-A_ref).flatten()**2))

print('Max error     %e'%(err_max))
print('Max rel error %f %%'%(err_max_rel*100))
print('RMS error     %e'%(err_rms))

# geometry plot
plt.figure()
plt.fill(np.concatenate((geo.L1(np.linspace(0,1,Nbd))[0][:,0], geo.L3(np.linspace(0,1,Nbd))[0][:,0], geo.L7(np.linspace(0,1,Nbd))[0][:,0])), np.concatenate((geo.L1(np.linspace(0,1,Nbd))[0][:,1], geo.L3(np.linspace(0,1,Nbd))[0][:,1], geo.L7(np.linspace(0,1,Nbd))[0][:,1])),'orange')
plt.fill(np.concatenate((geo.L8(np.linspace(0,1,Nbd))[0][:,0], geo.L2(np.linspace(0,1,Nbd))[0][:,0], geo.L6(np.linspace(0,1,Nbd)[::-1])[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,0])), np.concatenate((geo.L8(np.linspace(0,1,Nbd))[0][:,1], geo.L2(np.linspace(0,1,Nbd))[0][:,1], geo.L6(np.linspace(0,1,Nbd)[::-1])[0][:,1],geo.L1(np.linspace(0,1,Nbd))[0][:,1])),'grey')
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'g')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'g')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.text(0.30, 0.35, r'$\Gamma_{\rm D}$', fontsize=12)
plt.text(0.35, -0.025, r'$\Gamma_{\rm N}$', fontsize=12)
plt.text(0.13, 0.05, r'Air', fontsize=12)
plt.text(0.45, 0.25, r'Iron', fontsize=12)
plt.text(0.4, 0.061, r'Cu', fontsize=12)
plt.text(0.39, 0.03, r'$\vec{J}\neq\vec{0}$', fontsize=12)

#%% L2 error
Nsample = 1000000
x_mesh = np.random.rand(Nsample)
y_mesh = np.random.rand(Nsample)
A1_comp = model.Az1(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
A2_comp = model.Az2(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()
A3_comp = model.Az3(tf.constant(x_mesh.reshape([-1,1]),dtype=basic_type),tf.constant(y_mesh.reshape([-1,1]),dtype=basic_type)).numpy()


idx = geo.is_D1(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A1_comp[np.logical_not(idx)] = 0

idx = geo.is_D2(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A2_comp[np.logical_not(idx)] = 0

idx = geo.is_D3(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A3_comp[np.logical_not(idx)] = 0


A_comp = A1_comp + A2_comp + A3_comp
idx = geo.is_D(np.hstack((x_mesh.reshape([-1,1]),y_mesh.reshape([-1,1]))))
A_comp[np.logical_not(idx)] = np.nan

A_ref = fem.call_A(x_mesh.flatten(),y_mesh.flatten()).reshape(x_mesh.shape)

err_max = np.nanmax(np.abs(A_comp.flatten()-A_ref.flatten()))/np.nanmax(np.abs(A_ref.flatten()))
err_L2 = np.sqrt(np.nansum((A_comp.flatten()-A_ref.flatten())**2) / (np.nansum(A_ref*0+1) / Nsample) ) / np.sqrt(np.nansum(A_ref**2) / (np.nansum(A_ref*0+1) / Nsample) )
err_mean = np.nanmean(np.abs(A_comp.flatten()-A_ref.flatten())) / np.nanmean(A_ref)
print('\n\nErrors (relative):\n\tL2    %e\n\tLinf  %f %%\n\tmean  %f %%'%(err_L2,err_max*100,err_mean*100))