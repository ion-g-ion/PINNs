#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:44:10 2020

@author: ion
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
# set seed for reproducibility
# tf.random.set_seed(123)
# np.random.seed(123)




# materials
mu0 = 1 # 4*np.pi*1e-7
mur_Fe = 1000 # 3000
J0_ext = 5 # 551150
def f1(H):
    mur = mur_Fe
    B= mur * mu0 * H
   # B = mu*tf.tanh(H)+0.15*H
    return B

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

class MLP(Layer):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        
        

    def build(self, input_shape):
        
        nl  = [10]*7
        self.W1 = self.add_weight( shape=(1, nl[0]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[0])),trainable=True,dtype=basic_type)
        self.W2 = self.add_weight( shape=(1, nl[0]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[0])),trainable=True,dtype=basic_type) 
        self.B1 = self.add_weight(shape=(nl[0],),initializer='zeros',trainable=True,dtype=basic_type)       
        self.W3 = self.add_weight( shape=(nl[0], nl[1]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[1])),trainable=True,dtype=basic_type) 
        self.B3 = self.add_weight(shape=(nl[1],),initializer='zeros',trainable=True,dtype=basic_type) 
        self.W4 = self.add_weight( shape=(nl[1], nl[2]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[2])),trainable=True,dtype=basic_type) 
        self.B4 = self.add_weight(shape=(nl[2],),initializer='zeros',trainable=True,dtype=basic_type) 
        self.W5 = self.add_weight( shape=(nl[2], nl[3]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[3])),trainable=True,dtype=basic_type) 
        self.B5 = self.add_weight(shape=(nl[3],),initializer='zeros',trainable=True,dtype=basic_type) 
        self.W6 = self.add_weight( shape=(nl[3], nl[4]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[4])),trainable=True,dtype=basic_type) 
        self.B6 = self.add_weight(shape=(nl[4],),initializer='zeros',trainable=True,dtype=basic_type) 
        self.W7 = self.add_weight( shape=(nl[4], nl[5]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[5])),trainable=True,dtype=basic_type) 
        self.B7 = self.add_weight(shape=(nl[5],),initializer='zeros',trainable=True,dtype=basic_type) 
        self.W8 = self.add_weight( shape=(nl[5], nl[6]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[6])),trainable=True,dtype=basic_type) 
        self.B8 = self.add_weight(shape=(nl[6],),initializer='zeros',trainable=True,dtype=basic_type) 
        self.alpha = self.add_weight(shape=(1,),initializer='ones',trainable=True,dtype=basic_type) 
        
        self.W9 = self.add_weight( shape=(nl[6], 1),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/1)),trainable=True,dtype=basic_type)
        self.B9 = self.add_weight(shape=(1,),initializer='zeros',trainable=True,dtype=basic_type) 
        
        super(MLP, self).build(input_shape)
    
    def call(self, inputs,inputs2):
        L = tf.nn.tanh( self.alpha * (inputs @ self.W1 + inputs2 @ self.W2 + self.B1 ) )
        L = tf.nn.tanh( self.alpha * ( L @ self.W3 + self.B3) )
        L = tf.nn.tanh( self.alpha * ( L @ self.W4 + self.B4) )
        L = tf.nn.tanh( self.alpha * ( L @ self.W5 + self.B5) )
        L = tf.nn.tanh( self.alpha * ( L @ self.W6 + self.B6) )
        L = tf.nn.tanh( self.alpha * ( L @ self.W7 + self.B7) )
        L = tf.nn.tanh( self.alpha * ( L @ self.W8 + self.B8) )
        L = L @ self.W9 + self.B9
        return L
        

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)
    
    
# define our model
class PINN(ANN):

    def __init__(self,training_data):
        super(PINN, self).__init__()
        
        # create networks
        self.Az1 = MLP()
        self.Hx1 = MLP()
        self.Hy1 = MLP()
        
        self.Az2 = MLP()
        self.Hx2 = MLP()
        self.Hy2 = MLP()
        
        self.Az3 = MLP()
        self.Hx3 = MLP()
        self.Hy3 = MLP()
        
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
        
    def call(self, x , y):
        return self.Az1(x,y)*tf.cast((x**2+y**2 < 0.2**2),dtype=basic_type)+self.Az2(x,y)*tf.cast((x**2+y**2 >= 0.2**2),dtype=basic_type)
        # return 0
    # @tf.function
    def call_loss(self): 
        Hmult = 1
        J0 = tf.constant(J0_ext/Hmult,dtype = basic_type)
        
        
        wpde = tf.constant(1.0,dtype = basic_type)
        wbd = tf.constant(10.0,dtype = basic_type)
        wmat = tf.constant(1.0,dtype = basic_type)

        # boundary conditions
        Lbd = ( tf.reduce_mean(tf.square(self.Az1(self.PL2x,self.PL2y))) )
        Lbd += ( tf.reduce_mean(tf.square(self.Az3(self.PL4x,self.PL4y))) )
        # BD of tangential 0
        # employed on H        
        Lbd += ( tf.reduce_mean(tf.square( self.Hx1(self.PL8x,self.PL8y)*self.T8x+self.Hy1(self.PL8x,self.PL8y)*self.T8y )) )
        Lbd += ( tf.reduce_mean(tf.square( self.Hx2(self.PL3x,self.PL3y)*self.T3x+self.Hy2(self.PL3x,self.PL3y)*self.T3y )) )
        Lbd += ( tf.reduce_mean(tf.square( self.Hx3(self.PL5x,self.PL5y)*self.T5x+self.Hy3(self.PL5x,self.PL5y)*self.T5y )) )
      
        
        
        #interfaces
        Li12 = ( tf.reduce_mean(tf.square(self.Az1(self.P12x,self.P12y)-self.Az2(self.P12x,self.P12y))) )
        # print('\t',Li12.numpy())
        Li12 += ( tf.reduce_mean(tf.square((self.Hx1(self.P12x,self.P12y)*self.T12x+self.Hy1(self.P12x,self.P12y)*self.T12y) / Hmult- (self.Hx2(self.P12x,self.P12y)*self.T12x+self.Hy2(self.P12x,self.P12y)*self.T12y)*Hmult )) )
        # print('\t',Li12.numpy())
        
        Li13 = ( tf.reduce_mean(tf.square(self.Az1(self.P13x,self.P13y)-self.Az3(self.P13x,self.P13y))) )
        # print('\t',Li13.numpy())
        Li13 +=  ( tf.reduce_mean(tf.square((self.Hx1(self.P13x,self.P13y)*self.T13x+self.Hy1(self.P13x,self.P13y)*self.T13y) / Hmult - (self.Hx3(self.P13x,self.P13y)*self.T13x+self.Hy3(self.P13x,self.P13y)*self.T13y)*Hmult )) )
        # print('\t',Li13.numpy())
        
        Li23 = ( tf.reduce_mean(tf.square(self.Az2(self.P23x,self.P23y)-self.Az3(self.P23x,self.P23y))) )
        # print('\t',Li23.numpy())
        Li23 += ( tf.reduce_mean(tf.square(self.Hx2(self.P23x,self.P23y)*self.T23x+self.Hy2(self.P23x,self.P23y)*self.T23y   - (self.Hx3(self.P23x,self.P23y)*self.T23x+self.Hy3(self.P23x,self.P23y)*self.T23y) )) )
        # print('\t',Li23.numpy()) 
    

        # L material 1
        L_material1 = ( tf.reduce_mean(tf.square(Dy(self.Az1,self.P1x,self.P1y)-f1(self.Hx1(self.P1x,self.P1y)/Hmult))+tf.square(-Dx(self.Az1,self.P1x,self.P1y)-f1(self.Hy1(self.P1x,self.P1y)/Hmult))) )
        L_material2 = ( tf.reduce_mean(tf.square(Dy(self.Az2,self.P2x,self.P2y)-f2(self.Hx2(self.P2x,self.P2y)*Hmult))+tf.square(-Dx(self.Az2,self.P2x,self.P2y)-f2(self.Hy2(self.P2x,self.P2y)*Hmult))) )
        L_material3 = ( tf.reduce_mean(tf.square(Dy(self.Az3,self.P3x,self.P3y)-f3(self.Hx3(self.P3x,self.P3y)*Hmult))+tf.square(-Dx(self.Az3,self.P3x,self.P3y)-f3(self.Hy3(self.P3x,self.P3y)*Hmult))) )
           
        L_material = L_material1 + L_material2 + L_material3
        
        # PDE 1
        L_ampere1 = ( tf.reduce_mean(tf.square(Dx(self.Hy1,self.P1x,self.P1y)-Dy(self.Hx1,self.P1x,self.P1y))))
        # PDE 2
        L_ampere2 =  ( tf.reduce_mean(tf.square(Dx(self.Hy2,self.P2x,self.P2y)-Dy(self.Hx2,self.P2x,self.P2y)-J0)))
        # PDE 3
        L_ampere3 = ( tf.reduce_mean(tf.square(Dx(self.Hy3,self.P3x,self.P3y)-Dy(self.Hx3,self.P3x,self.P3y))))
        
        Lpde = L_ampere1+L_ampere2+L_ampere3
        # print('\t\t%e %e %e'%(L_ampere1.numpy(),L_ampere2.numpy(),L_ampere3.numpy()))
        Li = Li12+Li13+Li23
        
        Ltot = wpde*Lpde+wbd*(Lbd+Li)+wmat*L_material
        # print('000')
        print('loss %5.3e, pde %5.3e, int %5.3e bd %5.3e, mat %5.3e'%(Ltot.numpy(),Lpde.numpy(),Li.numpy(),Lbd.numpy(),L_material.numpy()))
        # print('\t%e %e %e'%(Li12,Li13,Li23))
        return Ltot

# parameters 
Nin = 80000
Nbd = 3000

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
opti_results = model.train_bfgs(num_iter=5000)
# opti_results = model.train_lbfgs(num_iter=5000)
# opti_results = model.train_backpropagation(50000)
# model.train_backpropagation(50000)
t_training = timeit.time.time() - t_training


#%% Visualization
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
fem = FEM(mur=mur_Fe,J0=J0_ext,mu0=mu0)
A_ref = fem.call_A(x_mesh.flatten(),y_mesh.flatten()).reshape(x_mesh.shape)


plt.figure()
plt.contourf(x_mesh,y_mesh,A_comp,levels=32)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Computed A')

plt.figure()
plt.contourf(x_mesh,y_mesh,A_ref,levels=12)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Reference A')


plt.figure()
plt.contourf(x_mesh,y_mesh,np.abs(A_comp-A_ref),levels=64)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Error abs')

plt.figure()
plt.contourf(x_mesh,y_mesh,np.log10(np.abs(A_comp-A_ref)),levels=64)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Error log')

plt.figure()
plt.contour(x_mesh,y_mesh,A_comp,levels=30)
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Computed A')

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
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Computed B')

plt.figure()
plt.quiver(x_mesh,y_mesh,Hx_comp,Hy_comp,np.sqrt(Bx_comp**2+By_comp**2))
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Computed H')

plt.figure()
plt.quiver(x_mesh,y_mesh,B_ref[:,0],B_ref[:,1],np.sqrt(B_ref[:,0]**2+B_ref[:,1]**2))
plt.plot(geo.L1(np.linspace(0,1,Nbd))[0][:,0],geo.L1(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L2(np.linspace(0,1,Nbd))[0][:,0],geo.L2(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L3(np.linspace(0,1,Nbd))[0][:,0],geo.L3(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L4(np.linspace(0,1,Nbd))[0][:,0],geo.L4(np.linspace(0,1,Nbd))[0][:,1],'orange')
plt.plot(geo.L5(np.linspace(0,1,Nbd))[0][:,0],geo.L5(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.plot(geo.L6(np.linspace(0,1,Nbd))[0][:,0],geo.L6(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L7(np.linspace(0,1,Nbd))[0][:,0],geo.L7(np.linspace(0,1,Nbd))[0][:,1],'k')
plt.plot(geo.L8(np.linspace(0,1,Nbd))[0][:,0],geo.L8(np.linspace(0,1,Nbd))[0][:,1],'r')
plt.colorbar()
plt.xlabel(r'$x/x_0$', fontsize=12)
plt.ylabel(r'$y/y_0$', fontsize=12)
plt.axis('equal')
plt.title('Reference B')

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

# for i in range(x_mesh.shape[0]):
#     for j in range(x_mesh.shape[1]):
#         if not (F1([x_mesh[i,j],y_mesh[i,j]]) or F2([x_mesh[i,j],y_mesh[i,j]])):
#                    comp[i,j] = np.nan
                   
# plt.figure()
# plt.contourf(x_mesh,y_mesh,comp,levels=50)

# x_mesh = tf.constant(np.linspace(0,1,1000).reshape([-1,1]),dtype=basic_type)
# res = model.call(x_mesh,0*x_mesh).numpy()              
# res = Dx(model.call,x_mesh,0*x_mesh).numpy() 
# plt.figure()                   
# plt.plot(x_mesh.numpy(),res)