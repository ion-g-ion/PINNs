#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 20:12:50 2020

@author: yonnss
"""



# import modules
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
from ANN import ANN,Laplace,Line2D,SampleBD,SampleInside,Dx,Dy
import timeit
import sys
# sys.path.insert(1, '../fem/')
from Geometry_Quad import Geometry
from quad1 import call_fem
# set seed for reproducibility
tf.random.set_seed(123)
np.random.seed(123)

# materials
def nu1(B):
    mu = 10
    nu= 1/mu +B*0
   # B = mu*tf.tanh(H)+0.15*H
    return nu

#define material
def nu2(B):
    mu = 1
    nu= 1/mu +B*0
   # B = mu*tf.tanh(H)+0.15*H
    return nu


#define material
def nu3(B):
    mu = 1
    nu= 1/mu +B*0
   # B = mu*tf.tanh(H)+0.15*H
    return nu


# display or hide details
show_details = True

# choose optimization method {'bfgs','backpropagation','lbfgs'}
opti_method = 'bfgs'

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

def curl_nu_curl_2d(Az,nu,x,y):
    
    with tf.GradientTape(persistent=True) as t:
        t.watch([x,y])
        with tf.GradientTape(persistent=True) as t2:
            t2.watch([x,y])
            A = Az(x,y)
            Bx = t2.gradient(A, y)
            By = -t2.gradient(A, x)
        Hx = nu(Bx)*Bx
        Hy = nu(By)*By
        cc = t.gradient(Hy, x) - t.gradient(Hx, y)
    return cc

def relu_cube(x):
    return K.relu(x)**3

class MLP(Layer):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        
        

    def build(self, input_shape):
        
        nl  = [16]*7
        self.W1 = self.add_weight( shape=(1, nl[0]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[0])),trainable=True)
        self.W2 = self.add_weight( shape=(1, nl[0]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[0])),trainable=True) 
        self.B1 = self.add_weight(shape=(nl[0],),initializer='zeros',trainable=True)       
        self.W3 = self.add_weight( shape=(nl[0], nl[1]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[1])),trainable=True) 
        self.B3 = self.add_weight(shape=(nl[1],),initializer='zeros',trainable=True) 
        self.W4 = self.add_weight( shape=(nl[1], nl[2]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[2])),trainable=True) 
        self.B4 = self.add_weight(shape=(nl[2],),initializer='zeros',trainable=True) 
        self.W5 = self.add_weight( shape=(nl[2], nl[3]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[3])),trainable=True) 
        self.B5 = self.add_weight(shape=(nl[3],),initializer='zeros',trainable=True) 
        self.W6 = self.add_weight( shape=(nl[3], nl[4]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[4])),trainable=True) 
        self.B6 = self.add_weight(shape=(nl[4],),initializer='zeros',trainable=True) 
        self.W7 = self.add_weight( shape=(nl[4], nl[5]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[5])),trainable=True) 
        self.B7 = self.add_weight(shape=(nl[5],),initializer='zeros',trainable=True) 
        self.W8 = self.add_weight( shape=(nl[5], nl[6]),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/nl[6])),trainable=True) 
        self.B8 = self.add_weight(shape=(nl[6],),initializer='zeros',trainable=True) 
        self.alpha = self.add_weight(shape=(1,),initializer='ones',trainable=True) 
        
        self.W9 = self.add_weight( shape=(nl[6], 1),initializer=tf.random_normal_initializer(mean=0.0,stddev = np.sqrt(1/1)),trainable=True)
        self.B9 = self.add_weight(shape=(1,),initializer='zeros',trainable=True) 
        
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
        
        self.Az2 = MLP()
        
        self.Az3 = MLP()
        
        # init networks
        test =tf.random.uniform([10,1],dtype=tf.float32)
        self.Az1(test,test)
        self.Az2(test,test)
        self.Az3(test,test)

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
        return self.Az1(x,y)*tf.cast((x**2+y**2 < 0.2**2),dtype=tf.float32)+self.Az2(x,y)*tf.cast((x**2+y**2 >= 0.2**2),dtype=tf.float32)
        # return 0
    # @tf.function
    def call_loss(self): 
        J0 = tf.constant(10.0,dtype = tf.float32)
        
        
        wpde = tf.constant(1.0,dtype = tf.float32)
        wbd = tf.constant(10.0,dtype = tf.float32)
        wmat = tf.constant(1.0,dtype = tf.float32)

        # boundary conditions
        Lbd = tf.reduce_mean(tf.square(self.Az1(self.PL2x,self.PL2y)))
        Lbd += tf.reduce_mean(tf.square(self.Az3(self.PL4x,self.PL4y)))
        # BD of tangential 0
        # employed on H        
        Lbd += tf.reduce_mean(tf.square( Dy(self.Az1,self.PL8x,self.PL8y)*self.T8x-Dx(self.Az1,self.PL8x,self.PL8y)*self.T8y ))
        Lbd += tf.reduce_mean(tf.square( Dy(self.Az2,self.PL3x,self.PL3y)*self.T3x-Dx(self.Az2,self.PL3x,self.PL3y)*self.T3y ))
        Lbd += tf.reduce_mean(tf.square( Dy(self.Az3,self.PL5x,self.PL5y)*self.T5x-Dx(self.Az3,self.PL5x,self.PL5y)*self.T5y ))
      
        
        
        #interfaces
        Li12 = tf.reduce_mean(tf.square(self.Az1(self.P12x,self.P12y)-self.Az2(self.P12x,self.P12y)))
        Bx1 = Dy(self.Az1,self.P12x,self.P12y)
        By1 = -Dx(self.Az1,self.P12x,self.P12y)
        Bx2 = Dy(self.Az2,self.P12x,self.P12y)
        By2 = -Dx(self.Az2,self.P12x,self.P12y)
        Li12 += tf.reduce_mean(tf.square(nu1(Bx1)*Bx1*self.T12x+nu1(By1)*By1*self.T12y  - (nu2(Bx2)*Bx2*self.T12x+nu2(By2)*By2*self.T12y) ))

        Li13 = tf.reduce_mean(tf.square(self.Az1(self.P13x,self.P13y)-self.Az3(self.P13x,self.P13y)))
        Bx1 = Dy(self.Az1,self.P13x,self.P13y)
        By1 = -Dx(self.Az1,self.P13x,self.P13y)
        Bx3 = Dy(self.Az3,self.P13x,self.P13y)
        By3 = -Dx(self.Az3,self.P13x,self.P13y)
        # Li13 += tf.reduce_mean(tf.square(self.Hx1(self.P13x,self.P13y)*self.T13x+self.Hy1(self.P13x,self.P13y)*self.T13y  - (self.Hx3(self.P13x,self.P13y)*self.T13x+self.Hy3(self.P13x,self.P13y)*self.T13y) ))
        Li13 += tf.reduce_mean(tf.square(nu1(Bx1)*Bx1*self.T13x+nu1(By1)*By1*self.T13y  - (nu3(Bx3)*Bx3*self.T13x+nu3(By3)*By3*self.T13y) ))

        Li23 = tf.reduce_mean(tf.square(self.Az2(self.P23x,self.P23y)-self.Az3(self.P23x,self.P23y)))
        Bx2 = Dy(self.Az2,self.P23x,self.P23y)
        By2 = -Dx(self.Az2,self.P23x,self.P23y)
        Bx3 = Dy(self.Az3,self.P23x,self.P23y)
        By3 = -Dx(self.Az3,self.P23x,self.P23y)
        # Li23 += tf.reduce_mean(tf.square(self.Hx2(self.P23x,self.P23y)*self.T23x+self.Hy2(self.P23x,self.P23y)*self.T23y  - (self.Hx3(self.P23x,self.P23y)*self.T23x+self.Hy3(self.P23x,self.P23y)*self.T23y) ))
        Li13 += tf.reduce_mean(tf.square(nu2(Bx2)*Bx2*self.T23x+nu2(By2)*By2*self.T23y  - (nu3(Bx3)*Bx3*self.T23x+nu3(By3)*By3*self.T23y) ))

        # Bx1 = Dy(self.Az1,self.P1x,self.P1y)
        # By1 = -Dx(self.Az1,self.P1x,self.P1y)
        # Bx2 = Dy(self.Az2,self.P1x,self.P1y)
        # By2 = -Dx(self.Az2,self.P1x,self.P1y)
        # Bx3 = Dy(self.Az3,self.P1x,self.P1y)
        # By3 = -Dx(self.Az3,self.P1x,self.P1y)
        

       
        
        # PDE 1
        L_pde1 =tf.reduce_mean(tf.square( curl_nu_curl_2d(self.Az1,nu1,self.P1x,self.P1y) ))
        L_pde2 =tf.reduce_mean(tf.square( curl_nu_curl_2d(self.Az2,nu2,self.P2x,self.P2y) - J0))
        L_pde3 =tf.reduce_mean(tf.square( curl_nu_curl_2d(self.Az3,nu3,self.P3x,self.P3y) ))
               
        Lpde = L_pde1+L_pde2+L_pde3
        
        Ltot = wpde*Lpde+wbd*(Lbd+Li12+Li13+Li23) # +wmat*L_material
        # print('000')
        print('loss %e, pde %e, bd %e'%(Ltot.numpy(),Lpde.numpy(),Lbd.numpy()))
        return Ltot

# parameters 
Nin = 20000
Nbd = 300

# points inside the domain
geo = Geometry(10)
Pts = np.random.rand(Nin,2)
P_D1 = Pts[geo.is_D1(Pts),:]
P_D2 = Pts[geo.is_D2(Pts),:]
P_D3 = Pts[geo.is_D3(Pts),:]
Nin_actual = P_D1.shape[0]+P_D2.shape[0]+P_D3.shape[0]

inputs = dict()
inputs['P1'] = tf.constant(P_D1,dtype=tf.float32)
inputs['P2'] = tf.constant(P_D2,dtype=tf.float32)
inputs['P3'] = tf.constant(P_D3,dtype=tf.float32)

P_L1 = geo.L1(np.random.rand(Nbd))
P_L2 = geo.L2(np.random.rand(Nbd))
P_L3 = geo.L3(np.random.rand(Nbd))
P_L4 = geo.L4(np.random.rand(Nbd))
P_L5 = geo.L5(np.random.rand(Nbd))
P_L6 = geo.L6(np.random.rand(Nbd))
P_L7 = geo.L7(np.random.rand(Nbd))
P_L8 = geo.L8(np.random.rand(Nbd))

inputs['L1'] = tf.constant(P_L1,dtype=tf.float32)
inputs['L2'] = tf.constant(P_L2,dtype=tf.float32)
inputs['L3'] = tf.constant(P_L3,dtype=tf.float32)
inputs['L4'] = tf.constant(P_L4,dtype=tf.float32)
inputs['L5'] = tf.constant(P_L5,dtype=tf.float32)
inputs['L6'] = tf.constant(P_L6,dtype=tf.float32)
inputs['L7'] = tf.constant(P_L7,dtype=tf.float32)
inputs['L8'] = tf.constant(P_L8,dtype=tf.float32)



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
opti_results = model.train_bfgs(num_iter=10000)
# opti_results = model.train_lbfgs(num_iter=5000)
# opti_results = model.train_backpropagation(50000)
# model.train_backpropagation(50000)
t_training = timeit.time.time() - t_training


#%% Visualization
x_mesh, y_mesh = np.meshgrid(np.linspace(0,0.75,400),np.linspace(0,0.55,400))
A1_comp = model.Az1(tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()
A2_comp = model.Az2(tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()
A3_comp = model.Az3(tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()


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
A_ref = call_fem(x_mesh.flatten(),y_mesh.flatten()).reshape(x_mesh.shape)

plt.figure()
plt.contourf(x_mesh,y_mesh,A_comp,levels=64)
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

# plot the B-field
x_mesh, y_mesh = np.meshgrid(np.linspace(0,0.75,40),np.linspace(0,0.55,40))
Bx_1 = Dy(model.Az1,tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()
Bx_2 = Dy(model.Az2,tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()
Bx_3 = Dy(model.Az3,tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()
By_1 = -Dx(model.Az1,tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()
By_2 = -Dx(model.Az2,tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()
By_3 = -Dx(model.Az3,tf.constant(x_mesh.reshape([-1,1]),dtype=tf.float32),tf.constant(y_mesh.reshape([-1,1]),dtype=tf.float32)).numpy()

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

err_max = np.nanmax(np.abs(A_comp-A_ref).flatten())
err_max_rel = np.nanmax(np.abs(A_comp-A_ref).flatten()) / np.nanmax(np.abs(A_ref).flatten())
err_rms = np.sqrt(np.nanmean((A_comp-A_ref).flatten()**2))

print('Max error     ',err_max)
print('Max rel error ',err_max_rel)
print('RMS error     ',err_rms)

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

# x_mesh = tf.constant(np.linspace(0,1,1000).reshape([-1,1]),dtype=tf.float32)
# res = model.call(x_mesh,0*x_mesh).numpy()              
# res = Dx(model.call,x_mesh,0*x_mesh).numpy() 
# plt.figure()                   
# plt.plot(x_mesh.numpy(),res)