# import torch as tn
# import functorch 
# import torch.nn as nn
# import numpy as np
# 
# 
# def gradient(func):
#     J = functorch.vmap(functorch.jacfwd(func))
#     return lambda x: J(x)[...,0,:]
#     
# def divergence(func):
#     J = functorch.vmap(functorch.jacfwd(func))
#     return lambda x: tn.sum(tn.diagonal(J(x),dim1=1,dim2=2),1, keepdims=True)
# 
# def laplace(func):
#     H = functorch.vmap(functorch.hessian(func))
#     return lambda x: tn.sum(tn.diagonal(H(x)[...,0,:,:],dim1=1,dim2=2),1, keepdims=True)
# 
# 
# model = nn.Sequential(nn.Linear(2,22),nn.Sigmoid(),nn.Linear(22,32),nn.Sigmoid(),nn.Linear(32,32),nn.Sigmoid(), nn.Linear(32,2))
# 
# 
# Phi_ref = lambda x:  tn.exp(2*x[:,0])*tn.sin(2*x[:,1])/8
# Rhs = lambda x: 0
# N_in = 5000
# N_bd = 500
# pts_inside = tn.tensor(np.random.rand(N_in,2)*2-1).to(dtype=tn.float32)
# 
# tmp1 = np.random.randint(0, high=2, size=(N_bd,1))
# tmp2 = np.random.randint(0, high=2, size=(N_bd,1))
# xbd_train = (np.random.rand(N_bd,1)*2-1)*tmp1 + (1 - tmp1)*(tmp2*(-1)+(1-tmp2)*1)
# tmp2 = np.random.randint(0, high=2, size=(N_bd,1))
# ybd_train = (np.random.rand(N_bd,1)*2-1)*(1-tmp1) + tmp1*(tmp2*(-1)+(1-tmp2)*1)
# pts_bd = tn.tensor(np.concatenate((xbd_train[:],ybd_train[:]),1)).to(dtype=tn.float32)
# bd_vals = Phi_ref(pts_bd).to(dtype=tn.float32)
# 
# @tn.jit.script
# def loss(pts_inside, pts_bd, bd_vals):
#     lbd = tn.sum((model(pts_bd)[:,0]-bd_vals)**2)
#     lpde = tn.sum(laplace(model)(pts_inside)**2)
#     return lbd+0.1*lpde
# 
# 
# N_iterations = 4000
# # trainer 
# optimizer = tn.optim.LBFGS(model.parameters(), max_iter=N_iterations)
# 
# 
# def loss_closure():
#     optimizer.zero_grad()
#     lv = loss(pts_inside, pts_bd, bd_vals)
#     lv.backward()
#     return lv
#     
# for i in range(N_iterations):
#     
# 
# 
#     optimizer.step(loss_closure)
#     
#     print('iteration ',i,' loss ',loss(pts_inside, pts_bd, bd_vals))
#     
#     
#     
# 
import jax
import jax.numpy as jnp
import numpy as np

class PINN():
    
    def __init__(self):
         
        pass
    
    def train(self, method = 'ADAM'):
        
        pass
        
        