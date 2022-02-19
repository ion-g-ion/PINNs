import torch as tn
import funtorch 

def jacobianBatch(f, x):
  return funtorch.vmap(funtorch.jacrev(f))(x)

tn.autograd.grad()
def div2d(func, input):
    J = tn.autograd.functional.jacobian(func,input)
    return  J[...,0]+J[...,1]

class PINN():
    
    def __init__(self):
        pass
    def __call__(self,inputs):
        pass
    
    
def func(x):
    return tn.cat((x[...,0,None]**3,2*tn.sin(x[...,1,None])),-1)

def ref(x):
    return 3*x[...,0]**2+2*tn.cos(x[...,1])

x = tn.rand((10,20,2))

f = func(x)
divf = div2d(func,x)
divf_ref = ref(x)

print(divf-divf_ref)