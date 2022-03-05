import pinns
import pinns.geometry
import pinns.bspline
import jax 
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

key = jax.random.PRNGKey(0)

geom = pinns.geometry.PatchParametrized(lambda uv: jnp.concatenate( ( ((1+uv[:,1])*jnp.sin(uv[:,0]*jnp.pi/2))[:,None] , ((1+uv[:,1])*jnp.cos(uv[:,0]*jnp.pi/2))[:,None] ),-1 ), 2, key)

xs = geom.sample_inside(2560)
xb1 = geom.sample_boundary(0,0,100)
xb2 = geom.sample_boundary(0,1,100)
xb3 = geom.sample_boundary(1,0,100)
xb4 = geom.sample_boundary(1,1,100)

plt.figure()
plt.scatter(xs[:,0], xs[:,1], s = 1, c = 'b')
plt.scatter(xb1[:,0], xb1[:,1],s=1)
plt.scatter(xb2[:,0], xb2[:,1],s=1)
plt.scatter(xb3[:,0], xb3[:,1],s=1)
plt.scatter(xb4[:,0], xb4[:,1],s=1)

b1 = pinns.bspline.BSplineBasis(np.linspace(0,1,4),2)
b2 = pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)

knots = np.concatenate( ( np.array([[0,1],[0,1],[0,1],[2,2],[3,3]])[...,None] , np.array([[0,0],[1,1],[3,2],[3,2],[3,2]])[...,None] ),-1)
weights = np.array([[1,1],[1,1],[1/np.sqrt(2),1/np.sqrt(2)],[1,1],[1,1]])
geom2 = pinns.geometry.PatchNURBS([b1, b2], knots, weights, key)

xs = geom2(np.random.rand(1000,2))
xb,xbt = geom2.sample_boundary(1,0,50)
xbn = pinns.geometry.tangent2normal_2d(xbt)

plt.figure()
plt.scatter(xs[:,0],xs[:,1], s = 1)
plt.scatter(xb[:,0],xb[:,1], s = 1)
plt.quiver(xb[:,0],xb[:,1],xbt[:,0,0],xbt[:,0,1])
plt.quiver(xb[:,0],xb[:,1],xbn[:,0,0],xbn[:,0,1])

R = 1
r = 0.2

knots = np.array([ [[R,0],[r,0]] , [[R,R],[r,r]] , [[0,R],[0,r]]  ])
weights = np.array([[1,1],[1/np.sqrt(2),1/np.sqrt(2)],[1,1]])

geom3 = pinns.geometry.PatchNURBS([pinns.bspline.BSplineBasis(np.linspace(0,1,2),2), pinns.bspline.BSplineBasis(np.linspace(0,1,2),1)],knots,weights,key)

pts, ws = geom3.importance_sampling_2d(100000)

plt.figure()
plt.scatter(pts[:,0],pts[:,1],s=1)
plt.scatter(knots[:,:,0][:],knots[:,:,1][:])

S = np.pi*R*R/4-np.pi*r*r/4
Sc = np.sum(ws)/ws.size

Ia = np.pi*(R-r)/2
Ic = np.sum((1/np.sqrt(pts[:,0]**2+pts[:,1]**2))*ws)/ws.size