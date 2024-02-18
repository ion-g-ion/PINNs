import numpy as np 
import jax.numpy as jnp 
import jax 

def quadratic(a: jax.Array, b: jax.Array, c: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Solve the quadratic equation adn return only the real roots.
        `a x^2 + b x + c = 0`
    Args:
        a (jax.Array): coefficients.
        b (jax.Array): coefficients.
        c (jax.Array): coefficients.

    Returns:
        tuple[jax.Array, jax.Array]: _description_
    """
    delta = b*b-4*a*c
    x1 = (-b+jnp.sqrt(delta))/(2*a)
    x2 = (-b-jnp.sqrt(delta))/(2*a)

    return jnp.where(delta>=0, x1, np.infty), jnp.where(delta>=0, x2, np.infty)
    
def cubic(a: jax.Array,b: jax.Array,c: jax.Array, d: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Get all REAL roots of the polynomial `ax^3+bx^2+cx+d=0`. Cmplex are replaced by np.infty.

    Args:
        a (jax.Array): _description_
        b (jax.Array): _description_
        c (jax.Array): _description_
        d (jax.Array): _description_

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: _description_
    """
    D0 = b*b-3*a*c
    D1 = 2*b**3-9*a*b*c+27*a**2*d
    
    p = (-D0)/(3*a*a)
    q = (D1)/(27*a**3)
    delta = 4*p**3+27*q*q
    sq = jnp.where(p<=0, jnp.sqrt(-p/3), np.infty)
    acos = jnp.arccos(3*q/(2*p)/sq)
    
    delta_quad = c*c-4*b*d 
    x1_quad = jnp.where(delta_quad>=0, (-c+jnp.sqrt(delta_quad))/(2*b), np.infty)
    x2_quad = jnp.where(delta_quad>=0, (-c-jnp.sqrt(delta_quad))/(2*b), np.infty)
    
    p0 = -jnp.sign(q)*(q**(1/3)) 
    
    t0 = 2*sq*jnp.cos(1/3*acos-2*np.pi*0/3)
    t1 = 2*sq*jnp.cos(1/3*acos-2*np.pi*1/3)
    t2 = 2*sq*jnp.cos(1/3*acos-2*np.pi*2/3)
    Q = (3*(c/a)-(b/a)**2)/9
    C = -0.5*q*(3/jnp.abs(p))**1.5 # 3*Q
    t_single = 2*jnp.sqrt(jnp.abs(p)/3)*jnp.where(p>0, jnp.sinh(1/3*jnp.arcsinh(C)),jnp.where(jnp.abs(C)<1, jnp.cos(1/3*jnp.arccos(C)), jnp.sign(C)*jnp.cosh(1/3*jnp.arccosh(jnp.abs(C)))))
    t_single = jnp.where(p==0, p0+b/(3*a), t_single) 
    
    t0 = jnp.where(jnp.logical_and(p!=0.0, delta<0), t0, t_single)
    t1 = jnp.where(jnp.logical_and(p!=0.0, delta<0), t1, np.infty)
    t2 = jnp.where(jnp.logical_and(p!=0.0, delta<0), t2, np.infty)

    r0 = jnp.where(a!=0.0, t0-b/(3*a), np.infty)
    r1 = jnp.where(a!=0.0, t1-b/(3*a), x1_quad)
    r2 = jnp.where(a!=0.0, t2-b/(3*a), x2_quad)
    
    return r0, r1, r2

    # n = -b**3/27/a**3 + b*c/6/a**2 - d/2/a
    # s = n**2 + (c/3/a - b**2/9/a**2)**3
    # NMS = n-s**0.5
    # NPS = n+s**0.5
    
    # r0 = jnp.where(jnp.logical_and(NMS >=0, NPS>=0),  NMS**(1/3)+NPS**(1/3) - b/3/a, np.infty)  
    # r1 = jnp.where(NPS>=0,  NPS**(1/3)+NPS**(1/3) - b/3/a, np.infty)  
    # r2 = jnp.where(NMS >=0,  NMS**(1/3)+NMS**(1/3) - b/3/a, np.infty)  
    
    # #r0 = (n-s)**(1/3)+(n+s)**(1/3) - b/3/a
    # #r1 = (n+s)**(1/3)+(n+s)**(1/3) - b/3/a
    # #r2 = (n-s)**(1/3)+(n-s)**(1/3) - b/3/a
    # return (r0,r1,r2)

def distance_to_bezier_curve_simple(pts: jax.Array, control_pts: jax.Array, deg: int, de: int) -> tuple[jax.Array, jax.Array, jax.Array]:

    n = pts.shape[0]
    
    if deg == 1:
        a = jnp.sum((control_pts[1,:] - control_pts[0,:])**2)
        b = jnp.sum( (control_pts[0,:] - pts) * (control_pts[1,:]-control_pts[0,:]) , axis = 1)
        
        t_mins = -b/a
        t_mins = jnp.where(t_mins <= 0, 0.0, t_mins)
        t_mins = jnp.where(t_mins >= 1, 1.0, t_mins)
        
        ps = jnp.einsum('i,k->ik', (1-t_mins), control_pts[0,:]) + jnp.einsum('i,k->ik', t_mins, control_pts[1,:])
        d_min = jnp.linalg.norm(ps-pts, axis=1)
        
    elif deg == 2:
        ak =    control_pts[0,:] - 2*control_pts[1,:] + control_pts[2,:]
        bk = -2*control_pts[0,:] + 2*control_pts[1,:]
        ck =    control_pts[0,:] -   pts
        A = jnp.sum(2*ak*ak)
        B = jnp.sum(3*ak*bk)
        C = jnp.sum(bk*bk+2*ak*ck, axis=1)
        D = jnp.sum(bk*ck, axis=1)
        
        t1,t2,t3 = cubic(A,B,C,D)
        t0 =       jnp.zeros_like(D)
        t4 =       jnp.ones_like(D)

        t1 = jnp.where(jnp.logical_and(t1>=0,t1<=1), t1, 0.0)
        t2 = jnp.where(jnp.logical_and(t2>=0,t2<=1), t2, 0.0)
        t3 = jnp.where(jnp.logical_and(t3>=0,t3<=1), t3, 0.0)

        pts_fun = lambda t: jnp.einsum('i,k->ik', (1-t)**2, control_pts[0,:]) + jnp.einsum('i,k->ik', (1-t)*t*2, control_pts[1,:]) + jnp.einsum('i,k->ik', t**2, control_pts[2,:]) 
        
        ps0 = pts_fun(t0).T
        ps1 = pts_fun(t1).T
        ps2 = pts_fun(t2).T
        ps3 = pts_fun(t3).T
        ps4 = pts_fun(t4).T

        d0 = jnp.linalg.norm(pts-ps0.T, axis=1)
        d1 = jnp.linalg.norm(pts-ps1.T, axis=1)
        d2 = jnp.linalg.norm(pts-ps2.T, axis=1)
        d3 = jnp.linalg.norm(pts-ps3.T, axis=1)
        d4 = jnp.linalg.norm(pts-ps4.T, axis=1)
        
        t_mins = jnp.where(d0<=d1, t0, t1)
        ps =     jnp.where(d0<=d1, ps0, ps1)
        d_min =  jnp.where(d0<=d1, d0, d1)

        t_mins = jnp.where(d_min<=d2, t_mins, t2)
        ps =     jnp.where(d_min<=d2, ps, ps2)
        d_min =  jnp.where(d_min<=d2, d_min, d2)

        t_mins = jnp.where(d_min<=d3, t_mins, t3)
        ps =     jnp.where(d_min<=d3, ps, ps3)
        d_min =  jnp.where(d_min<=d3, d_min, d3)

        t_mins = jnp.where(d_min<=d4, t_mins, t4)
        ps =     jnp.where(d_min<=d4, ps, ps4)
        d_min =  jnp.where(d_min<=d4, d_min, d4)
        
        ps = ps.T
    else: 
        raise NotImplementedError
    
    return t_mins, d_min, ps 

def distance_to_bezier_surface_simple(pts: jax.Array, control_pts: jax.Array, deg: int, de: int) -> tuple[jax.Array, jax.Array, jax.Array]:

    n = pts.shape[0]
    
    if deg == 1:
        # we have d(u,v) = \sum_k (P_11k-x_k + u(P_21k-P_11k) + v(P_12k-P_11k) + uv(P_11k-P_12k-P_21k+P_22k))^2
        ak = (control_pts[0,0,:] - pts)
        bk = control_pts[1,0,:] - control_pts[0,0,:]
        ck = control_pts[0,1,:] - control_pts[0,0,:]
        dk = control_pts[1,1,:] + control_pts[0,0,:] - control_pts[0,1,:] - control_pts[1,0,:]
        
        A = jnp.sum(ak*bk, axis=1)
        B = jnp.sum(bk*bk)
        C = jnp.sum(ak*dk+bk*ck, axis=1)
        D = jnp.sum(2*dk*bk)
        E = jnp.sum(ck*dk)

        AA = jnp.sum(ak*ck, axis=1)
        CC = jnp.sum(ck*ck)
        BB = jnp.sum(ak*dk+bk*ck, axis=1)
        DD = jnp.sum(2*dk*ck)
        EE = jnp.sum(bk*dk)

        v1, v2 = quadratic(D*CC-DD*CC, AA*D+BB*C+B*CC-AA*A,AA*B-BB*A)
        
        v1 = jnp.where(v1>=0, v1, 0)
        v1 = jnp.where(v1<=1, v1, 1)
        v2 = jnp.where(v2>=0, v2, 0)
        v2 = jnp.where(v2<=1, v2, 1)

        u1 = -(A+C*v1)/(B+D*v1)
        u2 = -(A+C*v2)/(B+D*v2)

        u1 = jnp.where(u1>=0, u1, 0)
        u1 = jnp.where(u1<=1, u1, 1)
        u2 = jnp.where(u2>=0, u2, 0)
        u2 = jnp.where(u2<=1, u2, 1)
        
        ps1 = jnp.einsum('k,i->ik',bk,u1)+jnp.einsum('k,i->ik',ck,v1)+jnp.einsum('k,i->ik',dk,u1*v1)+control_pts[0,0,:]
        ps2 = jnp.einsum('k,i->ik',bk,u2)+jnp.einsum('k,i->ik',ck,v2)+jnp.einsum('k,i->ik',dk,u2*v2)+control_pts[0,0,:]

        d1 = jnp.linalg.norm(pts-ps1, axis=1)
        d2 = jnp.linalg.norm(pts-ps2, axis=1)


        #    u12
        #  u11  u21
        #    u22

        v11, d11, ps11 = distance_to_bezier_curve_simple(pts, control_pts[0,:,:], 1, de)
        u11 = v11*0.0
        u12, d12, ps12 = distance_to_bezier_curve_simple(pts, control_pts[:,1,:], 1, de)
        v12 = u12*0.0+1.0
        v21, d21, ps21 = distance_to_bezier_curve_simple(pts, control_pts[1,:,:], 1, de)
        u21 = v21*0.0+1.0
        u22, d22, ps22 = distance_to_bezier_curve_simple(pts, control_pts[:,0,:], 1, de)
        v22 = u22*0.0

        u_min = jnp.where(d1<d2, u1,u2)
        v_min = jnp.where(d1<d2, v1,v2)
        ps =     jnp.where(d1<d2, ps1.T, ps2.T).T
        d_min = jnp.where(d1<d2, d1, d2)
        
        u_min = jnp.where(d_min<d11, u_min,u11)
        v_min = jnp.where(d_min<d11, v_min,v11)
        ps =     jnp.where(d_min<d11, ps.T, ps11.T).T
        d_min = jnp.where(d_min<d11, d_min, d11)

        u_min = jnp.where(d_min<d12, u_min,u12)
        v_min = jnp.where(d_min<d12, v_min,v12)
        ps =     jnp.where(d_min<d12, ps.T, ps12.T).T
        d_min = jnp.where(d_min<d12, d_min, d12)

        u_min = jnp.where(d_min<d21, u_min,u21)
        v_min = jnp.where(d_min<d21, v_min,v21)
        ps =     jnp.where(d_min<d21, ps.T, ps21.T).T
        d_min = jnp.where(d_min<d21, d_min, d21)
        
        u_min = jnp.where(d_min<d22, u_min,u22)
        v_min = jnp.where(d_min<d22, v_min,v22)
        ps =     jnp.where(d_min<d22, ps.T, ps22.T).T
        d_min = jnp.where(d_min<d22, d_min, d22)
        #u_min = 0.5*(u12+u22)
        #v_min = 0.5*(u11+u21)
        #ps = jnp.einsum('k,i->ik',bk,u_min)+jnp.einsum('k,i->ik',ck,v_min)+jnp.einsum('k,i->ik',dk,u_min*v_min)+control_pts[0,0,:]
        #d_min = jnp.linalg.norm(pts-ps1, axis=1)
    else: 
        raise NotImplementedError

    return u_min, v_min, d_min, ps 