import numpy as np
import torch.nn as nn
import sys
import torch
import torch.nn.utils.weight_norm as weight_norm

# modified by Yiran Wu @ Dec 22, 2020
# Caution: inputs of functions such as neural_net, Navier_Stokes_2D, Gradient_Velocity_2D, etc are changed

# original inputs (c, u, v, p, t, x, y) are now  (c, u, v, p, txy) for 2D functions
# (c, u, v, w, p, t, x, y, z) are now  (c, u, v, w, p, txyz) for 3D functions
# that is, input X are always a combined set instead of separate inputs
# This change is due to the reason that torch.autograd.grad will not accept separate inputs

loss = nn.MSELoss()

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt( mean_squared_error(pred, exact)/np.mean(np.square(exact - np.mean(exact))))
    
    return torch.sqrt( loss(pred, exact) / torch.mean(torch.square(exact - torch.mean(exact))) )

def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return loss(pred, exact)

def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, dummy, create_graph=True)[0]
    return G

def swish(x):
    return x * torch.sigmoid(x)

class neural_net(nn.Module):
    def __init__(self, layer_dim, X):
        super().__init__()
        
        self.X_mean = torch.mean(X, 0, True)
        self.X_std = torch.std(X, 0, True)
        
        temp = []
        for l in range(1, len(layer_dim)):
            temp.append(weight_norm(nn.Linear(layer_dim[l-1], layer_dim[l]), dim = 0))
        self.layers = nn.ModuleList(temp)
        print(self.layers)
        sys.stdout.flush()
        
    def forward(self, x):
        x = (x - self.X_mean) / self.X_std # z-score norm
        for l in self.layers:
            x = swish(l(x))
        return x


def Navier_Stokes_2D(c, u, v, p, txy, Pec, Rey):

    # gradients w.r.t each output and all inputs
    c_txy = fwd_gradients(c, txy)
    u_txy = fwd_gradients(u, txy)
    v_txy = fwd_gradients(v, txy)
    p_txy = fwd_gradients(p, txy)
    
    #---wanted
   # print(c_txy)
    c_t = c_txy[:,0:1]
    c_x = c_txy[:,1:2]
    c_y = c_txy[:,2:3]
    
    u_t = u_txy[:,0:1]
    u_x = u_txy[:,1:2]
    u_y = u_txy[:,2:3]
    
    v_t = v_txy[:,0:1]
    v_x = v_txy[:,1:2]
    v_y = v_txy[:,2:3]
    
    p_x = p_txy[:,1:2]
    p_y = p_txy[:,2:3]
    #wanted----
    
    # second gradient
   # print(c_x)
   # print(txy)
    c_x_txy = fwd_gradients(c_x, txy)
    c_y_txy = fwd_gradients(c_y, txy)
    c_xx = c_x_txy[:,1:2] #wanted
    c_yy = c_y_txy[:,2:3] #wanted
    
    u_x_txy = fwd_gradients(u_x, txy)
    u_y_txy = fwd_gradients(u_y, txy)
    u_xx = u_x_txy[:,1:2] #wanted
    u_yy = u_y_txy[:,2:3] #wanted
    
    v_x_txy = fwd_gradients(v_x, txy)
    v_y_txy = fwd_gradients(v_y, txy)
    v_xx = v_x_txy[:,1:2] #wanted
    v_yy = v_y_txy[:,2:3] #wanted
    
    e1 = c_t + (u*c_x + v*c_y) - (1.0/Pec)*(c_xx + c_yy)
    e2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Rey)*(u_xx + u_yy)
    e3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Rey)*(v_xx + v_yy)
    e4 = u_x + v_y
    
    return e1, e2, e3, e4



# can be done: improved version to return [u_x, v_x, u_y, v_y] directly from Navier_Stokes_2D
# instead of call fwd_gradients again
def Gradient_Velocity_2D(u, v, txy):
    
    u_txy = fwd_gradients(u, txy)
    v_txy = fwd_gradients(v, txy)
    
    u_x = u_txy[:,1:2]
    u_y = u_txy[:,2:3]
    
    v_x = v_txy[:,1:2]
    v_y = v_txy[:,2:3]
    
    return [u_x, v_x, u_y, v_y]


def Strain_Rate_2D(u, v, txy):
    
    [u_x, v_x, u_y, v_y] = Gradient_Velocity_2D(u, v, txy)
    
    eps11dot = u_x
    eps12dot = 0.5*(v_x + u_y)
    eps22dot = v_y
    
    return [eps11dot, eps12dot, eps22dot]


def Navier_Stokes_3D(c, u, v, w, p, txyz, Pec, Rey):
    
    # gradients w.r.t each output and all inputs
    c_txyz = fwd_gradients(c, txyz)
    u_txyz = fwd_gradients(u, txyz)
    v_txyz = fwd_gradients(v, txyz)
    w_txyz = fwd_gradients(w, txyz)
    p_txyz = fwd_gradients(p, txyz)
    
    c_t = c_txyz[:,0:1]
    c_x = c_txyz[:,1:2]
    c_y = c_txyz[:,2:3]
    c_z = c_txyz[:,3:4]
    
    u_t = u_txyz[:,0:1]
    u_x = u_txyz[:,1:2]
    u_y = u_txyz[:,2:3]
    u_z = u_txyz[:,3:4]
                           
    v_t = v_txyz[:,0:1]
    v_x = v_txyz[:,1:2]
    v_y = v_txyz[:,2:3]
    v_z = v_txyz[:,3:4]
    
    w_t = w_txyz[:,0:1]
    w_x = w_txyz[:,1:2]
    w_y = w_txyz[:,2:3]
    w_z = w_txyz[:,3:4]
                           
    p_x = p_txyz[:,1:2]
    p_y = p_txyz[:,2:3]
    p_z = p_txyz[:,3:4]

    # second gradient
    
    c_x_txyz = fwd_gradients(c_x, txyz)
    c_y_txyz = fwd_gradients(c_y, txyz)
    c_z_txyz = fwd_gradients(c_z, txyz)
    c_xx = c_x_txyz[:,1:2] #wanted
    c_yy = c_y_txyz[:,2:3] #wanted
    c_zz = c_z_txyz[:,3:4] #wanted
                           
                
    u_x_txyz = fwd_gradients(u_x, txyz)
    u_y_txyz = fwd_gradients(u_y, txyz)
    u_z_txyz = fwd_gradients(u_z, txyz)
    u_xx = u_x_txyz[:,1:2] #wanted
    u_yy = u_y_txyz[:,2:3] #wanted
    u_zz = u_z_txyz[:,3:4] #wanted
    
    v_x_txyz = fwd_gradients(v_x, txyz)
    v_y_txyz = fwd_gradients(v_y, txyz)
    v_z_txyz = fwd_gradients(v_z, txyz)
    v_xx = v_x_txyz[:,1:2] #wanted
    v_yy = v_y_txyz[:,2:3] #wanted
    v_zz = v_z_txyz[:,3:4] #wanted
                           
    w_x_txyz = fwd_gradients(w_x, txyz)
    w_y_txyz = fwd_gradients(w_y, txyz)
    w_z_txyz = fwd_gradients(w_z, txyz)
    w_xx = w_x_txyz[:,1:2] #wanted
    w_yy = w_y_txyz[:,2:3] #wanted
    w_zz = w_z_txyz[:,3:4] #wanted
    
    e1 = c_t + (u*c_x + v*c_y + w*c_z) - (1.0/Pec)*(c_xx + c_yy + c_zz)
    e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/Rey)*(u_xx + u_yy + u_zz)
    e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/Rey)*(v_xx + v_yy + v_zz)
    e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/Rey)*(w_xx + w_yy + w_zz)
    e5 = u_x + v_y + w_z
    
    return e1, e2, e3, e4, e5

def Gradient_Velocity_3D(u, v, w, txyz):
        
    u_txy = fwd_gradients(u, txyz)
    v_txy = fwd_gradients(v, txyz)
    w_txy = fwd_gradients(w, txyz)
    
    u_x = u_txyz[:,1:2]
    u_y = u_txyz[:,2:3]
    u_z = u_txyz[:,3:4]
    
    v_x = v_txyz[:,1:2]
    v_y = v_txyz[:,2:3]
    v_z = v_txyz[:,3:4]
    
    w_x = w_txyz[:,1:2]
    w_y = w_txyz[:,2:3]
    w_z = w_txyz[:,3:4]
    
    return [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z]

def Shear_Stress_3D(u, v, w, txyz, nx, ny, nz, Rey):
        
    [u_x, v_x, w_x, u_y, v_y, w_y, u_z, v_z, w_z] = Gradient_Velocity_3D(u, v, w, txyz)

    uu = u_x + u_x
    uv = u_y + v_x
    uw = u_z + w_x
    vv = v_y + v_y
    vw = v_z + w_y
    ww = w_z + w_z
    
    sx = (uu*nx + uv*ny + uw*nz)/Rey
    sy = (uv*nx + vv*ny + vw*nz)/Rey
    sz = (uw*nx + vw*ny + ww*nz)/Rey
    
    return sx, sy, sz
