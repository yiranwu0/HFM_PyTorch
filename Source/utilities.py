import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, layer_dim):
        super().__init__()
        
        temp = []
        for l in range(1, len(layer_dim)):
            temp.append(nn.Linear(layer_dim[l-1], layer_dim[l]))
        self.layers = nn.ModuleList(temp)
        print(self.layers)
        sys.stdout.flush()
        
    def forward(self, x):
        for l in self.layers:
            x = swish(l(x))
        return x


def Navier_Stokes_2D(c, u, v, p, txy, Pec, Rey):

    # gradients w.r.t each output and all inputs
    c_txy = fwd_gradients(c, txy)
    u_txy = fwd_gradients(c, txy)
    v_txy = fwd_gradients(c, txy)
    p_txy = fwd_gradients(c, txy)
    
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



# check
