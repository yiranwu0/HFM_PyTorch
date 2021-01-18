import numpy as np
import math
import scipy.io
import sys
import random
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.autograd as autograd
# self defined pack
from utilities import neural_net, Navier_Stokes_2D, mean_squared_error, relative_error
from dataManager import *


class HFM(object):
    
    def __init__(self, data, c_data, eqns, layers, Pec, Rey):
        # specs & flow properties
        self.Pec = Pec
        self.Rey = Rey
        # data
        [self.data, self.c_data, self.eqns] = [data, c_data, eqns]

        # physics "uninformed" neural networks
        self.net = neural_net(layers, data, USE_CUDA, device)
        self.net.load_state_dict(torch.load(name, map_location=device))
        print("Previous state loaded.")
        if(USE_CUDA):
            self.net.to(device)
            
        
    def predict(self, t_star, x_star, y_star):
        t = torch.from_numpy(t_star).float().to(device).detach()
        x = torch.from_numpy(x_star).float().to(device).detach()
        y = torch.from_numpy(y_star).float().to(device).detach()
        data = torch.cat((t, x, y), 1)
        out = self.net(data).cpu()
        [c_star, u_star, v_star, p_star] = torch.split(out,1,1)
        return c_star, u_star, v_star, p_star


version = sys.argv[1]
device_num = sys.argv[2] if len(sys.argv) >= 3 else 0

layers = [3] + 10*[4*50] + [4]
name = "../Results/model_updating_" + version + ".pth"

# If cuda is available, use cuda
USE_CUDA = torch.cuda.is_available()
print("Using cuda " + str(device_num)) if USE_CUDA else print("Not using cuda.")
device = torch.device('cuda:' + device_num) if USE_CUDA else torch.device('cpu')
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device) if USE_CUDA else autograd.Variable(*args, **kwargs)

T_data = 201
N_data = 15000

data = scipy.io.loadmat('../Data/Cylinder2D_flower.mat')
    
t_star = data['t_star'] # T x 1
x_star = data['x_star'] # N x 1
y_star = data['y_star'] # N x 1
    
T = t_star.shape[0]
N = x_star.shape[0]
        
U_star = data['U_star'] # N x T
V_star = data['V_star'] # N x T
P_star = data['P_star'] # N x T
C_star = data['C_star'] # N x T
    
# Rearrange Data
T_star = np.tile(t_star, (1,N)).T # N x T
X_star = np.tile(x_star, (1,T)) # N x T
Y_star = np.tile(y_star, (1,T)) # N x T
    
idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
idx_x = np.random.choice(N, N_data, replace=False)
t_data = T_star[:, idx_t][idx_x,:].flatten()[:,None]
x_data = X_star[:, idx_t][idx_x,:].flatten()[:,None]
y_data = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
c_data = C_star[:, idx_t][idx_x,:].flatten()[:,None]
        
T_eqns = T
N_eqns = N
idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
idx_x = np.random.choice(N, N_eqns, replace=False)
t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    
data = np.concatenate((t_data, x_data, y_data), 1)
eqns = np.concatenate((t_eqns, x_eqns, y_eqns), 1)
    
    
model = HFM(data, c_data, eqns, layers, Pec = 100, Rey = 100)

C_pred = 0*C_star
U_pred = 0*U_star
V_pred = 0*V_star
P_pred = 0*P_star

for snap in range(0, t_star.shape[0]):
    t_test = T_star[:,snap:snap+1]
    x_test = X_star[:,snap:snap+1]
    y_test = Y_star[:,snap:snap+1]
        
    c_test = C_star[:,snap:snap+1]
    u_test = U_star[:,snap:snap+1]
    v_test = V_star[:,snap:snap+1]
    p_test = P_star[:,snap:snap+1]
    
    # Prediction
    c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
        
    C_pred[:,snap:snap+1] = c_pred.detach().numpy()
    U_pred[:,snap:snap+1] = u_pred.detach().numpy()
    V_pred[:,snap:snap+1] = v_pred.detach().numpy()
    P_pred[:,snap:snap+1] = p_pred.detach().numpy()
    
    # Error
    error_c = relative_error(c_pred, torch.from_numpy(c_test).float())
    error_u = relative_error(u_pred, torch.from_numpy(u_test).float())
    error_v = relative_error(v_pred, torch.from_numpy(v_test).float())
    error_p = relative_error(p_pred - torch.mean(p_pred), torch.from_numpy(p_test - np.mean(p_test)).float())
    
    print('Error: c: %e, u: %e, v: %e, p: %e' % (error_c,error_u, error_v, error_p))
    
scipy.io.savemat(('../Results/Cylinder2D_flower_results_%d_%d_%s_' + version + '.mat') %(T_data, N_data, time.strftime('%d_%m_%Y')),
                     {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred, 'P_pred':P_pred})
 
