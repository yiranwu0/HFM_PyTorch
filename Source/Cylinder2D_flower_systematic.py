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
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions
    
    def __init__(self, data, c_data, eqns, layers, Pec, Rey):
        # specs & flow properties
        self.layers = layers
        self.Pec = Pec
        self.Rey = Rey
        # data
        [self.data, self.c_data, self.eqns] = [data, c_data, eqns]
        # dat manager util
        self.dm = dataManager(using_visdom, version)
        # physics "uninformed" neural networks
        self.net = neural_net(self.layers, data, USE_CUDA, device)
        if(USE_CUDA):
            self.net.to(device)
            
    
    def compute_loss(self, batch_size, it):
        idx_data = np.random.choice(N_data, min(batch_size, N_data))
        idx_eqns = np.random.choice(N_eqns, batch_size)
        
        # wrap with Variable, might be sent to GPU
        data_batch = Variable(torch.from_numpy(self.data[idx_data,:]).float(), requires_grad = True)
        c_data_batch = Variable(torch.from_numpy(self.c_data[idx_data,:]).float(), requires_grad = True)
        eqns_batch = Variable(torch.from_numpy(self.eqns[idx_eqns,:]).float(), requires_grad = True)

        # predict and split
        [c_data_pred,_,_,_] = torch.split(self.net(data_batch), 1,1)
        [c_eqns_pred, u_eqns_pred, v_eqns_pred, p_eqns_pred] = torch.split(self.net(eqns_batch), 1,1)
        [e1_eqns_pred, e2_eqns_pred, e3_eqns_pred, e4_eqns_pred] = \
            Navier_Stokes_2D(c_eqns_pred, u_eqns_pred, v_eqns_pred, p_eqns_pred,
                             eqns_batch, self.Pec, self.Rey)
        
        # get loss
        c_loss = mean_squared_error(c_data_pred, c_data_batch)
        e1_loss = mean_squared_error(e1_eqns_pred, torch.zeros_like(e1_eqns_pred))
        e2_loss = mean_squared_error(e2_eqns_pred, torch.zeros_like(e1_eqns_pred))
        e3_loss = mean_squared_error(e3_eqns_pred, torch.zeros_like(e1_eqns_pred))
        e4_loss = mean_squared_error(e4_eqns_pred, torch.zeros_like(e1_eqns_pred))
        loss = c_loss + (e1_loss + e2_loss + e3_loss + e4_loss)
        
        # update datamanger and return
        self.dm.update(c_loss, e1_loss, e2_loss, e3_loss, e4_loss, loss, it)
        return loss


    def train(self, total_time, batch_size, lr, T_star, X_star, Y_star, C_star, U_star, V_star, P_star):
        optimizer = Adam(self.net.parameters(), lr)
        
        start_time = time.time()
        running_time = 0
        it = 0
        min_loss = 1
        while running_time < total_time:
            optimizer.zero_grad()
            loss = self.compute_loss(batch_size, it)
            loss.backward()
            optimizer.step()
            
            if(loss < 1e-2 and loss < min_loss):
                min_loss = float(loss)
                torch.save(self.net.state_dict(), "../Results/model_min_loss_" + version + ".pth")
    
            if it % 10 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh'
                      %(it, loss, elapsed, running_time))
                sys.stdout.flush()
                start_time = time.time()

            if it % 100 == 0:
                self.get_error(T_star, X_star, Y_star, C_star, U_star, V_star, P_star, it)
                
            if it % 10000 == 0:
                torch.save(self.net.state_dict(), "../Results/model_updating_" + version + ".pth")
                self.dm.saveData()
            
            it += 1
            
        # try to save traing error after training is finished
        torch.save(self.net.state_dict(), "../Results/model_40_hours_" + version + ".pth")
        self.dm.saveData()
        
    def get_error(self, T_star, X_star, Y_star, C_star, U_star, V_star, P_star, it):
        
        snap = np.array([random.randint(0, 200)])
        t_test = T_star[:,snap]
        x_test = X_star[:,snap]
        y_test = Y_star[:,snap]
        
        c_test = C_star[:,snap]
        u_test = U_star[:,snap]
        v_test = V_star[:,snap]
        p_test = P_star[:,snap]
        
        # Prediction
        c_pred, u_pred, v_pred, p_pred = self.predict(t_test, x_test, y_test)
        
        # Error
        error_c = relative_error(c_pred, torch.from_numpy(c_test).float())
        error_u = relative_error(u_pred, torch.from_numpy(u_test).float())
        error_v = relative_error(v_pred, torch.from_numpy(v_test).float())
        error_p = relative_error(p_pred - torch.mean(p_pred), torch.from_numpy(p_test - np.mean(p_test)).float())

        self.dm.update_error(error_c, error_u, error_v, error_p, it)
        print('Error: c: %e, u: %e, v: %e, p: %e' % (error_c,error_u, error_v, error_p))

        
    def predict(self, t_star, x_star, y_star):
        t = torch.from_numpy(t_star).float().to(device).detach()
        x = torch.from_numpy(x_star).float().to(device).detach()
        y = torch.from_numpy(y_star).float().to(device).detach()
        data = torch.cat((t, x, y), 1)
        out = self.net(data).cpu()
        [c_star, u_star, v_star, p_star] = torch.split(out,1,1)
        return c_star, u_star, v_star, p_star







######################################################################
######################################################################
######################## Args Handling ###############################
######################################################################

version = "v5"
print(version)

# system in
T_data = int(sys.argv[1])
N_data = int(sys.argv[2])
device_num = sys.argv[3]
using_visdom = False
if(len(sys.argv) == 5):
    using_visdom = (sys.argv[4] == "visdom")

# model parameters
batch_size = 10000
layers = [3] + 10*[4*50] + [4]
lr = 1e-3
traing_time = 40
    
# If cuda is available, use cuda
USE_CUDA = torch.cuda.is_available()
print("Using cuda" + str(device_num)) if USE_CUDA else print("Not using cuda.")
device = torch.device('cuda:' + device_num)
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(device) if USE_CUDA else autograd.Variable(*args, **kwargs)


######################################################################
######################## Process Data  ###############################
######################################################################
# Load Data
print("Start Processing Data.")
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
    

# T_data -> system in
# N_data
    
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
    
#  c_data = Variable(torch.from_numpy(c_data).float(), requires_grad = False)
#  data = Variable(torch.from_numpy(np.concatenate((t_data, x_data, y_data), 1)).float(), requires_grad = True)
#  eqns = Variable(torch.from_numpy(np.concatenate((t_eqns, x_eqns, y_eqns), 1)).float(), requires_grad = True)
    
    
print("Data processed. Start training.")
    
#################################################################
################# Get Model and train ###########################
#################################################################
model = HFM(data, c_data, eqns, layers, Pec = 100, Rey = 100)
model.train(traing_time, batch_size, lr, T_star, X_star, Y_star, C_star, U_star, V_star, P_star)
    
#################################################################
##################### Save Predictions  #########################
#################################################################
print("Training finished. Test and Save Data.")
    
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
 
