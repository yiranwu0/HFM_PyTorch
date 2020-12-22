import numpy as np
import math
import scipy.io
import time
import sys
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable

from utilities import neural_net, Navier_Stokes_2D, \
                       mean_squared_error, relative_error

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
        # physics "uninformed" neural networks
        self.net = neural_net(self.layers, data)
        
    def compute_loss(self, batch_size):
        N_data = self.data.shape[0]
        N_eqns = self.eqns.shape[0]
        
        idx_data = np.random.choice(N_data, min(batch_size, N_data))
        idx_eqns = np.random.choice(N_eqns, batch_size)
        
        (data_batch, c_data_batch, eqns_batch) = (self.data[idx_data,:],
                                                  self.c_data[idx_data,:],
                                                  self.eqns[idx_eqns,:])

        # predict and split
        [c_data_pred,_,_,_] = torch.split(self.net(data_batch), 1,1)
        
        [c_eqns_pred, u_eqns_pred, v_eqns_pred, p_eqns_pred] = torch.split(self.net(eqns_batch), 1,1)
        
        [e1_eqns_pred, e2_eqns_pred, e3_eqns_pred, e4_eqns_pred] = \
            Navier_Stokes_2D(c_eqns_pred, u_eqns_pred, v_eqns_pred, p_eqns_pred,
                             eqns_batch, self.Pec, self.Rey)
        
        # loss
        loss = mean_squared_error(c_data_pred, c_data_batch)  + \
                mean_squared_error(e1_eqns_pred, torch.zeros_like(e1_eqns_pred)) + \
                mean_squared_error(e2_eqns_pred, torch.zeros_like(e1_eqns_pred)) + \
                mean_squared_error(e3_eqns_pred, torch.zeros_like(e1_eqns_pred)) + \
                mean_squared_error(e4_eqns_pred, torch.zeros_like(e1_eqns_pred))
        
        return loss


        
    def train(self, epochs, batch_size, lr):
        optimizer = Adam(self.net.parameters(), lr)
        
        for i in range(epochs):
            optimizer.zero_grad()
            loss = self.compute_loss(batch_size)
            loss.backward(retain_graph=True)
            optimizer.step()
            print("Epochs: " + str(i) + "    Loss: " + str(loss))

    
    def predict(self, t_star, x_star, y_star):
        t = torch.from_numpy(t_star).float()
        x = torch.from_numpy(x_star).float()
        y = torch.from_numpy(y_star).float()
        data = torch.cat((t, x, y), 1)
        
        [c_star, u_star, v_star, p_star] = torch.split(self.net(data),1,1)
        return c_star, u_star, v_star, p_star


def main():
    ######################################################################
    ######################## Process Data  ###############################
    ######################################################################
    
    # Load Data
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
    
    T_data = int(sys.argv[1])
    N_data = int(sys.argv[2])
    
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
    
    t_data = Variable(torch.from_numpy(t_data).float(), requires_grad = True)
    x_data = Variable(torch.from_numpy(x_data).float(), requires_grad = True)
    y_data = Variable(torch.from_numpy(y_data).float(), requires_grad = True)
    c_data = Variable(torch.from_numpy(c_data).float(), requires_grad = False)
    
    t_eqns = Variable(torch.from_numpy(t_eqns).float(), requires_grad = True)
    x_eqns = Variable(torch.from_numpy(x_eqns).float(), requires_grad = True)
    y_eqns = Variable(torch.from_numpy(y_eqns).float(), requires_grad = True)
    
    data = torch.cat((t_data, x_data, y_data), 1)
    eqns = torch.cat((t_eqns, x_eqns, y_eqns), 1)
    
    print("Data processed. Start training.")
    
    #################################################################
    ################# Get Model and train ###########################
    #################################################################
    # model parameters
    batch_size = 10000
    layers = [3] + 10*[4*50] + [4]
    lr = 1e-3
    epochs = 100
    
    model = HFM(data, c_data, eqns, layers, Pec = 100, Rey = 100)
    model.train(epochs, batch_size, lr)
    
    print("Training Finshed. Get an example test error.")
    # Single Test Data
    snap = np.array([100])
    t_test = T_star[:,snap]
    x_test = X_star[:,snap]
    y_test = Y_star[:,snap]
    
    c_test = C_star[:,snap]
    u_test = U_star[:,snap]
    v_test = V_star[:,snap]
    p_test = P_star[:,snap]
    
    # Prediction
    c_pred, u_pred, v_pred, p_pred = model.predict(t_test, x_test, y_test)
    
    # Error
    error_c = relative_error(c_pred, torch.from_numpy(c_test).float())
    error_u = relative_error(u_pred, torch.from_numpy(u_test).float())
    error_v = relative_error(v_pred, torch.from_numpy(v_test).float())
    error_p = relative_error(p_pred - torch.mean(p_pred), torch.from_numpy(p_test - np.mean(p_test)).float())

    print('Error c: %e' % (error_c))
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))
    
    
    ################# Save Data ###########################
    print("Test and Save Data.")
    
    C_pred = 0*C_star
    U_pred = 0*U_star
    V_pred = 0*V_star
    P_pred = 0*P_star
    print(t_star.shape[0])
    for snap in range(0, 10):
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
    
        print('Error c: %e' % (error_c))
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error p: %e' % (error_p))
    
    scipy.io.savemat('../Results/Cylinder2D_flower_results_%d_%d_%s.mat' %(T_data, N_data, time.strftime('%d_%m_%Y')),
                     {'C_pred':C_pred, 'U_pred':U_pred, 'V_pred':V_pred, 'P_pred':P_pred})
 

if __name__ == "__main__":
    main()
