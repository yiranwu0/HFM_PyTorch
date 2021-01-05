import numpy as np
from visdom import Visdom
from scipy.io import savemat
import torch

class dataManager(object):
    
    def __init__(self, using_visdom, version):
        self.using_visdom = using_visdom
        self.version = version
        
        self.c_loss = np.zeros(0)
        self.e1_loss = np.zeros(0)
        self.e2_loss = np.zeros(0)
        self.e3_loss = np.zeros(0)
        self.e4_loss = np.zeros(0)
        self.total_loss = np.zeros(0)
            
        self.error_c = np.zeros(0)
        self.error_u = np.zeros(0)
        self.error_v = np.zeros(0)
        self.error_p = np.zeros(0)
        
        if(using_visdom):
            self.vis = Visdom()
            self.total_loss_plot = self.newPlot(0,0, "MSE Total Loss", "Total Loss")
            self.c_loss_plot = self.newPlot(0,0, "MSE C pred Loss", "C Loss")
            self.e1_loss_plot = self.newPlot(0,0, "MSE e1 Loss", "e1 Loss")
            self.e2_loss_plot = self.newPlot(0,0, "MSE e2 Loss", "e2 Loss")
            self.e3_loss_plot = self.newPlot(0,0, "MSE e3 Loss", "e3 Loss")
            self.e4_loss_plot = self.newPlot(0,0, "MSE e4 Loss", "e4 Loss")
            
            self.error_c_plot = self.newPlot(0,0, "R2 Error C", "C Error")
            self.error_u_plot = self.newPlot(0,0, "R2 Error U", "U Error")
            self.error_v_plot = self.newPlot(0,0, "R2 Error V", "V Error")
            self.error_p_plot = self.newPlot(0,0, "R2 Error P", "P Error")

    
    def newPlot(self, x, y, title, ylabel):
        return self.vis.line(np.array([y]),np.array([x]),
            opts=dict(title = title,xlabel = "Iterations",ylable = ylabel))


    def update(self, c_loss, e1_loss, e2_loss, e3_loss, e4_loss, loss, it):
        c_loss = torch.unsqueeze(c_loss.detach().cpu(), 0)
        e1_loss = torch.unsqueeze(e1_loss.detach().cpu(), 0)
        e2_loss = torch.unsqueeze(e2_loss.detach().cpu(), 0)
        e3_loss = torch.unsqueeze(e3_loss.detach().cpu(), 0)
        e4_loss = torch.unsqueeze(e4_loss.detach().cpu(), 0)
        loss = torch.unsqueeze(loss.cpu(), 0)
        
        self.update_loss(c_loss, e1_loss, e2_loss, e3_loss, e4_loss, loss)
        if(self.using_visdom):
            self.update_plot(c_loss, e1_loss, e2_loss, e3_loss, e4_loss, loss, it)
            
        
    def update_plot(self, c_loss, e1_loss, e2_loss, e3_loss, e4_loss, loss, it):
        self.vis.line(c_loss, np.array([it]), self.c_loss_plot, update="append")
        self.vis.line(e1_loss, np.array([it]), self.e1_loss_plot, update="append")
        self.vis.line(e2_loss, np.array([it]), self.e2_loss_plot, update="append")
        self.vis.line(e3_loss, np.array([it]), self.e3_loss_plot, update="append")
        self.vis.line(e4_loss, np.array([it]), self.e4_loss_plot, update="append")
        self.vis.line(loss, np.array([it]), self.total_loss_plot, update="append")
    
    
    def update_loss(self, c_loss, e1_loss, e2_loss, e3_loss, e4_loss, loss):
        self.c_loss = np.append(self.c_loss, c_loss.detach().numpy())
        self.e1_loss = np.append(self.e1_loss, e1_loss.numpy())
        self.e2_loss = np.append(self.e2_loss, e2_loss.numpy())
        self.e3_loss = np.append(self.e3_loss, e3_loss.numpy())
        self.e4_loss = np.append(self.e4_loss, e4_loss.numpy())
        self.total_loss = np.append(self.total_loss, loss.detach().numpy(), axis = 0)
    

    def update_error(self, error_c, error_u, error_v, error_p, it):
        error_c = torch.unsqueeze(error_c.detach().cpu(), 0)
        error_u = torch.unsqueeze(error_u.detach().cpu(), 0)
        error_v = torch.unsqueeze(error_v.detach().cpu(), 0)
        error_p = torch.unsqueeze(error_p.detach().cpu(), 0)
        
        self.error_c = np.append(self.error_c, error_c.detach().numpy())
        self.error_u = np.append(self.error_u, error_u.detach().numpy())
        self.error_v = np.append(self.error_v, error_v.detach().numpy())
        self.error_p = np.append(self.error_p, error_p.detach().numpy())
            
        if(self.using_visdom):
            self.vis.line(error_c, np.array([it]), self.error_c_plot, update="append")
            self.vis.line(error_u, np.array([it]), self.error_u_plot, update="append")
            self.vis.line(error_v, np.array([it]), self.error_v_plot, update="append")
            self.vis.line(error_p, np.array([it]), self.error_p_plot, update="append")

        
    def saveData(self):
        print("Saving training error.")
        if(not self.using_visdom):
            savemat('../Results/Cylinder2D_flower_training_error_' + self.version + '.mat',
                     {'c_loss':self.c_loss, 'e1_loss':self.e1_loss, 'e2_loss':self.e2_loss, 'e3_loss':self.e3_loss, 'e4_loss':self.e4_loss, 'total_loss':self.total_loss,
                        'error_c':self.error_c, 'error_u':self.error_u, 'error_v':self.error_v, 'error_p':self.error_p})
        
