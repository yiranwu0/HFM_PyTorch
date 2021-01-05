import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable

font = {'size'   : 22}
matplotlib.rc('font', **font)

def nnan_count(a):
    k=0
    
    for i in range(len(a[0,:])):
        for j in range(len(a[:,0])):
            if(np.isnan(a[i,j])):
                k = k+1
                
    return len(a[0,:])* len(a[:,0]) - k

# X_star -> x,y
def plot_surface_griddata_flower(X_star, u_star, tit, ax):
    print("get " + tit + 'plot')
    n_x = 250
    n_y = 250
    
    # get x, y point index -> Xplot, Yplot
    x_l = np.min(X_star[:,0])
    x_r = np.max(X_star[:,0])

    y_l = np.min(X_star[:,1])
    y_r = np.max(X_star[:,1])
    
    x = np.linspace(x_l, x_r, n_x)
    y = np.linspace(y_l, y_r, n_y)

    [Xplot, Yplot] = np.meshgrid(x,y)
    
    # get c value -> Uplot
    idx_x_r = np.round((X_star[:,0]-x_l)/(x_r - x_l)*(n_x-1),0)
    idx_x_f = np.floor((X_star[:,0]-x_l)/(x_r - x_l)*(n_x-1))
    idx_x_c = np.ceil((X_star[:,0]-x_l)/(x_r - x_l)*(n_x-1))
    idx_x = np.concatenate((idx_x_r, idx_x_f, idx_x_c), axis = 0)
    
    idx_y_r = np.round((X_star[:,1]-y_l)/(y_r - y_l)*(n_y-1),0)
    idx_y_f = np.floor((X_star[:,1]-y_l)/(y_r - y_l)*(n_y-1))
    idx_y_c = np.ceil((X_star[:,1]-y_l)/(y_r - y_l)*(n_y-1))
    idx_y = np.concatenate((idx_y_r, idx_y_f, idx_y_c), axis = 0)
 
    idx = np.unique(np.concatenate((idx_x.reshape(len(idx_x), 1), idx_y.reshape(len(idx_y),1)), axis = 1), axis=0)
    idx = idx.astype(int)
    
    flags = Xplot * np.nan

    for i in range(0, np.size(idx, 0)):
        flags[idx[i,1], idx[i,0]] = 1 # to add 1 or not
    
    
    Uplot = griddata(X_star, u_star, (Xplot,Yplot))
    
    # plot heap map
    H = ax.pcolormesh(Xplot, Yplot, (flags*Uplot), shading='gouraud', cmap = 'jet')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    fig.colorbar(H, cax=cax, orientation='vertical')
    
    ax.set_xlabel('$x$',fontsize=20)
    ax.set_ylabel('$y$',fontsize=20)
    ax.set_title(tit,fontsize=30)
    ax.axis('equal')

    # plot circle
    cir = plt.Circle((0,0),0.5, edgecolor = 'k',facecolor = 'w')
    ax.add_artist(cir)
    lim = ax.get_xlim()
    ax.set_xlim(-0.6, lim[1]+0.1)
    lim = ax.get_ylim()
    ax.set_ylim(lim[0]-0.1, lim[1]+0.1)
    
    
    
data = scipy.io.loadmat("../Data/Cylinder2D_flower.mat")
name = "Cylinder2D_flower_results_201_15000_01_01_2021_v1"
print("Ploting " + name)
results = scipy.io.loadmat("../Results/" + name + ".mat")

t_star = np.array(data['t_star']) # T x 1
x_star = np.array(data['x_star']) # N x 1
y_star = np.array(data['y_star']) # N x 1
   
C_star = np.array(data['C_star']) # N x T
U_star = np.array(data['U_star']) # N x T
V_star = np.array(data['V_star']) # N x T
P_star = np.array(data['P_star']) # N x T

C_pred = np.array(results['C_pred']) # N x T
U_pred = np.array(results['U_pred']) # N x T
V_pred = np.array(results['V_pred']) # N x T
P_pred = np.array(results['P_pred']) # N x T

xy_star = np.concatenate((x_star,y_star), axis = 1)

###########################################################################
######################## plot training results ############################
###########################################################################
plt.clf
fig, ax = plt.subplots(2, 4, figsize= [40, 20])

for num in range(0, 1):
    print(num)
    plt.clf
    axes = plt.gca()
    
    # c(t,x,y)
    plot_surface_griddata_flower(xy_star, C_star[:,num],'Reference $c(t,x,y)$', ax[0,0])
    plot_surface_griddata_flower(xy_star, C_pred[:,num],'Regressed $c(t,x,y)$', ax[0,1])
    
    # u(t,x,y)
    plot_surface_griddata_flower(xy_star, U_star[:,num],'Reference $u(t,x,y)$', ax[0,2])
    plot_surface_griddata_flower(xy_star, U_pred[:,num],'Regressed $u(t,x,y)$', ax[0,3])
    
    # v(t,x,y)
    plot_surface_griddata_flower(xy_star, V_star[:,num],'Reference $v(t,x,y)$', ax[1,0])
    plot_surface_griddata_flower(xy_star, V_pred[:,num],'Regressed $v(t,x,y)$', ax[1,1])

    # p(t,x,y)
    plot_surface_griddata_flower(xy_star, P_star[:,num],'Reference $p(t,x,y)$', ax[1,2])
    plot_surface_griddata_flower(xy_star, P_pred[:,num],'Regressed $p(t,x,y)$', ax[1,3])

    fig.tight_layout()

plt.savefig("../Results/" + name + "_plot.png")


#############################################################################
#################### plot relative error u,v,w,p ############################
#############################################################################

def relative_error(pred, exact):
    return np.sqrt(mean_squared_error(pred, exact)/np.mean(np.square(exact - np.mean(exact))))
    
def mean_squared_error(pred, exact):
    return np.mean(np.square(pred - exact))

def plot_relative_error(ax, title, t_star, error):
    ax.plot(t_star, error)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Rel. $L_2$ Error')
    ax.set_title(title)
    
n_star = t_star.shape[0]
t_star = t_star + 132.08

print("Getting relative error...")
(errors_c,errors_u,errors_v,errors_p) = (np.zeros((n_star,1)),np.zeros((n_star,1)),np.zeros((n_star,1)),np.zeros((n_star,1)))

for num in range(n_star):
    errors_c[num] = relative_error(C_pred[:,num], C_star[:,num])
    errors_u[num] = relative_error(U_pred[:,num], U_star[:,num])
    errors_v[num] = relative_error(V_pred[:,num], V_star[:,num])
    errors_p[num] = relative_error(P_pred[:,num] - np.mean(P_pred[:,num]), P_star[:,num] - np.mean(P_star[:,num]))

fig, ax = plt.subplots(2,2, figsize = [10, 10])

plot_relative_error(ax[0,0], "$c(t,x,y)$", t_star, errors_c)
plot_relative_error(ax[0,1], "$u(t,x,y)$", t_star, errors_u)
plot_relative_error(ax[1,0], "$v(t,x,y)$", t_star, errors_v)
plot_relative_error(ax[1,1], "$p(t,x,y)$", t_star, errors_p)

fig.tight_layout()

plt.savefig("../Results/" + name + "_relative_error_plot.png")


#############################################################################
####################### plot training error #################################
#############################################################################
# utilities
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def semilogy_color(x, ax):
    l = ax.semilogy(running_mean(x, 500))
    return l[0].get_color()

def plot_error(ax, title, loss, c):
    l = ax.semilogy(running_mean(loss, 500))
    l[0].set_color(c)
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)
    ax.set_xlabel('Iteration',fontsize=20)
    ax.set_ylabel('Mean Squared Error',fontsize=20)
    ax.set_title(title,fontsize=30)
    ax.xaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)

name = "Cylinder2D_flower_training_error_v4"
print("Getting training error of " + name + "...")
#load data

errors = scipy.io.loadmat("../Results/" + name + ".mat")
total_loss = np.array(errors['total_loss'])
c_loss = np.array(errors['c_loss'])
e1_loss = np.array(errors['e1_loss'])
e2_loss = np.array(errors['e2_loss'])
e3_loss = np.array(errors['e3_loss'])
e4_loss = np.array(errors['e4_loss'])

# get plot
plt.clf
fig, ax = plt.subplots(2, 4, figsize = [40, 20])

# subplot 1: plot all losses in the same plot
ax[0,0].set_title("Losses",fontsize=30)
ax[0,0].set_xlabel('Iteration',fontsize=20)
ax[0,0].set_ylabel('Mean Squared Error',fontsize=20)
ax[0,0].xaxis.label.set_size(20)
ax[0,0].yaxis.label.set_size(20)
c1 = semilogy_color(total_loss, ax[0,0])
c2 = semilogy_color(c_loss , ax[0,0])
c3 = semilogy_color(e1_loss, ax[0,0])
c4 = semilogy_color(e2_loss, ax[0,0])
c5 = semilogy_color(e3_loss, ax[0,0])
c6 = semilogy_color(e4_loss, ax[0,0])

# plot each error
plot_error(ax[0,1], "Total Loss", total_loss, c1)
plot_error(ax[0,2], 'Loss $c(t,x,y)$',  c_loss, c2)
plot_error(ax[0,3], 'Loss $e1(t,x,y)$', e1_loss, c3)
plot_error(ax[1,0], 'Loss $e2(t,x,y)$', e2_loss, c4)
plot_error(ax[1,1], 'Loss $e3(t,x,y)$', e3_loss, c5)
plot_error(ax[1,2], 'Loss $e4(t,x,y)$', e4_loss, c6)

fig.tight_layout()


plt.savefig("../Results/" + name + "_plot.png")


