#!/usr/bin/env python
# coding: utf-8

import os, sys, time, portalocker
sys.stdout.reconfigure(line_buffering=True) #for tee

# === SETTINGS === 
paper_case = 3
#case 1: full dataset, PHY trained at beginning
#case 2: full dataset, PHY not trained at beginning
#case 3: 1350:-1350, PHY trained at beginning
emulate_exp_dir = 'exp_rc1_0.0_100'

from_exp_dir = False
really_from_exp_dir = 'exp_' in os.getcwd()
if really_from_exp_dir or emulate_exp_dir:
    from_exp_dir = True
    if really_from_exp_dir:
        print('niced to '+str(os.nice(10))+', waiting for BoundedSemaphore...')
        portbs = portalocker.BoundedSemaphore(4, directory='..') #By default, 4 processes can run in parallel in the case of the hyperparameter tuning in tmux. You can increase or decrease this number.
        while True:
            try:
                portlo = portbs.acquire()
                break
            except:
                sys.stdout.write('W')
                time.sleep(3)
                continue
        print('acquired.')

    if really_from_exp_dir: cwd = os.getcwd()
    else: 
        print('Emulating case: '+emulate_exp_dir)
        cwd = emulate_exp_dir

    the_expid = cwd.split('/')[-1].split('exp_')[1]
    if len(the_expid)<3:
        the_expnum = int(float(the_expid))
        exp_real_data = bool(the_expnum&1)
        exp_model_counter = (the_expnum>>1)&3
        exp_obj_counter = (the_expnum>>3)&3
    else:
        assert the_expid[0] in 'rs', 'the_expid[0] should be either "r" or "s"'
        exp_real_data = the_expid[0]=='r'
        exp_model_counter = int(the_expid[1],36) #0,1,2,3,...,9,a,b,c,d,e,f,g...
        exp_obj_counter = int(the_expid[2],36)
        exp_split = the_expid.split('_')                                                                                                                                    
        if len(exp_split)>=2:                                                                                                                                               
            exp_weight_decay = float(exp_split[1])                                                                                                                          
            exp_nn_size = int(exp_split[2])     
    print(f"Experiment description: exp_real_data({exp_real_data}), exp_model_counter({exp_model_counter}), exp_obj_counter({exp_obj_counter})")
    if exp_real_data: exp_real_data_explanation = 'real data'
    else: exp_real_data_explanation = 'simulated data'
    if exp_model_counter == 0: exp_model_counter_explanation =    '0 | 0.b. ANN only'
    elif exp_model_counter == 1: exp_model_counter_explanation =  '1 | 1.   ANN + PHY fixed ~tau_f'
    elif exp_model_counter == 2: exp_model_counter_explanation =  '2 |      ANN + PHY optimized ~tau_f'
    elif exp_model_counter == 3: exp_model_counter_explanation =  '3 |      PHY optimized tau_f' #not for simulation data
    elif exp_model_counter == 4: exp_model_counter_explanation =  '4 |      ANN(PHY fixed) tau_f' #not for simulation data
    elif exp_model_counter == 5: exp_model_counter_explanation =  '5 | 2.   ANN(PHY fixed) ~tau_f'
    elif exp_model_counter == 6: exp_model_counter_explanation =  '6 |      ANN + PHY fixed tau_f' #not for simulation data
    elif exp_model_counter == 7: exp_model_counter_explanation =  '7 |      ANN + PHY optimized tau_f' #not for simulation data
    elif exp_model_counter == 8: exp_model_counter_explanation =  '8 |      ANN(PHY fixed parts Minv) tau_f' #not for simulation data
    elif exp_model_counter == 9: exp_model_counter_explanation =  '9 |      ANN(PHY fixed parts ~Minv) tau_f' #not for simulation data
    elif exp_model_counter == 10: exp_model_counter_explanation = 'a | 3.a. ANN(PHY fixed parts Minv) ~tau_f' 
    elif exp_model_counter == 11: exp_model_counter_explanation = 'b | 3.b. ANN(PHY fixed parts ~Minv) ~tau_f'
    elif exp_model_counter == 12: exp_model_counter_explanation = 'c | 4.   PHY+M^-1[ANN_i(v_i)] ~tau_f'
    elif exp_model_counter == 13: exp_model_counter_explanation = 'd | 5.   PHY+M^-1[ANN_i(v_i x)] ~tau_f'
    elif exp_model_counter == 14: exp_model_counter_explanation = 'e |      PHY+M^-1[ANN_i(v_i)+ANN] ~tau_f'
    elif exp_model_counter == 15: exp_model_counter_explanation = 'f |      PHY+M^-1[ANN_i(v_i)]+ANN ~tau_f'
    if not exp_real_data: assert exp_model_counter not in [3,4,6,7,8,9]

    if exp_obj_counter == 0: exp_obj_counter_explanation = 'N-step-NRMS obj+NsN val'
    elif exp_obj_counter == 1: exp_obj_counter_explanation = 'FFT obj+FFT val'
    elif exp_obj_counter == 2: exp_obj_counter_explanation = 'FFT phaseshift obj+FFT val'
    elif exp_obj_counter == 3: exp_obj_counter_explanation = 'FFT phaseshift obj+NsN val'
    print(' - '+exp_real_data_explanation)
    print(' - '+exp_model_counter_explanation)
    print(' - '+exp_obj_counter_explanation)
    print(' - exp_nn_size: '+str(exp_nn_size))
    print(' - exp_weight_decay: '+str(exp_weight_decay))
    print(' - paper_case: '+str(paper_case))
    if really_from_exp_dir:
        print('Waiting 3 seconds to start...')
        time.sleep(3)

stiction_pad = 1350

very_debug = True if not from_exp_dir else False
epoch_patience_min = 5000 if paper_case != 2 else 0

os.system('rm epoch_*.html')

from copy import copy, deepcopy
import pdb
import deepSI
from deepSI import System_data, System_data_list
from time import time
import numpy as np
from matplotlib import pyplot as plt
import torch
torch.manual_seed(9)
torch.autograd.set_detect_anomaly(False) #this slows things down, only for debugging
from torch import optim, nn, tensor
from tqdm.auto import tqdm
import matplotlib
from scipy.io import loadmat, savemat
from pprint import pprint
import sympybotconv_M_code
import sympybotconv_C_code
import sympybotconv_g_code
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


#import casadi
import sys
import json
import math
import os
import signal

cuda_on = False
tgt_device = 'cpu' if not cuda_on else 'cuda' #TGT
assert not cuda_on or torch.cuda.is_available(), 'CUDA is not available'

#on USR1, we print where we are
import signal
import traceback
signal.signal(signal.SIGUSR2, lambda sig, stack: traceback.print_stack(stack))
signal.signal(signal.SIGUSR1, lambda sig, frame: pdb.Pdb().set_trace(frame)) #pdb is nice but then we cannot stop the optimization with ctrl+c...
#os.environ["PYTORCH_JIT_LOG_LEVEL"] = ">>autodiff:>>>profiling_graph_executor_impl:>>>create_autodiff_subgraphs:>>>subgraph_utils"
matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams['figure.figsize'] = [10, 20]

print('<DIFF>\\last commit:')
os.system('git rev-parse HEAD; git diff | cat') #this goes into the log, if there is one
print('</DIFF>')

##                _     _
##  _ __  ___  __| |___| |
## | '  \/ _ \/ _` / -_) |
## |_|_|_\___/\__,_\___|_|

def printgrad(x,msg):
    #print('grad at '+str(msg)+' is: '+str(x))
    print(str(msg)+';'+(';'.join([str(y) for y in x.cpu().detach().numpy()])))


class F_robust_nn(nn.Module):
    """
    This is the combined model, i.e. f_rhs(x,v,tau)=[v f_fwd(x,v,tau)]. 
    """
    def __init__(self, nx, nu, params, normalization):
        from deepSI.utils import simple_res_net
        super(F_robust_nn, self).__init__()
        assert nx == 12
        assert nu == 6
        self.robot = F_robot_sympybotics(nx,nu,params)
        self.CONF_SERIAL_ACC = 0
        self.CONF_PARALLEL_1SAMP = 1
        self.CONF_SERIAL_ACC_ALLIN = 2
        self.CONF_SERIAL_ACC_ALLIN_NOFRIC = 3
        self.CONF_FRICNN_V = 4
        self.CONF_FRICNN_V_P = 5
        self.CONF_FRICNN_ANN = 7
        self.CONF_FRICNN_ANN2 = 8

        #NN settings
        if not from_exp_dir: self.nn_configuration = self.CONF_FRICNN_V_P #else based on exp_model_counter
        else:
            if exp_model_counter in [4,5]: 
                self.nn_configuration = self.CONF_SERIAL_ACC 
            elif exp_model_counter in [8,9]: 
                self.nn_configuration = self.CONF_SERIAL_ACC_ALLIN 
            elif exp_model_counter in [10,11]:
                self.nn_configuration = self.CONF_SERIAL_ACC_ALLIN_NOFRIC
            elif exp_model_counter == 12:
                self.nn_configuration = self.CONF_FRICNN_V
            elif exp_model_counter == 13:
                self.nn_configuration = self.CONF_FRICNN_V_P
            elif exp_model_counter == 14:
                self.nn_configuration = self.CONF_FRICNN_ANN
            elif exp_model_counter == 15:
                self.nn_configuration = self.CONF_FRICNN_ANN2
            else: 
                self.nn_configuration = self.CONF_PARALLEL_1SAMP 
        self.robustify_nn = True if not from_exp_dir else not exp_model_counter==3  #TUNEIT #IMPORTANT

        self.disable_Minv = False if not from_exp_dir else exp_model_counter in [9,11]
        if self.nn_configuration == self.CONF_SERIAL_ACC:
            N_net_in = 24 #acc
        elif self.nn_configuration == self.CONF_PARALLEL_1SAMP:
            N_net_in = 18 #pos, vel, tau
        elif self.nn_configuration == self.CONF_SERIAL_ACC_ALLIN:
            N_net_in = 33+18 if not self.disable_Minv else 28+18 
        elif self.nn_configuration == self.CONF_SERIAL_ACC_ALLIN_NOFRIC:
            N_net_in = 33 if not self.disable_Minv else 28
        elif self.nn_configuration == self.CONF_FRICNN_V:
            N_net_in = 1
        elif self.nn_configuration == self.CONF_FRICNN_V_P:
            N_net_in = 7
        elif self.nn_configuration == self.CONF_FRICNN_ANN:
            N_net_in = 1
            N_net_in_2 = 18
        elif self.nn_configuration == self.CONF_FRICNN_ANN2:
            N_net_in = 1
            N_net_in_2 = 18
        else: #this is needed because of torchscript
            N_net_in = 0
        N_nodes_per_layer = 100 if not from_exp_dir else exp_nn_size
        if self.nn_configuration in [self.CONF_FRICNN_V, self.CONF_FRICNN_V_P, self.CONF_FRICNN_ANN, self.CONF_FRICNN_ANN2]:
            N_net_out = 1
            self.friction_nns = []
            for i in range(6):
                self.friction_nns.append(simple_res_net(n_in=N_net_in, n_out=N_net_out, n_nodes_per_layer=N_nodes_per_layer, n_hidden_layers=2, activation=nn.Tanh))
                setattr(self,'friction_nn_'+str(i),self.friction_nns[-1]) #add to the object so that torch / deepSI can discover it to optimize
            if self.nn_configuration in [self.CONF_FRICNN_ANN, self.CONF_FRICNN_ANN2]:
                N_net_out_2 = 6
                self.nn = simple_res_net(n_in=N_net_in_2, n_out=N_net_out_2, n_nodes_per_layer=N_nodes_per_layer, n_hidden_layers=2, activation=nn.Tanh)
        else:
            N_net_out = 6
            self.nn = simple_res_net(n_in=N_net_in, n_out=N_net_out, n_nodes_per_layer=N_nodes_per_layer, n_hidden_layers=2, activation=nn.Tanh)

        self.u_std = normalization['u_std'].to(tgt_device) if normalization is not None else torch.ones(6).to(tgt_device)
        self.u_mean = normalization['u_mean'].to(tgt_device) if normalization is not None else torch.zeros(6).to(tgt_device)
        self.pos_std = normalization['pos_std'].to(tgt_device) if normalization is not None else torch.ones(6).to(tgt_device)
        self.pos_mean = normalization['pos_mean'].to(tgt_device) if normalization is not None else torch.zeros(6).to(tgt_device)
        self.v_std = normalization['v_std'].to(tgt_device) if normalization is not None else torch.ones(6).to(tgt_device)
        self.v_mean = normalization['v_mean'].to(tgt_device) if normalization is not None else torch.zeros(6).to(tgt_device)
        self.a_std = normalization['a_std'].to(tgt_device) if normalization is not None else torch.ones(6).to(tgt_device)
        self.a_mean = normalization['a_mean'].to(tgt_device) if normalization is not None else torch.zeros(6).to(tgt_device)
        self.robot_out_std = self.a_std
        self.robot_out_mean = self.a_mean
        self.res_std = self.a_std #for only_ann
        self.res_mean = self.a_mean
        self.ann_influence = 1.0
        self.robot_influence = 1.0
        self.only_ann = False if not from_exp_dir else exp_model_counter==0 #PAPER
        self.disable_nn = False
        self.auto_decorrelation = False
        self.decorrelation_matrix = torch.empty(6,6).to(tgt_device)
        self.decorrelation_matrix_set = False
        self.inside_loss = False
        self.pvu_mean_set = False
        self.Minv_mean_set = False
        self.Minv_G_std = self.a_std
        self.Minv_G_mean = self.a_mean
        self.Minv_tau_hyd_std = self.a_std
        self.Minv_tau_hyd_mean = self.a_mean
        self.Minv_C_v_std = self.a_std
        self.Minv_C_v_mean = self.a_mean
        self.Minv_f_vis_std = self.a_std
        self.Minv_f_vis_mean = self.a_mean
        self.Minv_f_coul_v_std = self.a_std
        self.Minv_f_coul_v_mean = self.a_mean
        self.Minv_f_coul_2_v_std = self.a_std
        self.Minv_f_coul_2_v_mean = self.a_mean
        self.Minv_tau_m_u_std = self.a_std
        self.Minv_tau_m_u_mean = self.a_mean
        self.friction_mean = self.u_mean
        self.friction_std = self.u_std

    @staticmethod
    def get_decorrelation_matrix(X):
        C = torch.cov(X.T)
        L = torch.linalg.cholesky(C,upper=False)
        A = torch.linalg.inv(L)
        return A

    def forward(self, x, u):   
        if self.only_ann:
            robot_out = torch.zeros_like(x)
            robot_out[:,0:6] = x[:,6:12]
        else:
            if self.nn_configuration in [self.CONF_SERIAL_ACC_ALLIN, self.CONF_SERIAL_ACC_ALLIN_NOFRIC, self.CONF_FRICNN_V, self.CONF_FRICNN_V_P, self.CONF_FRICNN_ANN, self.CONF_FRICNN_ANN2]:
                self.robot.enable_parts_dict = True
            robot_out = self.robot(x,u)
        if self.robustify_nn and not self.disable_nn:
            if not self.pvu_mean_set and self.inside_loss:
                with torch.no_grad():
                    self.u_std = torch.std(u.reshape((-1,6)),dim=0)
                    """
                    #plot x with plotly on 6 subplots:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    for i in range(12):
                        fig.add_trace(go.Scatter(y=x[:,i].cpu().detach().numpy()))
                    fig.show()
                    """
                    self.u_mean = torch.mean(u.reshape((-1,6)),dim=0)
                    self.pos_std = torch.std(x.reshape((-1,12))[:,0:6],dim=0)
                    self.pos_mean = torch.mean(x.reshape((-1,12))[:,0:6],dim=0)
                    self.v_std = torch.std(x.reshape((-1,12))[:,6:12],dim=0)
                    self.v_mean = torch.mean(x.reshape((-1,12))[:,6:12],dim=0)
                    if self.robot.enable_parts_dict:
                        self.robot_out_mean = torch.mean(robot_out['x_dot_result'][:,6:12].reshape((-1,6)),dim=0)
                        self.robot_out_std = torch.std(robot_out['x_dot_result'][:,6:12].reshape((-1,6)),dim=0)
                    else:
                        self.robot_out_mean = torch.mean(robot_out[:,6:12].reshape((-1,6)),dim=0)
                        self.robot_out_std = torch.std(robot_out[:,6:12].reshape((-1,6)),dim=0)
                    self.pvu_mean_set = True
            #we get cmults through 1/std, and offsets through mean
            if self.nn_configuration == self.CONF_SERIAL_ACC:
                net_in = torch.concat( [ ((x[:,6:12]-self.v_mean)/self.v_std), ((x[:,0:6]-self.pos_mean)/self.pos_std), (u-self.u_mean)/self.u_std, (robot_out[:,6:12]-self.robot_out_mean)/self.robot_out_std], dim=1)
                net_out = self.nn(net_in)
            elif self.nn_configuration == self.CONF_PARALLEL_1SAMP:
                net_in = torch.concat( [ ((x[:,6:12]-self.v_mean)/self.v_std), ((x[:,0:6]-self.pos_mean)/self.pos_std), (u-self.u_mean)/self.u_std] , dim=1)
                #if x.shape[0] != 501: breakpoint() #break on loss
                #cov1 = net_in.T@net_in  # net_in.var(dim=0)
                if self.auto_decorrelation:
                    if not self.decorrelation_matrix_set and self.inside_loss:
                        #if x.shape[0] != 501: breakpoint() #break on loss
                        with torch.no_grad():
                            self.decorrelation_matrix = F_robust_nn.get_decorrelation_matrix(net_in).detach()
                            self.decorrelation_matrix_set = True
                    if self.decorrelation_matrix_set:
                        net_in = net_in @ self.decorrelation_matrix.T
                        #cov2 = net_in.T@net_in
                net_out = self.nn(net_in)
            elif self.nn_configuration == self.CONF_SERIAL_ACC_ALLIN or self.nn_configuration == self.CONF_SERIAL_ACC_ALLIN_NOFRIC:
                def M_inverse_at(x):
                    if type(x)==int: return torch.tensor(x)
                    if self.disable_Minv: return x
                    return torch.reshape((torch.stack((x,),dim=1)@robot_out['M_inverse']),(-1,6))
                def matmul2(C,qd_rad):
                    return torch.matmul(C, qd_rad[:,:,None])[:,:,0]
                Minv_G = M_inverse_at(robot_out['G'])
                Minv_tau_hyd = M_inverse_at(robot_out['G_hyd']-robot_out['G'])
                Minv_C_v = M_inverse_at(matmul2(robot_out['C'],robot_out['qd_rad']))
                Minv_f_vis = M_inverse_at(robot_out['f_vis_v'])
                Minv_f_coul_v = M_inverse_at(robot_out['f_coul_v'])
                Minv_f_coul_2_v = M_inverse_at(robot_out['f_coul_2_v'])
                Minv_tau_m_u = M_inverse_at(robot_out['tau_m_u'])
                if not self.Minv_mean_set and self.inside_loss:
                    with torch.no_grad():
                        self.Minv_G_std = torch.std(Minv_G.reshape((-1,6)),dim=0)
                        self.Minv_G_std[5] = 1. #remove any 0 elements from std
                        self.Minv_G_mean = torch.mean(Minv_G.reshape((-1,6)),dim=0)
                        self.Minv_tau_hyd_std = torch.std(Minv_tau_hyd.reshape((-1,6)),dim=0)
                        if self.disable_Minv: self.Minv_tau_hyd_std[[0,2,3,4,5]]=1.
                        else: self.Minv_tau_hyd_std[5] = 1.0 #remove any 0 elements from std
                        self.Minv_tau_hyd_mean = torch.mean(Minv_tau_hyd.reshape((-1,6)),dim=0)
                        self.Minv_C_v_std = torch.std(Minv_C_v.reshape((-1,6)),dim=0)
                        self.Minv_C_v_mean = torch.mean(Minv_C_v.reshape((-1,6)),dim=0)
                        if self.nn_configuration != self.CONF_SERIAL_ACC_ALLIN_NOFRIC:
                            self.Minv_f_vis_std = torch.std(Minv_f_vis.reshape((-1,6)),dim=0)
                            self.Minv_f_vis_mean = torch.mean(Minv_f_vis.reshape((-1,6)),dim=0)
                            self.Minv_f_coul_v_std = torch.std(Minv_f_coul_v.reshape((-1,6)),dim=0)
                            self.Minv_f_coul_v_mean = torch.mean(Minv_f_coul_v.reshape((-1,6)),dim=0)
                            self.Minv_f_coul_2_v_std = torch.std(Minv_f_coul_2_v.reshape((-1,6)),dim=0)
                            self.Minv_f_coul_2_v_mean = torch.mean(Minv_f_coul_2_v.reshape((-1,6)),dim=0)
                        self.Minv_tau_m_u_std = torch.std(Minv_tau_m_u.reshape((-1,6)),dim=0)
                        self.Minv_tau_m_u_mean = torch.mean(Minv_tau_m_u.reshape((-1,6)),dim=0)
                        self.Minv_mean_set = True
                tau_hyd_concat_full = (Minv_tau_hyd-self.Minv_tau_hyd_mean)/self.Minv_tau_hyd_std
                Minv_G_concat_full = ((Minv_G-self.Minv_G_mean)/self.Minv_G_std)
                if self.disable_Minv: 
                    tau_hyd_concat_reduced = tau_hyd_concat_full[:,[1,]]
                    Minv_G_concat_reduced = Minv_G_concat_full[:,[1,2,3,4]]
                else: 
                    tau_hyd_concat_reduced = tau_hyd_concat_full[:,0:5]
                    Minv_G_concat_reduced = Minv_G_concat_full[:,[0,1,2,3,4]]
                to_concat = [ ((x[:,6:12]-self.v_mean)/self.v_std), #0  +6
                    ((x[:,0:6]-self.pos_mean)/self.pos_std), #6  +6
                    (Minv_tau_m_u-self.Minv_tau_m_u_mean)/self.Minv_tau_m_u_std, #12  +6
                    ((Minv_C_v-self.Minv_C_v_mean)/self.Minv_C_v_std)[:,0:5], #18  +5
                    Minv_G_concat_reduced, #23  +5|4d
                    tau_hyd_concat_reduced, ] #28|27d  +5|1d     --> (end) 33 | 28d
                if self.nn_configuration == self.CONF_SERIAL_ACC_ALLIN:
                    #we don't add the friction terms in case of CONF_SERIAL_ACC_ALLIN_NOFRIC
                    to_concat.append((Minv_f_vis-self.Minv_f_vis_mean)/self.Minv_f_vis_std) 
                    to_concat.append((Minv_f_coul_v-self.Minv_f_coul_v_mean)/self.Minv_f_coul_v_std) 
                    to_concat.append((Minv_f_coul_2_v-self.Minv_f_coul_2_v_mean)/self.Minv_f_coul_2_v_std) 
                net_in = torch.concat(to_concat, dim=1)
                assert net_in.isnan().sum() == 0, "net_in has nan"
                net_out = self.nn(net_in)
                if very_debug:
                    #plot all the items in net_in (data is in 0th dimension, 1st dimension is the channels).
                    #plot each of the channels on subplots. 
                    fig = make_subplots(rows=net_in.shape[1], cols=1, shared_xaxes=True)
                    for i in range(net_in.shape[1]):
                        fig.add_trace(go.Scatter(y=net_in[:,i].detach(), name=f'net_in {i}'), row=i+1, col=1)
                    fig.update_layout(height=10000, width=1000, title_text="net_in")
                    fig.write_html('net_in.html')
                    #fig.show()
                    print
            elif self.nn_configuration in [self.CONF_FRICNN_V, self.CONF_FRICNN_V_P, self.CONF_FRICNN_ANN, self.CONF_FRICNN_ANN2]:
                if not self.Minv_mean_set and self.inside_loss:
                    with torch.no_grad():
                        ret_diff_fd = diff_fd(x[:,0:6], None, dim=0, dt=data_dt)
                        a_from_data = ret_diff_fd[2]
                        tau_f = -torch.matmul(robot_out['M'], a_from_data[:,:,None])[:,:,0]-robot_out['G_hyd']-torch.matmul(robot_out['C'], x[:,6:12][:,:,None])[:,:,0]+robot_out['tau_m_u'] #Double check parallelization with for loop 
                        #parallelization check:
                        #tau_f_b_collect = []
                        #for i in range(45000):
                        #    tau_f_b_collect.append(-torch.matmul(robot_out['M'][(i,),:,:], a_from_data[(i,),:,None])[:,:,0]-robot_out['G_hyd'][(i,),:]-torch.matmul(robot_out['C'][(i,),:,:], x[(i,),6:12][:,:,None])[:,:,0]+robot_out['tau_m_u'][(i,),:])
                        #torch.set_printoptions(precision=10)
                        #print((torch.cat(tau_f_b_collect, dim=0)-tau_f).abs().sum())
                        self.tau_f_std = tau_f.std(dim=0)
                        self.tau_f_mean = tau_f.mean(dim=0)
                        self.Minv_mean_set = True
                if self.nn_configuration == self.CONF_FRICNN_V:
                    #the net input should be only the velocities:
                    net_in = (x[:,6:12]-self.v_mean)/self.v_std
                    net_outs = []
                    for i in range(6):
                        net_outs.append(self.friction_nns[i](net_in[:,(i,)]))
                    net_out = torch.stack(net_outs, dim=1)[:,:,0]
                elif self.nn_configuration == self.CONF_FRICNN_V_P:
                    #the net input should be only the velocities:
                    net_in_v = (x[:,6:12]-self.v_mean)/self.v_std
                    net_in_p = (x[:,0:6]-self.pos_mean)/self.pos_std
                    net_outs = []
                    for i in range(6):
                        net_in = torch.cat((net_in_v[:,(i,)], net_in_p), dim=1)
                        net_outs.append(self.friction_nns[i](net_in))
                    net_out = torch.stack(net_outs, dim=1)[:,:,0]
                elif self.nn_configuration == self.CONF_FRICNN_ANN or self.nn_configuration == self.CONF_FRICNN_ANN2:
                    #the net input should be only the velocities:
                    net_in = (x[:,6:12]-self.v_mean)/self.v_std
                    net_outs = []
                    for i in range(6):
                        net_outs.append(self.friction_nns[i](net_in[:,(i,)]))
                    net_out = torch.stack(net_outs, dim=1)[:,:,0]
                    net_in_2 = torch.concat( [ ((x[:,6:12]-self.v_mean)/self.v_std), ((x[:,0:6]-self.pos_mean)/self.pos_std), (u-self.u_mean)/self.u_std] , dim=1)
                    net_out_2 = self.nn(net_in_2)
            else: #this is needed because of torchscript
                net_in = torch.zeros_like(robot_out[:,6:12])
                net_out = net_in
            if very_debug:
                print('net_in std:', net_in.std(dim=0))
                print('net_in mean:', net_in.mean(dim=0))
                print('net_out std:', net_out.std(dim=0))
                print('net_out mean:', net_out.mean(dim=0))
                if net_in.std(dim=0).max()>1.5 or net_in.std(dim=0).min()<0.5:
                    print('VERY_WARNING: net_in.std out of limits')
            if self.nn_configuration == self.CONF_PARALLEL_1SAMP:
                module_out = torch.cat((robot_out[:,0:6], self.robot_influence*robot_out[:,6:12] + self.ann_influence*(net_out*self.res_std+self.res_mean)), dim=1)
                torch.save(self.ann_influence*(net_out*self.res_std+self.res_mean), 'net_out.torch_save')
            elif self.nn_configuration == self.CONF_SERIAL_ACC:
                module_out = torch.cat((robot_out[:,0:6], net_out*self.a_std+self.a_mean), dim=1)
                torch.save(self.ann_influence*(net_out*self.a_std+self.a_mean), 'net_out.torch_save')
            elif self.nn_configuration == self.CONF_SERIAL_ACC_ALLIN or self.nn_configuration == self.CONF_SERIAL_ACC_ALLIN_NOFRIC:
                module_out = torch.cat((robot_out['x_dot_result'][:,0:6], net_out*self.a_std+self.a_mean), dim=1)
                torch.save(self.ann_influence*(net_out*self.a_std+self.a_mean), 'net_out.torch_save')
            elif self.nn_configuration == self.CONF_FRICNN_V or self.nn_configuration == self.CONF_FRICNN_V_P:
                def M_inverse_at(x):
                    return torch.reshape((torch.stack((x,),dim=1)@robot_out['M_inverse']),(-1,6))
                #net_out = torch.randn_like(net_out)
                robot_additional_part = (M_inverse_at(net_out*self.tau_f_std+self.tau_f_mean))
                #robot_additional_part_collect = []
                #for i in range(45000):
                #    robot_additional_part_collect.append(    torch.reshape((torch.stack((    (net_out[(i,),:]*self.tau_f_std+self.tau_f_mean)    ,),dim=1)@robot_out['M_inverse'][(i,),:,:]),(-1,6))   )
                #robot_additional_part_collect = torch.stack(robot_additional_part_collect, dim=0)[:,0,:]
                #torch.set_printoptions(precision=10)
                #print((robot_additional_part_collect - robot_additional_part).abs().sum())

                module_out = torch.cat( (robot_out['x_dot_result'][:,0:6], robot_out['x_dot_result'][:,6:12] + robot_additional_part), dim=1 )
                torch.save(robot_additional_part, 'net_out.torch_save')
            elif self.nn_configuration == self.CONF_FRICNN_ANN:
                def M_inverse_at(x):
                    return torch.reshape((torch.stack((x,),dim=1)@robot_out['M_inverse']),(-1,6))
                alpha_v_part = 0.5
                alpha_remains = 0.5
                robot_additional_part = (M_inverse_at((alpha_v_part*net_out+alpha_remains*net_out_2)*self.tau_f_std+self.tau_f_mean))
                module_out = torch.cat( (robot_out['x_dot_result'][:,0:6], robot_out['x_dot_result'][:,6:12] + robot_additional_part), dim=1 )
                torch.save(robot_additional_part, 'net_out.torch_save')
            elif self.nn_configuration == self.CONF_FRICNN_ANN2:
                def M_inverse_at(x):
                    return torch.reshape((torch.stack((x,),dim=1)@robot_out['M_inverse']),(-1,6))
                robot_additional_part = (M_inverse_at(net_out*self.tau_f_std+self.tau_f_mean)) + (self.res_std*net_out_2+self.res_mean)
                module_out = torch.cat( (robot_out['x_dot_result'][:,0:6], robot_out['x_dot_result'][:,6:12] + robot_additional_part), dim=1 )
                torch.save(robot_additional_part, 'net_out.torch_save')
            #we_have_nans = torch.any(net_out[:,i].std().isnan()) or torch.any(net_out[:,i].std().isnan())
            #print('NaNs:',we_have_nans)
            #print('net_in covariance:', net_in.T@net_in)
            #print('net_in mean|std:',float(net_in.mean()),float(net_in.std()), ' and channel wise:',[(float(net_in[:,i].mean()),float(net_in[:,i].std())) for i in range(net_in.shape[1])])
            #print('net_in covariance:', net_in.T@net_in)
            #print('net_out mean|std:' ,float(net_out.mean()),float(net_out.std()),' and channel wise:',[(float(net_out[:,i].mean()),float(net_out[:,i].std())) for i in range(net_out.shape[1])])
        else:
            module_out = robot_out if not self.robot.enable_parts_dict else robot_out['x_dot_result']
        return module_out


class F_robot_sympybotics(nn.Module):

    @staticmethod
    def hydraulic_weight_counterbalance_hwc_params(angle_ax_2, hwc_params):
        Z_176bar = 1.01                    # compressibility factor nitrogen at pressure of 176 bar and T = 295.5 K (at 328 K: Z = 1.02986)
        realGasFactor_nitrogen = 296.8
        temperature = 328                  # max operating temperature of the robot: T_max = 328 K
        mass_nitrogen = 2*0.0795           # in kg
        lumped_ZRealTemperatureMass = (Z_176bar * realGasFactor_nitrogen * temperature * mass_nitrogen) #default value: 15633.50073600 

        #Area_Piston can be lumped into a single, positive number
        Diameter_Rod = 30e-3                                    # in m
        Diameter_Piston = 55e-3                                 # in m, approximated using Website of Bucher, DIN EN 22553 and the width of welding seam on piston as well as taking the max ammount of force needed)
        lumped_Area_Piston = (1e-3*hwc_params[0])+math.pi*(Diameter_Piston**2 - Diameter_Rod**2)/4 # in m^2; it should be 0.0016689710972195777 by default

        v_max = (1e-4*hwc_params[1])+2* 0.00048                 # in m^3
        setDeviation = ((10*hwc_params[2])+95)*math.pi/180      # set deviation from vertical pose

        D = (0.1*hwc_params[3])+8e-1 # direct distance between anchor point of spring on link 1 to axis 2
        k = (0.1*hwc_params[4])+2e-1 # distance anchor point of spring on link 2 to axis 2
        l = D-k # length of spring in fully retracted position

        delta = angle_ax_2 + setDeviation  # set deviation from vertical pose -5deg (due to mount of spring, the vertical pose is not at minimum force. That is at -95deg)
        dl = torch.sqrt(D**2 + k**2 - 2*D*k*torch.cos(delta))-l
        h = (torch.sin(delta)*D*k)/(dl+l) # utilizing sine-rule
        press = lumped_ZRealTemperatureMass / ( v_max - dl * lumped_Area_Piston)
        tau = h*press*lumped_Area_Piston
        return tau

    @staticmethod
    def x_dot(x, tau_m__or__qdd_rad, inverse_mode, friction_params, hwc_params, dyn_params, u_params, modifier, enable_parts_dict):
        assert x.shape[1]==12,"x must have 12 states"
        assert x.ndim==2,"x must be 2D [batch_size,12]"
        assert tau_m__or__qdd_rad.shape[1]==6,"tau_m__or__qdd_rad must have 6 items per row"
        assert tau_m__or__qdd_rad.ndim==2,"tau_m__or__qdd_rad must be 2D [batch_size,6]"
        assert x.shape[0]==tau_m__or__qdd_rad.shape[0],"x and tau_m__or__qdd_rad must have the same batch size"
        assert friction_params.ndim==1,"friction_params must be 1D"
        assert hwc_params.ndim==1,"hwc_params must be 1D"
        assert dyn_params.ndim==1,"dyn_params must be 1D"
        assert u_params.ndim==1,"u_params must be 1D"

        if inverse_mode:
            qdd_rad = tau_m__or__qdd_rad
        else:
            tau_m = tau_m__or__qdd_rad
        tgt_device = 'cpu' #this is needed to be specified here because of torchscript 

        friction_scale = torch.tensor([
            5e2, 5e2, 5e2, 1e2, 1e2, 1e2,
            2e1, 2e1, 2e1, 2e1, 2e1, 2e1,
            5e1, 1e2, 5e1, 1e1, 1e1, 1e1,
            5e2, 5e3, 5e3, 5e2, 5e2, 5e2,
            5e0, 1e1, 1e1, 1e1, 1e1, 1e1,
            1e0,1e0,1e0,1e0
            ], device=tgt_device) #transposed compared to the MATLAB version

        friction_base_scaled = torch.tensor([
            10, #f_vis
            10,
            10,
            10,
            10,
            10,
            2.265625, #f_coul
            5.97812499999999946709294817992486059665679931640625,
            5.359375,
            9.690625000000000710542735760100185871124267578125,
            0.4093750000000000444089209850062616169452667236328125,
            7.8343749999999996447286321199499070644378662109375,
            0.4093750000000000444089209850062616169452667236328125, #f_a
            2.265625,
            9.0718750000000003552713678800500929355621337890625,
            6.59687499999999982236431605997495353221893310546875,
            4.7406249999999996447286321199499070644378662109375,
            1.646875000000000088817841970012523233890533447265625,
            1.646875000000000088817841970012523233890533447265625, #f_b
            5.97812499999999946709294817992486059665679931640625,
            7.8343749999999996447286321199499070644378662109375,
            9.690625000000000710542735760100185871124267578125,
            4.7406249999999996447286321199499070644378662109375,
            4.12187500000000017763568394002504646778106689453125,
            2.8125, #f_asym
            -2.1875,
            -0.3125,
            -4.0625,
            -0.3125,
            -4.6875
            ], device=tgt_device)

        general_extra_scaling = 0.01

        f_vis  = (friction_base_scaled[0:6]   * (1+0.01*friction_params[0:6]  )) * friction_scale[0:6]
        f_coul = (friction_base_scaled[6:12]  * (1+0.01*friction_params[6:12] )) * friction_scale[6:12]
        f_a    = (friction_base_scaled[12:18] * (1+0.01*friction_params[12:18])) * friction_scale[12:18]
        f_b    = (friction_base_scaled[18:24] * (1+0.01*friction_params[18:24])) * friction_scale[18:24]
        f_asym = (friction_base_scaled[24:30] * (1+0.01*friction_params[24:30])) * friction_scale[24:30]

        N_batches = x.shape[0]
        q_rad = x[:,0:6]
        qd_rad = x[:,6:12]
        s_f_ode = 1 #this comes from additional_params, but there it is 1
        f_sign = torch.tanh(s_f_ode * qd_rad)
        if ((False or modifier) if not from_exp_dir else (exp_model_counter in [3,4,6,7,8,9])): #have friction term on True:
            f_vis_v = f_vis * qd_rad
            f_coul_v = f_coul*f_sign  
            f_coul_2_v = f_a*torch.tanh(f_b * qd_rad)
            tau_f = f_asym + f_vis_v + f_coul_v  #+ f_coul_2_v 
            sys.stdout.write('F'); sys.stdout.flush()
        else:
            f_vis_v = 0
            f_coul_v = 0
            f_coul_2_v = 0
            tau_f = 0
            sys.stdout.write('N'); sys.stdout.flush()
        
        #tau_f = f_asym + f_vis * qd_rad + f_coul*f_sign + f_a*torch.tanh(f_b * qd_rad)  #the full one

        #TODO somehow get the parameter values calculated in advance

        u = torch.tensor([ 256.85714285714283278139191679656505584716796875,
            267.4285714285714448124053888022899627685546875,
            252.333333333333342807236476801335811614990234375,
            221.0,
            239.619047619047620401033782400190830230712890625,
            154.32313432835820776745094917714595794677734375,
            ], device=tgt_device)+(0*0.0001*u_params)
        tau_dist = 0
        # <[dyn_M_inverse, dyn_C, dyn_G] = robot_dynamics(q_rad, qd_rad)>
        # transfer from relative to global coordiantes
        qs_rad = torch.tensor([0., -math.pi/2, math.pi/2, 0., 0., 0.], device=tgt_device) #This is something that Jonas wrote
        q_global_rad = q_rad + qs_rad

        deg_to_rad = math.pi/180

        L_1xx = 16733765e-6
        L_1yy = 29799119e-6
        L_1zz = 32814954e-6
        L_1xy = -1703664e-6
        L_1yz = 129384e-6
        L_1xz = 4404062e-6
        r_1 = [-353e-3, 174e-3, -4e-3]
        m_1 = 395.40
        l_1x = m_1*r_1[0]
        l_1y = m_1*r_1[1]
        l_1z = m_1*r_1[2]
        Ia_1 = 0.00923 * (u[0]**2)
        fv_1orig =  0.001476346 # We don't use these here
        fc_1orig = 0.73192436 # We don't use these here
        qlim_1low = -147.*deg_to_rad
        qlim_1high = 147.*deg_to_rad #TODO Use these limits for checking

        L_2xx = 7507819e-6
        L_2yy = 88886691e-6
        L_2zz = 88673761e-6
        L_2xy = 430046e-6
        L_2yz = 173157e-6
        L_2xz = -4396607e-6
        r_2 = [-709e-3, -1e-3, 23e-3]
        m_2 = 339.12
        l_2x = m_2*r_2[0]
        l_2y = m_2*r_2[1]
        l_2z = m_2*r_2[2]
        Ia_2 = 0.0118 * (u[1]**2)
        fv_2orig = 0.010681832
        fc_2orig = 1.832264957
        qlim_2low = -140*deg_to_rad
        qlim_2high = -5*deg_to_rad

        L_3xx = 10783798e-6
        L_3yy = 10186135e-6
        L_3zz = 1943179e-6
        L_3xy = 118474e-6
        L_3yz = -606384e-6
        L_3xz = -101950e-6
        r_3 = [-18e-3, -18e-3, -76e-3]
        m_3 = 105.21
        l_3x = m_3*r_3[0]
        l_3y = m_3*r_3[1]
        l_3z = m_3*r_3[2]
        Ia_3 = 0.0118 * (u[2]**2)
        fv_3orig = 0.08398663
        fc_3orig = 1.862615588
        qlim_3low = -112*deg_to_rad
        qlim_3high = 153*deg_to_rad

        L_4xx = 425035e-6
        L_4yy = 217993e-6
        L_4zz = 428465e-6
        L_4xy = -4e-6
        L_4yz = -11353e-6
        L_4xz = -27e-6
        r_4 = [0., 236e-3, 1e-3]
        m_4 = 35.80
        l_4x = m_4*r_4[0]
        l_4y = m_4*r_4[1]
        l_4z = m_4*r_4[2]
        Ia_4 = 0.00173 * (u[3]**2)
        fv_4orig = 0.001916077
        fc_4orig = 0.447963801
        qlim_4low = -350.*deg_to_rad
        qlim_4high = 350.*deg_to_rad

        L_5xx = 283826e-6
        L_5yy = 229562e-6
        L_5zz = 258266e-6
        L_5xy = -4e-6
        L_5yz = -19521e-6
        L_5xz = -120e-6
        r_5 = [0., -80e-3, -9e-3]
        m_5 = 31.46
        l_5x = m_5*r_5[0]
        l_5y = m_5*r_5[1]
        l_5z = m_5*r_5[2]
        Ia_5 = 0.00173 * (u[4]**2)
        fv_5orig = 0.001629877
        fc_5orig = 0.413155803
        qlim_5low = -122.5*deg_to_rad
        qlim_5high = 122.5*deg_to_rad

        # payload is defined as additional weight on joint 6
        m_axis6 = 17.78 # kg
        r_axis6 = -50 # mm
        r_payload = 100 # mm
        m_payload = 150 # kg 
        m_payload_axis = m_payload + m_axis6
        r_payload_axis = (m_payload*r_payload + m_axis6*r_axis6)/m_payload_axis

        L_6xx = 0.
        L_6xy = 0.
        L_6xz = 0.
        L_6yy = 0.
        L_6yz = 0.
        L_6zz = 0.
        r_6 = [0., 0., r_payload_axis*1e-3]
        m_6 = m_payload_axis
        l_6x = m_6*r_6[0]
        l_6y = m_6*r_6[1]
        l_6z = m_6*r_6[2]
        Ia_6 = 0.00173 * (u[5]**2)
        fv_6orig = 0.002205322
        fc_6orig = 0.372594817
        qlim_6low = -350.*deg_to_rad
        qlim_6high = 350.*deg_to_rad

        dyn_params_default=torch.tensor([
            L_1xx, L_1xy, L_1xz, L_1yy, L_1yz, L_1zz, l_1x, l_1y, l_1z, m_1, 0.,
            L_2xx, L_2xy, L_2xz, L_2yy, L_2yz, L_2zz, l_2x, l_2y, l_2z, m_2, 0.,
            L_3xx, L_3xy, L_3xz, L_3yy, L_3yz, L_3zz, l_3x, l_3y, l_3z, m_3, 0.,
            L_4xx, L_4xy, L_4xz, L_4yy, L_4yz, L_4zz, l_4x, l_4y, l_4z, m_4, 0.,
            L_5xx, L_5xy, L_5xz, L_5yy, L_5yz, L_5zz, l_5x, l_5y, l_5z, m_5, 0.,
            L_6xx, L_6xy, L_6xz, L_6yy, L_6yz, L_6zz, l_6x, l_6y, l_6z, m_6, 0.
            ], device=tgt_device)

        dyn_params_default[11-1] = Ia_1 #this is hack for torchscript
        dyn_params_default[22-1] = Ia_2
        dyn_params_default[33-1] = Ia_3
        dyn_params_default[44-1] = Ia_4
        dyn_params_default[55-1] = Ia_5
        dyn_params_default[66-1] = Ia_6

        #Example values: big_review_january
        #L_3xx = 10783798e-6
        #L_3xy = 118474e-6
        #L_3xz = -101950e-6
        #L_3yy = 10186135e-6
        #L_3yz = -606384e-6
        #L_3zz = 1943179e-6
        #l_3x = -1.89
        #l_3y = -1.89
        #l_3z = -8
        #m_3 = 105.21
        #Ia_3 = 2.9736

        dyn_params_somezero = torch.zeros_like(dyn_params)
        dyn_params_somezero[0:55] = dyn_params[0:55]
        dyn_params_somezero[55:63] = 0.
        dyn_params_somezero[63:66] = dyn_params[63:66]
        dyn_params_apply = (dyn_params_default*(1+0.001*dyn_params_somezero))
        N_batch_items = q_global_rad.shape[0]
        M = sympybotconv_M_code.M(dyn_params_apply, q_global_rad, tgt_device=tgt_device).reshape((N_batch_items, 6, 6)) 
        M_inverse = torch.linalg.inv(M)
        #M2 = inertia_150( q_global2_rad ) #LOOKOUT all the indexing should be -1!

        C = sympybotconv_C_code.C(dyn_params_apply, q_global_rad, qd_rad, tgt_device=tgt_device ).reshape((N_batch_items, 6, 6))
        #C2 = coriolis_150( q_global2_rad, qd_rad )

        G = sympybotconv_g_code.g(dyn_params_apply, q_global_rad, tgt_device=tgt_device)
        #G2 = gravload_150( q_global2_rad )

        #tau_hyd  = F_robot_manual_python_rewrite.hydraulic_weight_counterbalance(q_global_rad[:,1])
        tau_hyd = F_robot_sympybotics.hydraulic_weight_counterbalance_hwc_params(q_global_rad[:,1], 0*general_extra_scaling*hwc_params) #big_review_january we don't change the HWC parameters

        #G[:,1] += tau_hyd
        G_hyd = torch.cat([G[:,(0,)], G[:,(1,)]+tau_hyd[:,None], G[:,2:6]], dim=1)
        # </robot_dynamics>
        if enable_parts_dict:
            parts_dict = dict()
            parts_dict['M'] = M
            parts_dict['M_inverse'] = M_inverse
            parts_dict['C'] = C
            parts_dict['G'] = G
            parts_dict['tau_hyd'] = tau_hyd
            parts_dict['G_hyd'] = G_hyd
            parts_dict['tau_m_u'] = tau_m*u
            parts_dict['f_asym'] = f_asym
            parts_dict['f_vis_v'] = f_vis_v
            parts_dict['f_coul_v'] = f_coul_v
            parts_dict['f_coul_2_v'] = f_coul_2_v
            parts_dict['tau_f'] = tau_f
            parts_dict['q_global_rad'] = q_global_rad
            parts_dict['qd_rad'] = qd_rad
        if inverse_mode:
            #q, dq, ddq --> tau
            #assert N_batch_items == 1
            x_dot_result = torch.reshape(  (torch.matmul(M, qdd_rad[:,:,None])[:,:,0] + torch.matmul(C, qd_rad[:,:,None])[:,:,0] + G_hyd  + tau_f)/u , (-1,6)) #LOOKOUT check that if it does work also if the batch size is >1
        else:
            #q, dq, tau --> ddq
            x_dot_result = torch.empty((N_batches, 12), device=tgt_device)
            x_dot_result[:,0:6] = qd_rad
            x_dot_result[:,6:12] = torch.reshape((torch.stack(((-tau_f -G_hyd -torch.matmul(C, qd_rad[:,:,None])[:,:,0] + tau_m*u + tau_dist ),),dim=1)@M_inverse),(-1,6)) #LOOKOUT check that it does the same as einsum
        if enable_parts_dict:
            parts_dict['x_dot_result'] = x_dot_result
            return parts_dict
        else:
            return x_dot_result



        #plot M*tonorm_a, C, G, M_inverse, tau_hyd, G_hyd, x_dot_result for the first 6 channels, with plotly, on 6 subplots, in 6 rows:
        tonorm_a = torch.load('tonorm_a.torchsave')
        def M_inverse_at(x):
            return torch.reshape((torch.stack((x,),dim=1)@M_inverse),(-1,6))
        def matmul2(C,qd_rad):
            return torch.matmul(C, qd_rad[:,:,None])[:,:,0]
        Minv_G = M_inverse_at(G)
        Minv_tau_hyd = M_inverse_at(G_hyd-G)
        Minv_C_v = M_inverse_at(matmul2(C,qd_rad))
        Minv_f_asym = M_inverse_at(f_asym[None,:].repeat((15000,1)))
        Minv_f_vis = M_inverse_at(f_vis_v)
        Minv_f_coul_v = M_inverse_at(f_coul_v)
        Minv_f_coul_2_v = M_inverse_at(f_coul_2_v)
        Minv_tau_m_u = M_inverse_at(tau_m*u)
        import plotly
        cols = plotly.colors.DEFAULT_PLOTLY_COLORS
        subplots=6
        fig = make_subplots(rows=subplots, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True},]]*subplots)
        for i in range(subplots):
            fig.add_trace(go.Scatter(marker=dict(color=cols[0]), y=tonorm_a[:,i], name='tonorm_a #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[1]), y= Minv_G[:,i], name='Minv_G #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[2]), y= Minv_tau_hyd[:,i], name='Minv_tau_hyd #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[3]), y= Minv_C_v[:,i], name='Minv_C_v #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[4]), y= Minv_f_vis[:,i], name='Minv_f_vis #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[5]), y= Minv_f_coul_v[:,i], name='Minv_f_coul_v #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[6]), y= Minv_f_coul_2_v[:,i], name='Minv_f_coul_2_v #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[7]), y= Minv_tau_m_u[:,i], name='Minv_tau_m_u #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[8]), y= -Minv_f_vis[:,i]-Minv_f_coul_v[:,i]-Minv_f_asym[:,i]-Minv_G[:,i]-Minv_tau_hyd[:,i] -Minv_C_v[:,i]+Minv_tau_m_u[:,i] , name='Minv_combined #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[1]), y= x_dot_result[:,6+i], name='x_dot_result[:,6+i] #'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[0]), y= -Minv_f_vis[:,i]-Minv_f_coul_v[:,i]-Minv_f_coul_2_v[:,i]-Minv_f_asym[:,i]-Minv_G[:,i]-Minv_tau_hyd[:,i] -Minv_C_v[:,i]+Minv_tau_m_u[:,i] , name='Minv_combined_2 #'+str(i)), row=i+1, col=1, secondary_y=False)
        fig.show(renderer='firefox')

        #plot M*tonorm_a, C, G, M_inverse, tau_hyd, G_hyd, x_dot_result for the first 6 channels, with plotly, on 6 subplots, in 6 rows:
        tonorm_a = torch.load('tonorm_a.torchsave')
        M_a = torch.matmul(M, tonorm_a[:,:,None])[:,:,0]
        C_v = torch.matmul(C, qd_rad[:,:,None])[:,:,0]
        import plotly
        cols = plotly.colors.DEFAULT_PLOTLY_COLORS
        subplots=6
        fig = make_subplots(rows=subplots, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True},]]*subplots)
        for i in range(subplots):
            #plot tonorm_a:
            fig.add_trace(go.Scatter(marker=dict(color=cols[1]), y=tonorm_a[:,i], name='tonorm_a'+str(i)), row=i+1, col=1, secondary_y=True)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[7]), y= (qd_rad.diff(dim=0)/data_dt)[:,i], name='d qd_rad'+str(i)), row=i+1, col=1, secondary_y=True)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[2]), y=qd_rad[:,i], name='qd_rad'+str(i)), row=i+1, col=1, secondary_y=True)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[0]), y=M_a[:,i], name='M_a '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[3]), y=C_v[:,i], name='C_v '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[4]), y=G[:,i], name='G '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[9]), y= (G_hyd)[:,i], name=' G+hwc '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[1]), y= (tau_f)[:,i], name='tau_f '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[5]), y= (tau_m*u)[:,i], name=' tau_m*u '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[5]), y= (-tau_m*u)[:,i], name=' -tau_m*u '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[6]), y= (M_a+C_v+G)[:,i], name=' Ma+Cv+G '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[8]), y= (M_a+C_v+G+tau_f)[:,i], name=' Ma+Cv+G+f '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[7]), y= (M_a+C_v)[:,i], name=' Ma+Cv '+str(i)), row=i+1, col=1, secondary_y=False)
            #fig.add_trace(go.Scatter(marker=dict(color=cols[2]), y= (M_a+C_v+tau_f)[:,i], name=' Ma+Cv+f '+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[8]), y= (M_a+C_v+G_hyd+tau_f)[:,i], name=' Ma+Cv+G+f '+str(i)), row=i+1, col=1, secondary_y=False)
        fig.show(renderer='firefox')

        #plot torch.linalg.cond(M) on plotly:
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(y=torch.linalg.cond(M), name='cond(M)'), row=1, col=1)
        fig.show(renderer='firefox')

        #plot all of those above, in the frequency domain ( using numpy.fft), on dB scale, on 6 subplots, using plotly:
        import numpy as np
        import plotly
        cols = plotly.colors.DEFAULT_PLOTLY_COLORS
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True},]]*6)
        for i in range(6):
            fig.add_trace(go.Scatter(marker=dict(color=cols[1]), y=20*np.log10(np.abs(np.fft.fft(tonorm_a[:,i]))), name='tonorm_a'+str(i)), row=i+1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(marker=dict(color=cols[7]), y=20*np.log10(np.abs(np.fft.fft((qd_rad.diff(dim=0)/data_dt)[:,i]))), name='d qd_rad'+str(i)), row=i+1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(marker=dict(color=cols[2]), y=20*np.log10(np.abs(np.fft.fft(qd_rad[:,i]))), name='qd_rad'+str(i)), row=i+1, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(marker=dict(color=cols[0]), y=20*np.log10(np.abs(np.fft.fft(M_a[:,i]))), name='M_a'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[3]), y=20*np.log10(np.abs(np.fft.fft(C_v[:,i]))), name='C_v'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[4]), y=20*np.log10(np.abs(np.fft.fft(G[:,i]))), name='G'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[5]), y=20*np.log10(np.abs(np.fft.fft((tau_m*u)[:,i]))), name=' tau_m*u'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[6]), y=20*np.log10(np.abs(np.fft.fft((M_a+C_v+G)[:,i]))), name=' Ma+Cv+G'+str(i)), row=i+1, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(marker=dict(color=cols[8]), y=20*np.log10(np.abs(np.fft.fft((M_a+C_v+G+tau_f)[:,i]))), name=' Ma+Cv+G+f'+str(i)), row=i+1, col=1, secondary_y=False)
        fig.show()

    def check_if_params_phy_meaningful(self):
        #display warning messages if the parameters are not physically meaningful
        general_extra_scaling = 0.01
        #get elements 1 and 3 out of a torch.tensor:
        if torch.any(torch.abs(self.friction_params) > (1/general_extra_scaling)*0.1):
            print('check_if_params_phy_meaningful :: WARNING: friction_params changed more than 10%'); sys.stdout.flush()
        if torch.any(torch.abs(self.dyn_params) > (1/general_extra_scaling)*0.1):
            print('check_if_params_phy_meaningful :: WARNING: dyn_params changed more than 10%'); sys.stdout.flush()
        if torch.any(torch.abs(self.hwc_params) > (1/general_extra_scaling)*0.1):
            print('check_if_params_phy_meaningful :: WARNING: hwc_params changed more than 10%'); sys.stdout.flush()
        #if not torch.all(self.dyn_params.take(torch.tensor((10,10+11,10+22,10+33,10+44,10+55))) > 0):
        #    print('check_if_params_phy_meaningful :: WARNING: some Ia are negative'); sys.stdout.flush()
        #if not torch.all(self.dyn_params.take(torch.tensor((9,9+11,9+22,9+33,9+44,9+55))) > 0):
        #    print('check_if_params_phy_meaningful :: WARNING: some m are negative'); sys.stdout.flush()
        #if not torch.all(self.dyn_params.take(torch.tensor((0,0+11,0+22,0+33,0+44,0+55))) > 0):
        #    print('check_if_params_phy_meaningful :: WARNING: some L_xx are negative'); sys.stdout.flush()
        #if not torch.all(self.dyn_params.take(torch.tensor((3,3+11,3+22,3+33,3+44,3+55))) > 0):
        #    print('check_if_params_phy_meaningful :: WARNING: some L_yy are negative'); sys.stdout.flush()
        #if not torch.all(self.dyn_params.take(torch.tensor((5,5+11,5+22,5+33,5+44,5+55))) > 0):
        #    print('check_if_params_phy_meaningful :: WARNING: some L_zz are negative'); sys.stdout.flush()

    def __init__(self, nx, nu, params):
        super(F_robot_sympybotics, self).__init__()
        assert nx == 12
        assert nu == 6
        self.modifier = False
        self.enable_parts_dict = False
        for key, value in params.items():
            setattr(self,key,nn.Parameter(value))

    def hook_it(x,msg):
        # type: (Tensor, str) -> Tensor
        if x.requires_grad: #during validation and simulation, when we are inside torch.no_grad, then we don't register the hook
            h_hook = x.register_hook(lambda x: printgrad(x,msg))
        return x

    @torch.jit.export
    def x_dot_inverse(self, x, a):
        return self.x_dot(x, a, True, self.friction_params, self.hwc_params, self.dyn_params, self.u_params, self.modifier, self.enable_parts_dict)

    def forward(self, x, u):
        #sys.stdout.write("T" if self.modifier else "F")
        #sys.stdout.flush()
        return self.x_dot(x, u, False, self.friction_params, self.hwc_params, self.dyn_params, self.u_params, self.modifier, self.enable_parts_dict)


class robot_open_loop_simulator(nn.Module):
    #this one can only do one simulation at a time, not a batch of simulations specified by a tensor
    def __init__(self, fn):
        super(robot_open_loop_simulator, self).__init__()
        self.fn = fn

    def forward(self, x0, tau):
        #x_ref is [N_steps, 18]: I need only 2 dimensions there
        #x0 is [12]
        #whenever you call me, make sure to use torch.no_grad: do an assert on that:
        assert not x0.requires_grad
        assert not tau.requires_grad

        x = x0 #that's the initial state
        xsimlist = []
        N_steps = tau.shape[0]
        sys.stdout.write('olsimulation: ')
        nan_detected = False
        for i in range(N_steps):
            tau_m = tau[i, :]
            x = self.fn(x[None,:],tau_m[None,:])[0,:]
            #check if any elements of x are NaN
            if torch.any(torch.isnan(x)):
                print('NaN detected')
                nan_detected = True
                break
            xsimlist.append(x[None,:])
            if (i%100)==99:
                sys.stdout.write('▄')
                sys.stdout.flush()
        sys.stdout.write('\n')
        xsim = torch.cat(xsimlist,dim=0)
        return xsim

    def apply_experiment(self, x0, tau):
        with torch.no_grad():
            xsim = self.forward(x0, tau)
        output_data = System_data(tau, xsim, dt=data_dt)
        return output_data

class robot_closed_loop_simulator(nn.Module):
    def __init__(self, fn, robot):
        super(robot_closed_loop_simulator, self).__init__()
        self.fn = fn
        self.robot = robot
        #self.K_vel_eff = torch.tensor([
        #    1046.41836177691538978251628577709197998046875,
        #    1137.639697096119334673858247697353363037109375,
        #    1073.424635846213504919433034956455230712890625,
        #    63.12320263748606663511964143253862857818603515625,
        #    68.44127465456364234341890551149845123291015625,
        #    44.07859945638431753422992187552154064178466796875
        #    ], device=tgt_device)
        #self.K_pos_eff = torch.tensor([
        #    572.9577951308232286464772187173366546630859375,
        #    572.9577951308232286464772187173366546630859375,
        #    572.9577951308232286464772187173366546630859375,
        #    572.9577951308232286464772187173366546630859375,
        #    572.9577951308232286464772187173366546630859375,
        #    572.9577951308232286464772187173366546630859375
        #    ], device=tgt_device)
        self.K_vel_eff = 25 * torch.tensor(5.0, device=tgt_device)
        self.K_pos_eff = 25 * torch.tensor(20.0, device=tgt_device)

        #u = torch.tensor([1798/7, 1872/7, 757/3, 221/1, 5032/21, 206793/1340], device=tgt_device)
        #u_inv = 1/u
        #K_vel = torch.tensor([1, 1, 1, 1, 1, 1], device=tgt_device) #proportional gain of the feedback velocity control
        #K_pos = torch.tensor([1, 1, 1, 1, 1, 1], device=tgt_device) #proportional gain of the feedback position control
        #self.K_vel_eff = K_vel
        #self.K_pos_eff = K_pos
        self.check_x_dot_inverse = False

    def forward(self, x0, x_ref):
        #x_ref is [N_steps, 18]: I need only 2 dimensions there
        #x0 is [12]
        #whenever you call me, make sure to use torch.no_grad

        x = x0 #that's the initial state
        xsimlist = []
        tausimlist = []
        tauffsimlist = []
        N_steps = x_ref.shape[0]
        sys.stdout.write('clsimulation: ')
        tau_ff_init = None
        nan_detected = False
        for i in range(N_steps):
            tau_ff = self.robot.x_dot_inverse(x_ref[(i,), 0:12], x_ref[(i,), 12:18])[0]
            #tau_ff = self.robot.x_dot_inverse(x_ref[((i+1)%N_steps,), 0:12], x_ref[((i+1)%N_steps,), 12:18])[0] #go 1 step ahead in the reference trajectory
            if self.check_x_dot_inverse:
                x_dot_result = self.robot.x_dot(x_ref[(i,), 0:12], tau_ff, False, self.robot.friction_params, self.robot.hwc_params, self.robot.dyn_params, self.robot.u_params)[0]
                if not torch.allclose(x_ref[i, 12:18], x_dot_result[6:12], atol=1e-6):
                    print('WARNING: x_dot_inverse is not the inverse of x_dot!')
                    sys.stdout.flush()
            tauffsimlist.append(tau_ff[None,:])
            if tau_ff_init is None: tau_ff_init = tau_ff
            #tau_m = (tau_ff - tau_ff_init) + self.K_vel_eff*( x_ref[i, 6:12] - x[6:12] ) + self.K_pos_eff*(x_ref[i, 0:6] - x[0:6])
            tau_m = tau_ff + self.K_vel_eff * (x_ref[i, 6:12] - x[6:12]) + self.K_pos_eff * (x_ref[i, 0:6] - x[0:6])
            xdsave = x_ref[i, 6:12] - x[6:12]
            xsimlist.append(x[None,:])
            tausimlist.append(tau_m[None,:])
            x = self.fn(x[None,:],tau_m[None,:])[0,:]
            #check if any elements of x are NaN
            if torch.any(torch.isnan(x)):
                print('NaN detected')
                nan_detected = True
                break
            if (i%100)==99:
                print('xd',xdsave.abs().max(),'@',i,'/',N_steps)
                #sys.stdout.write('▄')
                #sys.stdout.flush()
        sys.stdout.write('\n')
        xsim = torch.cat(xsimlist,dim=0)
        tausim = torch.cat(tausimlist,dim=0)
        tauffsim = torch.cat(tauffsimlist,dim=0)
        return (xsim, tausim, tauffsim)

    def apply_experiment(self, x0, x_ref):
        with torch.no_grad():
            xsim, tausim, tauffsim = self.forward(x0, x_ref)
        #plt.clf(); plt.plot(tauffsim); mkpng('temprect_tau_ff')
        output_data = System_data(tausim, xsim, dt=data_dt)
        return (output_data, tausim)

class H_eye(nn.Module):  # all states are measured
    def __init__(self, nx, ny):
        super().__init__()

    def forward(self, x):
        return x

class H_some_measurable(nn.Module):  # some states are measured
    def __init__(self, nx, ny, measurable_states):
        self.measurable_states = measurable_states
        assert ny == measurable_states.shape[0]
        super().__init__()

    def forward(self, x):
        if x.ndim == 1:
            return x[self.measurable_states]
        else:
            return x[:,self.measurable_states]


class E_ypast(nn.Module):
    def __init__(self, nb, nu, na, ny, nx):
        super().__init__()
        self.nx = nx
        self.ny = ny

    def forward(self, upast, ypast):
        # ypast[:,-1] #(Batch, na+na_right=1)
        # upast[:,-1] #(Batch, nb+nb_right=0)
        x = ypast[:, -1][:, None] if self.ny == None else ypast[:, -1]
        return x

class E_specify_initial_states(nn.Module):
    #for simulation it makes sense to create a dummy encoder that will give us the initial states that we want
    def __init__(self, nb, nu, na, ny, nx, x_initial_state):
        super().__init__()
        self.x_initial_state = x_initial_state
        assert x_initial_state.shape[1]==nx
    def forward(self, upast, ypast):
        return self.x_initial_state.repeat(upast.shape[0],1)

class my_nn(nn.Module): #my NN implementation that does not have a residual layer
    def __init__(self, n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        #linear + non-linear part
        super(my_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_hidden_layers>0:
            self.net_non_lin = deepSI.utils.feed_forward_nn(n_in,n_out,n_nodes_per_layer=n_nodes_per_layer,n_hidden_layers=n_hidden_layers,activation=activation)
        else:
            self.net_non_lin = None

    def forward(self,x):
        return self.net_non_lin(x)

class E_normalized_nn(nn.Module): #a simple FC net with a residual layer (default approach) #TUNEIT \/--activation
    def __init__(self, nb, nu, na, ny, nx, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, \
        u_norm_offsets=None, u_norm_cmults=None, y_norm_offsets=None, y_norm_cmults=None, x_norm_offsets=None, x_norm_cmults=None, \
        decorrelation_matrix=None, manual_decorrelation = False, only_velocities = False):

        super(E_normalized_nn, self).__init__()
        self.na = na
        self.nb = nb
        self.u_norm_offsets=torch.tensor(u_norm_offsets, dtype=torch.float32, device=tgt_device)
        self.u_norm_cmults=torch.tensor(u_norm_cmults, dtype=torch.float32, device=tgt_device)
        self.y_norm_offsets=torch.tensor(y_norm_offsets, dtype=torch.float32, device=tgt_device)
        self.y_norm_cmults=torch.tensor(y_norm_cmults, dtype=torch.float32, device=tgt_device)
        self.x_norm_offsets=torch.tensor(x_norm_offsets, dtype=torch.float32, device=tgt_device)
        self.x_norm_cmults=torch.tensor(x_norm_cmults, dtype=torch.float32, device=tgt_device)
        self.nu = tuple() if nu is None else ((nu,) if isinstance(nu,int) else nu)
        self.ny = tuple() if ny is None else ((ny,) if isinstance(ny,int) else ny)
        self.net = my_nn(n_in=nb*np.prod(self.nu,dtype=int) + na*np.prod(self.ny,dtype=int), n_out=nx if not only_velocities else nx//2, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)
        self.y_channel_filter = None #in simulation, we might get more channels in Y than what we trained the encoder for
        self.decorrelation_matrix = decorrelation_matrix
        self.manual_decorrelation = manual_decorrelation
        self.only_velocities = only_velocities

    def forward(self, upast, ypast):
        #this norming could be done in a much nicer way, but this is a prototype
        # note that:
        # u = c*(u+o)
        # y = c*(y+o)
        # x = c*x+o
        #nn_help = torch.cat( [\
        #    ypast[:,-1,0].reshape((ypast.shape[0],-1)),\
        #    torch.zeros(ypast.shape[0]),\
        #    ((ypast[:,-1,0]-ypast[:,-2,0])/data_dt).reshape((ypast.shape[0],-1)),\
        #    torch.zeros(ypast.shape[0])], axis=2)
        #nn_help = [ypast[:,-1,0], ypast[:,-1,0], (ypast[:,-1,0]-ypast[:,-2,0])/data_dt, (ypast[:,-1,0]-ypast[:,-2,0])/data_dt] )
        upastc = upast.clone()
        ypastc = ypast.clone()
        # ypastc[ █ ,█ ,█ ]
        #         |  |  |_ channel
        #         |  |____ sample
        #         |_______ batch
        #in the 4-state simulation, we will give the encoder more states than it was trained for, in that case it makes sense to filter these channels to the measurable_states used during training
        if self.y_channel_filter is not None: ypastc = ypastc[:, :, self.y_channel_filter]
        if self.decorrelation_matrix is not None:
            self.u_norm_cmults/=self.u_norm_cmults #make all elements 1
            self.y_norm_cmults/=self.y_norm_cmults

        assert (self.u_norm_cmults is not None) == (self.u_norm_offsets is not None)
        if self.u_norm_offsets is not None:
            if upastc.ndim==2:
                assert self.u_norm_offsets.ndim==0
                assert self.u_norm_cmults.ndim==0
                upastc = self.u_norm_cmults * (upastc + self.u_norm_offsets)
            else:
                assert upastc.shape[2]==len(self.u_norm_offsets)
                for i in range(len(self.u_norm_offsets)):
                    upastc[:,:,i] = self.u_norm_cmults[i] * (upastc[:,:,i] + self.u_norm_offsets[i])
        assert (self.y_norm_cmults is not None) == (self.y_norm_offsets is not None)
        if self.y_norm_offsets is not None:
            if ypastc.ndim==2:
                assert self.y_norm_offsets.ndim==0
                assert self.y_norm_cmults.ndim==0
                ypastc = self.y_norm_cmults * (ypastc + self.y_norm_offsets)
            else:
                assert ypastc.shape[2]==len(self.y_norm_offsets)
                for i in range(len(self.y_norm_offsets)):
                    ypastc[:,:,i] = self.y_norm_cmults[i] * (ypastc[:,:,i] + self.y_norm_offsets[i])

        #manual decorrelation of the measured output channel
        if self.manual_decorrelation:
            ypastcc = ypastc.clone()
            manual_decorrelation_factor_per_channel = [1.5*12, 1.2*12, 1.5*12, 1.5*12, 2*12, 1.5*12, 1.2*12, 1*12, 1.2*12, 1.2*12, 1.2*12, 1.2*12]
            for i_channel in range(ypastc.shape[2]):
                for i_sample in range(1,ypastc.shape[1]):
                    ypastc[:,i_sample-1,i_channel] = manual_decorrelation_factor_per_channel[i_channel]*(ypastcc[:,i_sample,i_channel] - ypastcc[:,i_sample-1,i_channel])

        net_in = torch.cat([upastc.view(upastc.shape[0],-1),ypastc.view(ypastc.shape[0],-1)],axis=1)
        if self.decorrelation_matrix is not None: net_in = net_in @ self.decorrelation_matrix.T
        net_out = self.net(net_in)
        if ypastc.shape[0] > 100:
            pass
            #PRINTIT for debug purposes: print mean and std of ypastc, upastc, net_out, and net_in
            #pprint(('ypastc mean|std|size:',float(ypastc.mean()),float(ypastc.std()),ypastc.shape, ' and channel/samplewise:', \
            #    [('channel='+str(i_channel), [('sample='+str(i_sample), float(ypastc[:,i_sample,i_channel].mean()),float(ypastc[:,i_sample,i_channel].std())) for i_sample in range(ypastc.shape[1])] \
            #    ) for i_channel in range(ypastc.shape[2])] ))
            #print('upastc mean|std:',float(upastc.mean()),float(upastc.std()))
            #we_have_nans = torch.any(net_out[:,i].std().isnan()) or torch.any(net_out[:,i].std().isnan())
            #print('NaNs:',we_have_nans)
            #print('net_in mean|std:',float(net_in.mean()),float(net_in.std()), ' and channel wise:',[(float(net_in[:,i].mean()),float(net_in[:,i].std())) for i in range(net_in.shape[1])])
            #print('net_in covariance:', net_in.T@net_in)
            #print('net_out mean|std:' ,float(net_out.mean()),float(net_out.std()),' and channel wise:',[(float(net_out[:,i].mean()),float(net_out[:,i].std())) for i in range(net_out.shape[1])])
        assert (self.x_norm_offsets is not None) == (self.x_norm_cmults is not None)
        if self.only_velocities:
            last_y = ypast[:, -1][:, None] if self.ny == None else ypast[:, -1]
            modified_net_out = torch.cat((last_y[:,0:len(self.x_norm_offsets)//2],net_out),dim=1)
            #print(modified_net_out.shape)
            net_out = modified_net_out
        if self.x_norm_offsets is not None:
            assert self.only_velocities or (net_out.shape[1]==len(self.x_norm_offsets))
            for i in range((0 if not self.only_velocities else len(self.x_norm_offsets)//2),len(self.x_norm_offsets)) :
                net_out[:,i] = (self.x_norm_cmults[i] * net_out[:,i] + self.x_norm_offsets[i]) #*([0.7,0.7,data_dt,data_dt][i]) #TUNEIT some additional tweaking of the normalization
        return net_out



#     _                          _        _                                _   _   _  classes, helpers, settings
#  __| |__ _ ______ ___ ___     | |_  ___| |_ __  ___ _ _ ___      ___ ___| |_| |_(_)_ _  __ _ ___
# / _| / _` (_-<_-</ -_|_-<  _  | ' \/ -_) | '_ \/ -_) '_(_-<  _  (_-</ -_)  _|  _| | ' \/ _` (_-<
# \__|_\__,_/__/__/\___/__/ ( ) |_||_\___|_| .__/\___|_| /__/ ( ) /__/\___|\__|\__|_|_||_\__, /__/
#                           |/             |_|                |/                         |___/

def ply(what):
    if type(what) is torch.Tensor or type(what) is np.ndarray:
        if type(what) is np.ndarray: what = torch.tensor(what)
        if len(what.shape) == 1:
            what = what[:,None]
        fig = make_subplots(rows=what.shape[1], cols=1)
        for i in range(what.shape[1]):
            fig.add_trace(go.Scatter(y=what[:,i].detach()))
        fig.show()

def diff_fd(x, u=None, dim=0, dt=0.004): 
    assert dim==0, "Only dim=0 is supported for now"
    if u is None: u = torch.zeros_like(x)
    F_x = torch.fft.rfft(x, dim=dim)
    F_u = torch.fft.rfft(u, dim=dim)

    w = ((((torch.arange(0,(x.shape[dim]//2+1)))/x.shape[dim])*(2*torch.pi))) #TODO to make it general
    j = torch.tensor(complex(0,1))
    F_d_x = (w*j)[None,:].T*F_x
    F_dd_x = (-w*w)[None,:].T*F_x
    
    F_x[filter_from:,:] = 0
    F_d_x[filter_from:,:] = 0
    F_dd_x[filter_from:,:] = 0
    F_u[filter_from:,:] = 0
    x = torch.fft.irfft(F_x,dim=dim)
    d_x = torch.fft.irfft(F_d_x,dim=dim)*(1/dt)
    dd_x = torch.fft.irfft(F_dd_x,dim=dim)*(1/dt)**2
    u = torch.fft.irfft(F_u,dim=dim)
    return (x, d_x, dd_x, u)

class integrator_with_intermediate_steps(nn.Module):
    def __init__(self, integrator, intermediate_steps): #If intermediate_steps=1, then it's the same as the plain integrator. No extra step is taken in between samples.
        super(integrator_with_intermediate_steps, self).__init__()
        self.integrator = integrator
        self.intermediate_steps = intermediate_steps


    def forward(self, x, u): #u constant on segment, zero-order hold
        self.integrator.dt /= self.intermediate_steps
        for i in range(self.intermediate_steps):
            x = self.integrator.forward(x,u)
        self.integrator.dt *= self.intermediate_steps
        return x

    @torch.jit.export
    def simulate(self, x, u, N_steps):
        y = x #that's the initial state
        ysimlist = []
        for i in range(N_steps):
            y = self.forward(y,u[:,i,:])
            ysimlist.append(y[:,None,:])
        ysim = torch.cat(ysimlist,dim=1)
        return ysim

    @property
    def dt(self):
        return self.integrator.dt
    @dt.setter
    def dt(self,dt):
        self.integrator.dt = dt

    @property
    def param_group_kwargs(self):
        return self.integrator.param_group_kwargs

def plot_yderivn(system, data, state_name, show=False):

    yfuture = torch.tensor(data.y, dtype=torch.float32)[None,:,:]
    ufuture = torch.tensor(data.u, dtype=torch.float32)[None,:,:]

    with torch.no_grad():
        N_phaseshifts = 1 #okay so phase shifting here -\/- is basically switched off
        N_samples = 15000
        N_half_samples_p1 = N_samples//2+1
        w = ((((torch.arange(0,N_half_samples_p1))/N_samples)*(2*torch.pi))).to(tgt_device)
        j = torch.tensor(complex(0,1),device=tgt_device)
        phaseshift_amounts = torch.cat((torch.tensor([0.0],device=tgt_device),torch.rand(N_phaseshifts-1,device=tgt_device)*N_samples))
        phaseshift_mult = torch.exp(phaseshift_amounts.unsqueeze(1).unsqueeze(2).repeat(1,N_half_samples_p1,6)*j*w.unsqueeze(0).unsqueeze(2).repeat(N_phaseshifts,1,6))

        F_yfuture_p = torch.fft.rfft(yfuture[:,:,0:6], dim=1)
        F_yfuture_p[:,filter_from:,:] = 0 #filtering

        yfuture_p = yfuture[:,:,0:6]
        yfuture_p_phaseshifted = torch.fft.irfft(F_yfuture_p.repeat((N_phaseshifts,1,1)), dim=1)

        F_yfuture_v = (w*j)[None,:].T*F_yfuture_p
        yfuture_v_phaseshifted = torch.fft.irfft(F_yfuture_v.repeat((N_phaseshifts,1,1)),dim=1)*(1/data_dt)

        F_yfuture_a = (-w*w)[None,:].T*F_yfuture_p
        yfuture_a_phaseshifted = torch.fft.irfft(F_yfuture_a.repeat((N_phaseshifts,1,1)),dim=1)*(1/data_dt)**2
        yfuture_a_phaseshifted_cut = yfuture_a_phaseshifted[:,N_samples_cutoff:N_samples-N_samples_cutoff,:]

        yfuture_p_v_phaseshifted = torch.cat((yfuture_p_phaseshifted,yfuture_v_phaseshifted),dim=2) #this time we calculate the derivatives of the positions ourselves with FFT, using trig. interpolation

        F_ufuture = torch.fft.rfft(ufuture, dim=1)
        F_ufuture[:,filter_from:,:] = 0 #filtering
        ufuture_phaseshifted = torch.fft.irfft(F_ufuture.repeat((N_phaseshifts,1,1)), dim=1)

        yfuture_p_v_phaseshifted_cut = yfuture_p_v_phaseshifted[:,N_samples_cutoff:N_samples-N_samples_cutoff,:]
        ufuture_phaseshifted_cut = ufuture_phaseshifted[:,N_samples_cutoff:N_samples-N_samples_cutoff,:]

        yderivn_v_a = system.derivn(torch.cat((yfuture_p_phaseshifted, yfuture_v_phaseshifted), dim=2).reshape((-1,12)),ufuture_phaseshifted_cut.reshape((-1,6))).reshape(yfuture_p_v_phaseshifted_cut.shape)
        yderivn_a = yderivn_v_a[:,:,6:12]

        #create a new plot with plotly, 6 subplots, based on these above
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02)
        for i in range(6):
            fig.add_trace(go.Scatter(y=yderivn_a[0,:,i].detach().cpu().numpy(), name='yderivn_a'), row=1+i, col=1)
            fig.add_trace(go.Scatter(y=yfuture_a_phaseshifted_cut[0,:,i].detach().cpu().numpy(), name='yfuture_a_phaseshifted_cut'), row=1+i, col=1)
            fig.add_trace(go.Scatter(y=(yderivn_a[0,:,i]-yfuture_a_phaseshifted_cut[0,:,i]).detach().cpu().numpy(), name='yderivn_a-yfuture_a_phaseshifted_cut'), row=1+i, col=1)
        fig.update_layout(title='yderivn_a vs yfuture_a_phaseshifted_cut', xaxis_title='sample id', yaxis_title='acceleration')
        fig.write_image(f"{state_name}_yderivn_a_vs_yfuture_a_phaseshifted_cut.png", width=1000, height=1000)
        fig.write_html(f"{state_name}_yderivn_a_vs_yfuture_a_phaseshifted_cut.html")
        if show: fig.show()

def plot_training_validation_loss_history(system, state_name='custom', show=False):
    #plot the same as above, just on semilogy scale:
    fig = go.Figure()
    #src: plt.semilogy(sys_trained.batch_id,sys_trained.Loss_val,label='validation loss')
    fig.add_trace(go.Scatter(x=system.batch_id, y=system.Loss_val, name='validation loss'))
    fig.update_layout(title=state_name+': validation loss', xaxis_title='batch id (number of updates)', yaxis_title='error', yaxis_type="log")
    if show: fig.show()
    fig.write_image(f"{state_name}_validation_loss.png", width=1000, height=1000)
    fig.write_html(f"{state_name}_validation_loss.html")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=system.batch_id, y=system.Loss_train, name='train loss'))
    fig.update_layout(title=state_name+': train loss', xaxis_title='batch id (number of updates)', yaxis_title='error', yaxis_type="log")
    if show: fig.show()
    fig.write_image(f"{state_name}_train_loss.png", width=1000, height=1000)
    fig.write_html(f"{state_name}_train_loss.html")
    to_return = {
        'first_loss_val': system.Loss_val[0],
        'last_loss_val': system.Loss_val[-1],
        'num_loss_val': system.Loss_val.shape[0],
        'first_loss_train': system.Loss_train[1], #first item is nan (at least once I checked)
        'last_loss_train': system.Loss_train[-1],
        'num_loss_train': system.Loss_train.shape[0],
        }
    return to_return
    
def plot_many_simulations(n_step_nrms_simulation_data, data, state_name, n_sim_steps = None):
    if n_sim_steps is None:
        n_sim_steps = n_step_nrms_simulation_data['y_window'].shape[1]
    else: n_sim_steps += 1

    #plot r-squared per simulation, per channel:
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
    for i in range(6):
        fig.add_scatter(y=n_step_nrms_simulation_data['r_squared_per_sim'][:,i].cpu().detach().numpy(), row=i+1, col=1, name='r2['+str(i+1)+']: '+"{:.3e}".format(n_step_nrms_simulation_data['r_squared_per_channel'][i].item()))
    fig.write_html(f'{state_name}_{n_sim_steps-1}_r_squared_per_sim.html')
    torch.save(n_step_nrms_simulation_data['r_squared_per_sim'], f"{state_name}_{n_sim_steps-1}_r_squared_per_sim.torch_save")

    # plot n-step-nrms per simulation, per channel:
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
    for i in range(6):
        fig.add_scatter(y=n_step_nrms_simulation_data['r'].mean(axis=1)[:,i].cpu().detach().numpy(), row=i+1, col=1, line_color='green')

    #for i in range(1, 7):  # for each row
    #    fig.update_yaxes(tickformat=".2e", row=i, col=1)

    # Remove the legend
    fig.update_layout(showlegend=False)

    # Set common labels
    fig.update_xaxes(title_text="# sample", row=6, col=1)
    fig.add_annotation(
        text="error",
        xref="paper", yref="paper",
        x=-0.047, y=0.5, # these values may need tweaking
        showarrow=False,
        font=dict(size=14),
        xanchor='center', yanchor='middle',
        textangle=-90
        )

    fig.write_html(f'{state_name}_{n_sim_steps-1}_nsn_per_sim.html')
    fig.update_layout(autosize=False, width=1250, height=640)
    fig.write_image(f'{state_name}_{n_sim_steps-1}_nsn_per_sim.svg')

    """
    #plot n-step-nrms per simulation, in 3D:
    # Assuming that n_step_nrms_simulation_data['r'] is a 3D tensor with shape (number_of_simulations, number_of_timesteps, number_of_channels)
    n_step_nrms_simulation_data_r = n_step_nrms_simulation_data['r'].cpu().detach().numpy()
    fig = go.Figure()

    # Generate x, y grid (this depends on your actual values)
    x = np.arange(n_step_nrms_simulation_data_r.shape[0])  # simulations
    y = np.arange(n_step_nrms_simulation_data_r.shape[1])  # time steps

    for i in range(n_step_nrms_simulation_data_r.shape[2]):  # loop over channels
        z = n_step_nrms_simulation_data_r[:,:,i]  # select data for specific channel
        fig.add_trace(go.Surface(x=x, y=y, z=z, name='channel #'+str(i+1)))

    fig.update_layout(
        title=f'{state_name}_{n_sim_steps-1}_nsn_per_sim',
        scene = dict(
            xaxis_title='Simulation',
            yaxis_title='Time Step',
            zaxis_title='n_step_nrms'
        ),
        autosize=False,
        width=1000,
        height=800,
    )

    fig.write_html(f'{state_name}_{n_sim_steps-1}_nsn_per_sim_3d.html')
    """
    
    # plot short simulations on data, per channel:
    fig = make_subplots(rows=6, cols=2)
    for ch_i in range(12):
        for sim_start_place_i in range(0, n_step_nrms_simulation_data['y_window'].shape[0], round((n_sim_steps-1)*(2/3))):
            fig.add_trace( go.Scatter( x=np.arange(n_sim_steps)+sim_start_place_i, y=n_step_nrms_simulation_data['y_window'][sim_start_place_i,0:n_sim_steps,ch_i], mode='lines', line=dict(color='rgba(0,0,0,1)') ), row=1+(ch_i%6), col=1+(ch_i//6) )
            fig.add_trace( go.Scatter( x=np.arange(n_sim_steps)+sim_start_place_i, y=n_step_nrms_simulation_data['y_stack'][sim_start_place_i,0:n_sim_steps,ch_i], mode='lines', line=dict(color='rgba(255,0,0,1)' if (paper_case == 3) and (sim_start_place_i < stiction_pad or sim_start_place_i > 15000-stiction_pad) else 'rgba(0,0,255,1)') ), row=1+(ch_i%6), col=1+(ch_i//6) )
        #fig.update_yaxes(tickformat=".2e", row=1+(ch_i%6), col=1+(ch_i//6))

    # Set common labels
    fig.update_xaxes(title_text="# sample", row=6, col=1)
    fig.update_xaxes(title_text="# sample", row=6, col=2)

    # Add common y-axis labels using annotations
    fig.add_annotation(
        text="angular positions [rad]",
        xref="paper", yref="paper",
        x=-0.035, y=0.5, # these values may need tweaking
        showarrow=False,
        font=dict(size=14),
        xanchor='center', yanchor='middle',
        textangle=-90
    )
    fig.add_annotation(
        text="angular velocities [rad/sec]",
        xref="paper", yref="paper",
        x=0.513, y=0.5, # these values may need tweaking
        showarrow=False,
        font=dict(size=14),
        xanchor='center', yanchor='middle',
        textangle=-90
    )

    fig.update_layout(showlegend=False)
    fig.write_html(f'{state_name}_{n_sim_steps-1}_short_sims.html')
    fig.update_layout( autosize=False, width=1250, height=640)
    fig.write_image(f'{state_name}_{n_sim_steps-1}_short_sims.svg')

def plot_n_step_nrms(system, data, plot_steps, state_name, show, return_these=None, extra_plots=False):
    tic = time() 
    sys.stdout.write('plot_n_step_nrms @ '+state_name+': calculating... '); sys.stdout.flush()
    n_step_nrms_simulation_data = system.andras_n_step_nrms(data, plot_steps, weigh_by_y_var=True, one_number=False, return_simulation_data=True)
    n_step_nrms = n_step_nrms_simulation_data['to_return']
    n_step_nrms_chmean = n_step_nrms.mean(dim=1)
    sys.stdout.write('done, elapsed time: '+str(time()-tic)+'s\n')
    sys.stdout.flush()
    #what will be the size and meaning of the output? [16,12] --> [plot_steps, channels]
    #plot 2D array n_step_nrms along dim=0 on the x axis, and dim=1 are separate lines on the same plot:
    fig = go.Figure()
    if n_step_nrms is not None:
        for i in range(6):
            fig.add_trace(go.Scatter(x=np.arange(n_step_nrms.shape[0]), y=n_step_nrms[:,i], name='joint '+str(i)))
    fig.add_trace(go.Scatter(x=np.arange(n_step_nrms_chmean.shape[0]), y=n_step_nrms_chmean, name='mean over channels'))
    fig.update_layout(title=state_name+f": {plot_steps}-step-NRMS", xaxis_title='step', yaxis_title='error')
    if show: fig.show()
    fig.write_image(f"{state_name}_{plot_steps}_step_nrms.png", width=1000, height=1000)
    fig.write_html(f"{state_name}_{plot_steps}_step_nrms.html")
    print(f"{state_name} {plot_steps}-step-nrms mean = {n_step_nrms_chmean[-1]}")

    #make extra plots:
    if extra_plots: 
            for n_sim_steps in ( return_these if return_these is not None else [None] ):
                plot_many_simulations(n_step_nrms_simulation_data, data, state_name, n_sim_steps)

    if return_these is not None:
        to_return = dict()
        to_return['rsq'] = n_step_nrms_simulation_data['r_squared_per_channel'].mean()
        for i in return_these:
            to_return[i] = n_step_nrms_chmean[i]
        return to_return

    #torch.save(n_step_nrms_simulation_data, f"{state_name}_n_step_nrms_simulation_data.torch_save")

def plot_full_system_analysis(system, training_data=None, validation_data=None, test_data=None, state_name='custom', plot_loss_history=True, show=False, without_nn=True, with_nn=True, extra_plots=False):
    if plot_loss_history: 
        to_return_loss_hist = plot_training_validation_loss_history(system, state_name=state_name, show=show)
    else: to_return_loss_hist = None
    
    def full_system_analysis(state_name):
        to_return = dict()
        if training_data is not None: 
            #to_return['training'] = {mon_steps[0]:0, mon_steps[1]:0}
            to_return['training'] = plot_n_step_nrms(system, training_data, mon_steps[1], state_name=state_name+"_training", show=show, return_these=[mon_steps[0],mon_steps[1]])
            to_return['training']['loss'] = system.cal_validation_error_derivn(training_data)
            plot_yderivn(system, training_data, state_name=state_name+"_training", show=show)
        if validation_data is not None: 
            #to_return['validation'] = {mon_steps[0]:0, mon_steps[1]:0}
            to_return['validation'] = plot_n_step_nrms(system, validation_data, mon_steps[1], state_name=state_name+"_validation", show=show, return_these=[mon_steps[0],mon_steps[1]])
            to_return['validation']['loss'] = system.cal_validation_error_derivn(validation_data)
            plot_yderivn(system, validation_data, state_name=state_name+"_validation", show=show)
        if test_data is not None: 
            to_return['test'] = plot_n_step_nrms(system, test_data, mon_steps[1], state_name=state_name+"_test", show=show, return_these=[mon_steps[0],mon_steps[1]], extra_plots=extra_plots)
            to_return['test']['loss'] = system.cal_validation_error_derivn(test_data)
            plot_yderivn(system, test_data, state_name=state_name+"_test", show=show)
        return to_return

    to_return = None
    if with_nn:
        if system.derivn.robustify_nn:
            to_return = full_system_analysis(state_name=state_name+'_with_nn')
            to_return['with_nn'] = True
        else:
            print('plot_full_system_analysis: warning: sys_trained.derivn.robustify_nn is switched off, skipping with_nn plots')
    if without_nn: 
        system.derivn.disable_nn = True
        to_return_temp = full_system_analysis(state_name=state_name+'_without_nn')
        if to_return is None:
            to_return = to_return_temp
            to_return['with_nn'] = False
        system.derivn.disable_nn = False
        to_return['without_nn_results'] = to_return_temp
    to_return['loss_history'] = to_return_loss_hist
    return to_return

class SS_encoder_deriv_weighted_general(deepSI.fit_systems.SS_encoder_deriv_general):
    def __init__(self, obj_weights=None, lambda_encoder_in_obj=None, intermediate_steps=1, nx=10, na=20, nb=20, feedthrough=False, f_norm=0.1, dt_base=1., cut_off=float('inf'), \
                e_net=deepSI.fit_systems.encoders.default_encoder_net, f_net=deepSI.fit_systems.encoders.default_state_net, \
                integrator_net=deepSI.utils.integrator_RK4, h_net=deepSI.fit_systems.encoders.default_output_net, \
                e_net_kwargs={}, f_net_kwargs={}, integrator_net_kwargs={}, h_net_kwargs={}, na_right=0, nb_right=0, robustify_fd=False, n_step_nrms_mode=False):
        super(SS_encoder_deriv_weighted_general, self).__init__(nx=nx, na=na, nb=nb, feedthrough=feedthrough, f_norm=f_norm, dt_base=dt_base, cut_off=cut_off, \
                e_net=e_net, f_net=f_net, integrator_net=integrator_net, h_net=h_net, \
                e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, integrator_net_kwargs=integrator_net_kwargs, h_net_kwargs=h_net_kwargs,\
                na_right=na_right, nb_right=nb_right)
        self.obj_weights = obj_weights #this is used in old loss()
        self.lambda_encoder_in_obj = lambda_encoder_in_obj #this is used in old loss()
        self.intermediate_steps = intermediate_steps
        self.n_step_nrms_mode = n_step_nrms_mode
        self.nn_weight_decay = 0 if not from_exp_dir else exp_weight_decay

    def init_nets(self, nu, ny):
        if hasattr(self,'fn'):
            print('WARNING: init_nets already done, skipping')
            return
        #from: SS_encoder_general
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        self.encoder = self.e_net(nb=(self.nb+nb_right), nu=nu, na=(self.na+na_right), ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.fn =      self.f_net(nx=self.nx, nu=nu,                                **self.f_net_kwargs)
        #self.fn = torch.jit.script(self.fn) #TODO #False #IMPORTANT
        #self.encoder = torch.jit.script(self.encoder)
        if self.feedthrough:
            self.hn =      self.h_net(nx=self.nx, ny=ny, nu=nu,                     **self.h_net_kwargs)
        else:
            self.hn =      self.h_net(nx=self.nx, ny=ny,                            **self.h_net_kwargs)

        #from: SS_encoder_deriv_general
        self.derivn = self.fn  #move fn to become the derivative net
        self.excluded_nets_from_parameters = ['derivn']
        self.fn = self.integrator_net(self.derivn, f_norm=self.f_norm, dt_base=self.dt_base, **self.integrator_net_kwargs) #has no torch parameters?

        #we have copied all the initializations here on purpose
        #super(SS_encoder_deriv_weighted_general, self).init_nets(nu,ny)
        self.fn = integrator_with_intermediate_steps(self.fn, self.intermediate_steps)

    def cal_validation_error_derivn(self, val_sys_data, validation_measure=''):
        N_samples = 15000
        N_half_samples_p1 = N_samples//2+1
        N_samples_cutoff = 0 #LOOKOUT this is defined at 3 places, 2x here and also in fit_system.py in deepSI
        yfuture = torch.tensor(val_sys_data.y, dtype=torch.float32)[None,:,:]
        ufuture = torch.tensor(val_sys_data.u, dtype=torch.float32)[None,:,:]
        assert val_sys_data.y.shape[0]==N_samples

        with torch.no_grad():
            # <copy_and_paste_from_loss> 📄

            w = ((((torch.arange(0,N_half_samples_p1))/N_samples)*(2*torch.pi))).to(tgt_device)
            j = torch.tensor(complex(0,1),device=tgt_device)
            N_phaseshifts = 1 #here the phaseshifting is basically switched off
            phaseshift_amounts = torch.cat((torch.tensor([0.0],device=tgt_device),torch.rand(N_phaseshifts-1,device=tgt_device)*N_samples))
            phaseshift_mult = torch.exp(phaseshift_amounts.unsqueeze(1).unsqueeze(2).repeat(1,N_half_samples_p1,6)*j*w.unsqueeze(0).unsqueeze(2).repeat(N_phaseshifts,1,6))

            F_yfuture_p = torch.fft.rfft(yfuture[:,:,0:6], dim=1)
            F_yfuture_p[0,filter_from:,:] = 0
            yfuture_p_phaseshifted = torch.fft.irfft(phaseshift_mult*F_yfuture_p.repeat((N_phaseshifts,1,1)), dim=1)

            F_yfuture_v = (w*j)[None,:].T*F_yfuture_p
            yfuture_v_phaseshifted = torch.fft.irfft(phaseshift_mult*F_yfuture_v.repeat((N_phaseshifts,1,1)),dim=1)*(1/data_dt)

            F_yfuture_a = (-w*w)[None,:].T*F_yfuture_p
            yfuture_a_phaseshifted = torch.fft.irfft(phaseshift_mult*F_yfuture_a.repeat((N_phaseshifts,1,1)),dim=1)*(1/data_dt)**2

            yfuture_p_v_phaseshifted = torch.cat((yfuture_p_phaseshifted,yfuture_v_phaseshifted),dim=2) #this time we calculate the derivatives of the positions ourselves with FFT, using trig. interpolation

            F_ufuture = torch.fft.rfft(ufuture, dim=1)
            F_ufuture[0,filter_from:,:] = 0
            ufuture_phaseshifted = torch.fft.irfft(phaseshift_mult*F_ufuture.repeat((N_phaseshifts,1,1)), dim=1)

            yderivn_v_a = self.derivn(yfuture_p_v_phaseshifted.reshape((-1,12)),ufuture_phaseshifted.reshape((-1,6))).reshape(yfuture_p_v_phaseshifted.shape)
            yderivn_a = yderivn_v_a[:,:,6:12]

            # </copy_and_paste_from_loss> 📄

            diff_final = yfuture_a_phaseshifted - yderivn_a
            #return torch.mean((diff_final**2).reshape(-1)).item()
            diff_center =  diff_final[:,stiction_pad:-stiction_pad,:] if paper_case == 3 else diff_final
            return torch.mean(  (torch.mean((diff_final**2),dim=1)[0,:]*(1/self.derivn.a_std**2))   ) # + constraint_val

        #plot yderivn_a and yfuture_a_phaseshifted_cut on each other, using plotly on 6 subplots
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
        for i in range(6):
            #fig.add_scatter(y=self.derivn(torch.cat((yfuture_p_phaseshifted, yfuture_v_phaseshifted), dim=2).reshape((-1,12)),ufuture_phaseshifted_cut.reshape((-1,6)))[:,6+i].cpu().detach().numpy(), row=i+1, col=1, name='a*=PHY([x_sim, v_dx], tau_sim)')
            fig.add_scatter(y=yderivn_a[0,:,i].cpu().detach().numpy(), row=i+1, col=1, name='a*=PHY([x_sim, v_dx], tau_sim)')
            fig.add_scatter(y=yfuture_a_phaseshifted[0,:,i].cpu().numpy(), row=i+1, col=1, name='a_ddx')
        fig.show(renderer='firefox')

    @staticmethod
    def log_barrier_multi(z_multi, t):
        retsum = torch.tensor(0.0, device=tgt_device)
        for z in z_multi:
            retsum = retsum + SS_encoder_deriv_weighted_general.log_barrier(z,t) #LOOKOUT in-place operation! 
        return retsum

    @staticmethod
    def log_barrier(z, t):
        assert z.shape == ()
        return -(1/t)*torch.log(-z) if z <= -(1/(t**2)) else t*z-(1/t)*torch.log(1/(t**2))+(1/t)
        #s.t. k_nlspring3 <= 5000
        #constraint_val = self.log_barrier(self.derivn.param_scale['k3_nlspring']*param_groups[1]['params'][4]-5000, torch.tensor(1000))

    def dump_params(self, file=sys.stdout):
        print('abs max [0] friction parameters:  ', self.optimizer.param_groups[0]['params'][0].abs().max().item(), file=file)
        print('abs max [1] dynamical parameters: ', self.optimizer.param_groups[0]['params'][1].abs().max().item(), file=file)
        print('abs max [2] hwc parameters:       ', self.optimizer.param_groups[0]['params'][2].abs().max().item(), file=file)
        print('abs max [3] u parameters:         ', self.optimizer.param_groups[0]['params'][3].abs().max().item(), file=file)

        #set printoptions to print all elements of a tensor, with full floating point precision:
        np.set_printoptions(precision = None, threshold = 1000)
        print('f_vis  =', self.optimizer.param_groups[0]['params'][0][0:6].data, file=file)
        print('f_coul =', self.optimizer.param_groups[0]['params'][0][6:12].data, file=file)
        print('f_a    =', self.optimizer.param_groups[0]['params'][0][12:18].data, file=file)
        print('f_b    =', self.optimizer.param_groups[0]['params'][0][18:24].data, file=file)
        print('f_asym =', self.optimizer.param_groups[0]['params'][0][24:30].data, file=file)
        #The elements of tensor x are printed one by one, with num=1 as: "L_1xx = 0.1, L_1xy=0.1, L_1xz=0.1, L_1yy=0.1, L_1yz=0.1, L_1zz=0.1, l_1x=0.1, l_1y=0.1, l_1z=0.1, m_1=0.1, Ia_1=0.1". Make this formatting for me:
        def print_dyn(x, num):
            print('L_'+str(num)+'xx =', x[0], ', L_'+str(num)+'xy =', x[1], ', L_'+str(num)+'xz =', x[2], ', L_'+str(num)+'yy =', x[3], ', L_'+str(num)+'yz =', x[4], ', L_'+str(num)+'zz =', x[5], ', l_'+str(num)+'x =', x[6], ', l_'+str(num)+'y =', x[7], ', l_'+str(num)+'z =', x[8], ', m_'+str(num)+' =', x[9], ', Ia_'+str(num)+' =', x[10], file=file)
        for i in range(6):
            try:
                print_dyn(self.optimizer.param_groups[0]['params'][1][(11*i):(11*i+11)].detach().numpy(), i+1)
            except:
                pass

    def andras_n_step_nrms(self, data, n_steps, weigh_by_y_var=True, one_number=False, return_simulation_data=False, mean_channels=False):
        with torch.no_grad():
            y=torch.tensor(data.y)
            u=torch.tensor(data.u)
            #n_timeswecando = y.shape[0]-n_steps-1 #the amount of times the window of n_steps fits in the length of the data [[XXXX].......] 
            sliding_window = torch.arange(0, y.shape[0]).unfold(0,n_steps+1,1)
            y_window = y[sliding_window, :] #[n_timeswecando x n_steps+1 x n_states]
            u_window = u[sliding_window, :]
            y_current = y_window[:,0,:]
            y_stack = []
            y_stack.append(y_current)
            for i in range(n_steps):
                if (i%50==0): sys.stdout.write('▁')
                y_current = self.fn(y_current, u_window[:,i,:])
                y_stack.append(y_current)
            y_stack = torch.stack(y_stack, dim=1)
            
            r_sqr = (y_stack[:,:,0:6]-y_window[:,:,0:6])**2
            if weigh_by_y_var: r = r_sqr / y.std(dim=0).pow(2)[None,None,0:6].repeat(y_stack.shape[0],y_stack.shape[1],1)
            if one_number:
                to_return = torch.mean(r.reshape((-1,)))
            elif mean_channels: 
                to_return = torch.mean(torch.mean(r,dim=0),dim=1)
            else: 
                r_center = r[stiction_pad:-stiction_pad,:,:] if paper_case == 3 else r
                to_return = torch.mean(r_center,dim=0)
                print('BEWARE: using hardcoded slice of data for andras_n_step_nrms')
            if return_simulation_data:
                r_sqr = (y_stack[:,:,0:12]-y_window[:,:,0:12])**2
                r_squared_top = r_sqr.sum(dim=1) #n_timeswecando, channels(6)
                r_squared_bottom = ((y_window[:,:,0:12]-y_window[:,:,0:12].mean(dim=1)[:,None,:].repeat((1,n_steps+1,1)))**2).sum(dim=1)
                r_squared_per_sim = 100*(1 - (r_squared_top / r_squared_bottom))
                r_squared_per_channel = r_squared_per_sim.mean(dim=0)
                return {
                    'to_return':to_return,
                    'y_stack':y_stack,
                    'y_window':y_window,
                    'u_window':u_window,
                    'r':r,
                    'r_squared_per_sim':r_squared_per_sim,
                    'r_squared_per_channel':r_squared_per_channel
                    }
            else:
                return to_return

    def loss(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        """
        this is the acceleration-based loss
        """
        epoch = Loss_kwargs['epoch']

        #LOOKOUT this is for training first theta_phy/f_phy only for epoch_patience_min epochs, then afterwards for the ANN parameters
        if epoch >= epoch_patience_min:
            self.derivn.robustify_nn = True
        elif epoch > 1:
            self.derivn.robustify_nn = False
        
        self.derivn.inside_loss = True
        N_samples = 15000
        N_half_samples_p1 = N_samples//2+1
        N_samples_cutoff = 0

        fd_mode = False
        td_mode = True
        n_step_nrms_mode = self.n_step_nrms_mode
        n_steps = 10 #PAPER
        n_batch = 75*5
        force_cartoon = True
        def get_the_diff():

            ann_rake = False #IMPORTANT
            random_phase_on = False if not from_exp_dir else exp_obj_counter>=2
            one_random_phase = True

            epoch = Loss_kwargs['epoch']
            if self.derivn.robustify_nn and ann_rake:
                #if epoch%250==1:
                #    self.bestfit = 1e10
                #    print('#### reset validation loss ####')

                #epoch_modulo = epoch%3000
                #if epoch_modulo < 250:
                #    self.derivn.ann_influence=0.5
                #    self.derivn.robot_influence=2
                #elif epoch_modulo < 500:
                #    self.derivn.ann_influence=1
                #    self.derivn.robot_influence=1
                #elif epoch_modulo < 750:
                #    self.derivn.ann_influence=2
                #    self.derivn.robot_influence=0.5
                #else:
                #    self.derivn.ann_influence=0
                #    self.derivn.robot_influence=1

                if epoch%1000==1:
                    self.bestfit = 1e10
                    print('#### reset validation loss ####')
                """
                epoch_modulo = epoch%5000
                if 0 < epoch_modulo <= 1000:
                    self.derivn.ann_influence=1
                elif epoch_modulo <= 1300:
                    self.derivn.ann_influence=.2
                elif epoch_modulo <= 1700:
                    self.derivn.ann_influence=.4
                elif epoch_modulo <= 2000:
                    self.derivn.ann_influence=.6
                elif epoch_modulo <= 2300:
                    self.derivn.ann_influence=.8
                elif epoch_modulo <= 2700:
                    self.derivn.ann_influence=1
                elif epoch_modulo <= 3000:
                    self.derivn.ann_influence=1.2
                elif epoch_modulo <= 4300:
                    self.derivn.ann_influence=1.4
                elif epoch_modulo <= 3700:
                    self.derivn.ann_influence=1.6
                elif epoch_modulo <= 4000:
                    self.derivn.ann_influence=1.8
                else:
                    self.derivn.ann_influence=0
                """
                #read ann_influence from ann_influence.txt
                with open('ann_influence.txt', 'r') as f:
                    self.derivn.ann_influence = float(f.read())

                #original:
                #epoch_modulo = epoch%3000
                #if epoch_modulo < 1000:
                #    self.derivn.ann_influence=1
                #elif epoch_modulo < 2000:
                #    self.derivn.ann_influence=2
                #else:
                #    self.derivn.ann_influence=0

                print('loss :: ann_influence = ', self.derivn.ann_influence, ' robot_influence = ', self.derivn.robot_influence)
            if True:
                with torch.no_grad():
                    w = ((((torch.arange(0,N_half_samples_p1))/N_samples)*(2*torch.pi))).to(tgt_device)
                    j = torch.tensor(complex(0,1),device=tgt_device)
                    if random_phase_on:
                        if one_random_phase:
                            N_phaseshifts = 1
                            phaseshift_amounts = torch.rand(N_phaseshifts,device=tgt_device)*N_samples
                        else:
                            N_phaseshifts = 1 #the batch size
                            phaseshift_amounts = torch.cat((torch.tensor([0.0],device=tgt_device),torch.rand(N_phaseshifts-1,device=tgt_device)*N_samples))
                    else:
                        N_phaseshifts = 1
                        phaseshift_amounts = torch.tensor([0.0],device=tgt_device)
                    phaseshift_mult = torch.exp(phaseshift_amounts.unsqueeze(1).unsqueeze(2).repeat(1,N_half_samples_p1,6)*j*w.unsqueeze(0).unsqueeze(2).repeat(N_phaseshifts,1,6))

                    F_yfuture_p = torch.fft.rfft(yfuture[:,:,0:6], dim=1)
                    F_yfuture_p[:,filter_from:,:] = 0 #filtering
                    yfuture_p_phaseshifted = torch.fft.irfft(phaseshift_mult*F_yfuture_p.repeat((N_phaseshifts,1,1)), dim=1)

                    F_yfuture_v = (w*j)[None,:].T*F_yfuture_p
                    yfuture_v_phaseshifted = torch.fft.irfft(phaseshift_mult*F_yfuture_v.repeat((N_phaseshifts,1,1)),dim=1)*(1/data_dt)

                    F_yfuture_a = (-w*w)[None,:].T*F_yfuture_p
                    yfuture_a_phaseshifted = torch.fft.irfft(phaseshift_mult*F_yfuture_a.repeat((N_phaseshifts,1,1)),dim=1)*(1/data_dt)**2
                    yfuture_a_phaseshifted_cut = yfuture_a_phaseshifted[:,N_samples_cutoff:N_samples-N_samples_cutoff,:]

                    yfuture_p_v_phaseshifted = torch.cat((yfuture_p_phaseshifted,yfuture_v_phaseshifted),dim=2) #this time we calculate the derivatives of the positions ourselves with FFT, using trig. interpolation

                    F_ufuture = torch.fft.rfft(ufuture, dim=1)
                    F_ufuture[:,filter_from:,:] = 0 #filtering
                    ufuture_phaseshifted = torch.fft.irfft(phaseshift_mult*F_ufuture.repeat((N_phaseshifts,1,1)), dim=1)

                    yfuture_p_v_phaseshifted_cut = yfuture_p_v_phaseshifted[:,N_samples_cutoff:N_samples-N_samples_cutoff,:]
                    ufuture_phaseshifted_cut = ufuture_phaseshifted[:,N_samples_cutoff:N_samples-N_samples_cutoff,:]

                #yderivn_v_a = self.derivn(yfuture_p_v_phaseshifted_cut.reshape((-1,12)),ufuture_phaseshifted_cut.reshape((-1,6))).reshape(yfuture_p_v_phaseshifted_cut.shape)
                #torch.cat((ufuture[:,1:,:],ufuture[:,(0,),:]), dim=1).reshape((-1,6))
                #yderivn_v_a = self.derivn(yfuture[:,:,0:12].reshape((-1,12)),torch.cat((ufuture[:,1:,:],ufuture[:,(0,),:]), dim=1).reshape((-1,6))).reshape(yfuture_p_v_phaseshifted_cut.shape)
                #yderivn_v_a = self.derivn(torch.cat((yfuture[:,:,0:6], yfuture_v_phaseshifted), dim=2).reshape((-1,12)),torch.cat((ufuture[:,(-1,),:],ufuture[:,:-1,:]), dim=1).reshape((-1,6))).reshape(yfuture_p_v_phaseshifted_cut.shape)
                #yderivn_v_a = self.derivn(yfuture.reshape((-1,12)),ufuture.reshape((-1,6))).reshape(yfuture_p_v_phaseshifted_cut.shape)

                #yderivn_v_a = self.derivn(torch.cat((yfuture_p_phaseshifted, yfuture_v_phaseshifted), dim=2).reshape((-1,12)),ufuture_phaseshifted_cut.reshape((-1,6))).reshape(yfuture_p_v_phaseshifted_cut.shape)
                #no reshapes: #yderyvn
                if yfuture_p_phaseshifted.shape[0]!=1:
                    assert "initial_std_mean" in Loss_kwargs.keys()
                    yfuture_p_phaseshifted = yfuture_p_phaseshifted.reshape((-1,6))[None,:,:]
                    yfuture_v_phaseshifted = yfuture_v_phaseshifted.reshape((-1,6))[None,:,:]
                    ufuture_phaseshifted_cut = ufuture_phaseshifted_cut.reshape((-1,6))[None,:,:]
                yderivn_v_a = self.derivn(torch.cat((yfuture_p_phaseshifted, yfuture_v_phaseshifted), dim=2)[0,:,:],ufuture_phaseshifted_cut[0,:,:])[None,:,:]
                yderivn_a = yderivn_v_a[:,:,6:12]
                if "initial_std_mean" in Loss_kwargs.keys(): return (None, None, None, None)
                assert yfuture_p_phaseshifted.shape[0]==1, "we have removed the reshape, so batch size must be 1 (unless initial_std_mean, but then we don't get to this point)"
                """
                #plot yderivn_a and yfuture_a_phaseshifted_cut on each other, using plotly on 6 subplots
                fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
                for i in range(6):
                    #fig.add_scatter(y=self.derivn(torch.cat((yfuture_p_phaseshifted, yfuture_v_phaseshifted), dim=2).reshape((-1,12)),ufuture_phaseshifted_cut.reshape((-1,6)))[:,6+i].cpu().detach().numpy(), row=i+1, col=1, name='a*=PHY([x_sim, v_dx], tau_sim)')
                    fig.add_scatter(y=yderivn_a[0,:,i].cpu().detach().numpy(), row=i+1, col=1, name='a*=PHY([x_sim, v_dx], tau_sim)')
                    fig.add_scatter(y=yfuture_a_phaseshifted_cut[0,:,i].cpu().numpy(), row=i+1, col=1, name='a_ddx')
                fig.show()

                #plot yfuture_p_phaseshifted  yfuture_p
                fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
                for i in range(6):
                    fig.add_scatter(y=yfuture_p[0,:,i].cpu().detach().numpy(), row=i+1, col=1, name='yfuture_p')
                    fig.add_scatter(y=yfuture_p_phaseshifted[0,:,i].cpu().numpy(), row=i+1, col=1, name='yfuture_p_phaseshifted')
                fig.show()
                fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
                for i in range(6):
                    fig.add_scatter(y=yfuture_v[0,:,i].cpu().detach().numpy(), row=i+1, col=1, name='yfuture_v')
                    fig.add_scatter(y=yfuture_v_phaseshifted[0,:,i].cpu().numpy(), row=i+1, col=1, name='yfuture_v_phaseshifted')
                fig.show()
                fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
                for i in range(6):
                    fig.add_scatter(y=yfuture_a[0,:,i].cpu().detach().numpy(), row=i+1, col=1, name='yfuture_a')
                    fig.add_scatter(y=yfuture_a_phaseshifted[0,:,i].cpu().numpy(), row=i+1, col=1, name='yfuture_a_phaseshifted')
                fig.show()
                fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
                for i in range(6):
                    fig.add_scatter(y=ufuture[0,:,i].cpu().detach().numpy(), row=i+1, col=1, name='ufuture')
                    fig.add_scatter(y=ufuture_phaseshifted[0,:,i].cpu().numpy(), row=i+1, col=1, name='ufuture_phaseshifted')
                fig.show()

                fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
                for i in range(6):
                    fig.add_scatter(y=(yfuture_a_phaseshifted[0,:,i].cpu().numpy()- yderivn_a[0,:,i].cpu().detach().numpy()), row=i+1, col=1, name='a_ddx - PHY([x_sim, v_dx], tau_sim)')
                fig.show()

                # plot yfuture[:,:,0:6]-tensor_p with ploty on 6 subplots
                fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
                for i in range(6):
                    fig.add_scatter(y=(yfuture[0,:,i]).cpu().numpy(), row=i+1, col=1, name='yfuture')
                    fig.add_scatter(y=(tensor_p[:,i]).cpu().numpy(), row=i+1, col=1, name='tensor_p')
                    fig.add_scatter(y=data_fit_input_training.y[:,i], row=i+1, col=1, name='data_fit_input_training.y')
                    fig.add_scatter(y=(yfuture[0,:,i]-tensor_p[:,i]).cpu().numpy(), row=i+1, col=1, name='yfuture-tensor_p')
                    fig.add_scatter(y=(yfuture[0,:,i]).cpu().numpy()-data_fit_input_training.y[:,i], row=i+1, col=1, name='yfuture-tensor_p')
                fig.show()
                """
                diff_final = yfuture_a_phaseshifted_cut - yderivn_a
                if fd_mode:
                    pass
                    F_yderivn_a = torch.fft.rfft(yderivn_a,dim=1) #we don't filter this, because this is the simulation output
                    ##it would possibly be worth to weigh all these frequencies with -1/w**2, because even if it's in FD, the fact that we are fitting acceleration to acceleration
                    ##means that the higher frequencies are more emphasized
                    #F_yderivn_p_from1 = (-(1/(w[1:]*w[1:]))[None,:].T*F_yderivn_a[:,1:,:])
                    #diff_fd = F_yderivn_p_from1 - F_yfuture_p[:,1:,:]
                    diff_fd = F_yfuture_a - F_yderivn_a
                    #raise Exception('diff_final shall not work here')
                    with open('diff_fd.txt', 'r') as f:
                        index1 = int(f.readline())
                        index2 = int(f.readline())
                    diff_final_fd_0 = torch.real(diff_fd/N_samples)**2+torch.imag(diff_fd/N_samples)**2
                    diff_final_fd = torch.zeros_like(diff_final_fd_0)
                    diff_final_fd[:,index1:index2,:] = diff_final_fd_0[:,index1:index2,:] 
                else: diff_final_fd = None
                return (diff_final, yderivn_a, yfuture_a_phaseshifted_cut, diff_final_fd)

            #plot F_yfuture_a, F_yderivn_a, diff_fd on dB scale with plotly on 6 subplots:
            fig = make_subplots(rows=6, cols=1)
            for i in range(6):
                fig.add_scatter(y=20*torch.log10(torch.abs(F_yfuture_a[0,:,0+i])), row=i+1, col=1, name='F_yfuture_a')
                fig.add_scatter(y=20*torch.log10(torch.abs(F_yderivn_a[0,:,0+i])), row=i+1, col=1, name='F_yderivn_a')
                fig.add_scatter(y=20*torch.log10(torch.abs(diff_fd[0,:,0+i])), row=i+1, col=1, name='diff_fd')
            fig.show()

            #plot diff_final with plotly on 6 subplots:
            fig = make_subplots(rows=6, cols=1)
            for i in range(6):
                fig.add_scatter(y=diff_final[0,:,0+i].detach(), row=i+1, col=1)
            fig.show()

            #plot diff_final using plotly on 6 subplots:
            fig = make_subplots(rows=6, cols=1)
            for i in range(6):
                fig.add_scatter(y=diff_final[0,:,0+i].detach(), row=i+1, col=1)
            fig.show()
            
            fig = make_subplots(rows=6, cols=1)
            for i in range(6):
                fig.add_scatter(y=(diff_final[0,:,0+i]*(self.derivn.a_std[i])).detach(), row=i+1, col=1)
                fig.add_scatter(y=(yfuture_a_phaseshifted_cut[0,:,0+i]).detach(), row=i+1, col=1)
            fig.show()

        if "initial_std_mean" in Loss_kwargs.keys(): 
            get_the_diff() #key here: we don't disable_nn so that the initial_std_mean is activated
            return 0 

        diff_final_onlyrobot = None
        if self.derivn.robustify_nn and not self.derivn.only_ann:
            self.derivn.disable_nn = True
            with torch.no_grad():
                diff_final, graph_yderivn_no_ann, _, _ = get_the_diff()
                self.derivn.res_mean = torch.mean((diff_final).reshape((-1,6)),dim=0).detach()
                self.derivn.res_std = torch.std((diff_final).reshape((-1,6)),dim=0).detach()
                print(f'loss :: Calculated normalization for derivn:\nmean = {self.derivn.res_mean}\nstd = {self.derivn.res_std}\n...and now we recalculate the loss:')
                sys.stdout.flush()
                diff_final_onlyrobot = diff_final
                del diff_final
            self.derivn.disable_nn = False

        if not n_step_nrms_mode:
            diff_final, graph_yderivn_ann, graph_a_ddx, diff_final_fd = get_the_diff()
            if not self.derivn.robustify_nn: diff_final_onlyrobot = diff_final
            diff_center = diff_final[:, stiction_pad:(15000-stiction_pad), :] if paper_case == 3 else diff_final #BEWARE: this is a hack to remove the first and last stiction_pad samples
            loss_value_diff = torch.mean(  (torch.mean(((diff_center**2 if not fd_mode else diff_final_fd)),dim=1)[0,:]*(1/self.derivn.a_std**2))   )
        else:
            y=yfuture[0,:,:]
            u=ufuture[0,:,:]
            #n_timeswecando = y.shape[0]-n_steps-1 #the amount of times the window of n_steps fits in the length of the data [[XXXX].......] 
            sliding_window = torch.arange(0, y.shape[0]).unfold(0,n_steps+1,1)
            sliding_window = sliding_window[torch.randperm(sliding_window.shape[0])[:n_batch],:] #TODO check
            y_window = y[sliding_window, :] #[n_timeswecando x n_steps+1 x n_states]
            u_window = u[sliding_window, :]
            y_current = y_window[:,0,:]
            y_stack = []
            y_stack.append(y_current)
            for i in range(n_steps):
                if (i%50==0): sys.stdout.write('▁')
                y_current = self.fn(y_current, u_window[:,i,:])
                y_stack.append(y_current)
            y_stack = torch.stack(y_stack, dim=1)
            
            r = (y_stack[:,:,0:6]-y_window[:,:,0:6])**2
            r = r / y.std(dim=0).pow(2)[None,None,0:6].repeat(y_stack.shape[0],y_stack.shape[1],1) #mode: weigh_by_y_var
            loss_value_diff = torch.mean(r.reshape((-1,))) #mode: one_number
            if force_cartoon: 
                diff_final, graph_yderivn_ann, graph_a_ddx, diff_final_fd = get_the_diff()
                if not self.derivn.robustify_nn: diff_final_onlyrobot = diff_final

        param_groups = Loss_kwargs['param_groups']
        l2_norm = sum(p.pow(2).sum() for p in param_groups[0]['params'])
        reg_val = (0 if not from_exp_dir else 0)*0.00001*l2_norm

        loss_value = loss_value_diff + reg_val

        self.derivn.inside_loss = False
        #s.t. k_nlspring3 <= 5000
        #constraint_val = self.log_barrier(self.derivn.param_scale['k3_nlspring']*param_groups[1]['params'][4]-5000, torch.tensor(1000))

        #s.t. x<=20
        #s.t. -x<=20 ~ x>=-20
        #friction_params, dyn_params, hwc_params, u_params
        #constraint_val = 0 + \
        #    self.log_barrier_multi(param_groups[0]['params'][0]-20,   torch.tensor(1000)) + \
        #    self.log_barrier_multi(-param_groups[0]['params'][0]-20,  torch.tensor(1000)) + \
        #    self.log_barrier_multi(param_groups[0]['params'][1]-20,   torch.tensor(1000)) + \
        #    self.log_barrier_multi(-param_groups[0]['params'][1]-20,  torch.tensor(1000)) + \
        #    self.log_barrier_multi(param_groups[0]['params'][2]-20,   torch.tensor(1000)) + \
        #    self.log_barrier_multi(-param_groups[0]['params'][2]-20,  torch.tensor(1000)) + \
        #    self.log_barrier_multi(param_groups[0]['params'][3]-200,  torch.tensor(1000)) + \
        #    self.log_barrier_multi(-param_groups[0]['params'][3]-200, torch.tensor(1000))
        # LOOKOUT in-place op in log_barrier_multi!

        #return torch.mean((diff_final**2).reshape(-1))

        #our cartoon
        if epoch%500==0:
            #load netout.torch_save:
            try:
                net_out_scaled = torch.load('net_out.torch_save')
            except:
                net_out_scaled = None
            #plot diff_final using plotly on 6 subplots:
            fig = make_subplots(rows=6, cols=2)
            for i in range(6):
                if diff_final_onlyrobot is not None: fig.add_scatter(y=diff_final_onlyrobot[0,:,0+i].detach(), row=i+1, col=1, name="diff_final", line=dict(color="#636efa"))
                #fig.add_scatter(y=net_out_scaled[:,0+i].flip(dims=(0,)).roll(000, dims=(0,)).detach(), row=i+1, col=1, name="net_out_scaled")
                if self.derivn.robustify_nn and net_out_scaled is not None: fig.add_scatter(y=net_out_scaled[:,0+i].detach(), row=i+1, col=1, name="net_out_scaled", line=dict(color="#ef553b"))
                #fig.update_yaxes(range=[[-1,2],[-4,3],[-10,-3],[-7,5],[-10,25],[-18,10]][i], row=i+1, col=1) #commented this out for fd_mode
            for i in range(6):
                if self.derivn.robustify_nn: 
                    if not self.derivn.only_ann: fig.add_scatter(y=graph_yderivn_no_ann[0,:,0+i].detach(), row=i+1, col=2, name="yderivn_no_ann", line=dict(color="#109618"))
                    fig.add_scatter(y=graph_yderivn_ann[0,:,0+i].detach(), row=i+1, col=2, name="yderivn_ann", line=dict(color="#1f77b4"))
                else:
                    fig.add_scatter(y=graph_yderivn_ann[0,:,0+i].detach(), row=i+1, col=2, name="yderivn", line=dict(color="#1f77b4"))
                fig.add_scatter(y=graph_a_ddx[0,:,0+i].detach(), row=i+1, col=2, name="a_ddx", line=dict(color="#ff7f0e"))
                #fig.update_yaxes(range=[[-1,2],[-4,3],[-10,-3],[-7,5],[-10,25],[-18,10]][i], row=i+1, col=1)
            fig.update_layout(title_text="epoch "+str(epoch)+", loss = "+str(loss_value.detach().numpy())+", reg_val = "+str(reg_val.detach().numpy()))
            #convert epoch to add leading zeros:
            epoch_str = str(epoch)
            while len(epoch_str)<7: epoch_str = '0'+epoch_str
            fig.write_html("epoch_"+str(epoch_str)+".html")
            #fig.show()
        return loss_value




    def loss_simulation(self, uhist, yhist, ufuture, yfuture, **Loss_kwargs):
        print('loss :: size of this batch:', ufuture.shape[0])
        flush = lambda:sys.stdout.flush()
        x = self.encoder(uhist, yhist) #this fails if dt starts to change
        #in case of only_velocities = True or E_ypast, this additional step shall make it correct: #TUNEIT #LOOKOUT
        #x = self.fn(x,uhist[:,-1,:]) #verified that this is the correct element to take from uhist #actually it was incorrect because na_right=1

        epoch = Loss_kwargs['epoch']
        bestfit = Loss_kwargs['bestfit']
        param_groups = Loss_kwargs['param_groups']
        diff = []
        if self.obj_weights is not None: assert self.obj_weights.shape[0] == yfuture.shape[2] and len(self.obj_weights.shape) == 1
        xfuture = []
        fd_mode = False
        if not fd_mode:
            nf_internal=min(max((epoch-100)//10,2),300)
            print('nf_internal: ', nf_internal)
        else:
            nf_internal = None

        #maxnf = yfuture.shape[1]
        #from math import sqrt
        #nf_internal=min(maxnf,max(2,maxnf-int(sqrt(max(maxnf**2-epoch,0)))))

        #l2_norm = sum(p.pow(2).sum() for p in param_groups[0]['params'])
        #reg_val = 0.00001*l2_norm
        reg_val = 0 #TUNEIT #switch regularization on/off
        #debug params: [torch.detach(p).numpy() for p in param_groups[1]['params']]

        sqr_cut_off = 500 #TUNEIT

        print('loss :: simulating...'); flush()
        if fd_mode: yhats = torch.zeros(yfuture.shape, device=tgt_device)
        for i,(u,y) in enumerate(zip(torch.transpose(ufuture,0,1), torch.transpose(yfuture,0,1))): #iterate over time
            #if (i%50==0):
            #    with open('derivn-'+str(i)+'.txt','w') as f:
            #        f.write(str(self.derivn.graph_for((x,u))))
            if (i%50==0):
                sys.stdout.write('▃'); flush()
            yhat = x #self.hn(x) if not self.feedthrough else self.hn(x,u) #for the robot, it's just not needed
            if fd_mode:
                yhats[:,i,:] = yhat
            else:
                dy = (yhat - y)**2 # (Nbatch, ny)
                if self.obj_weights is not None:
                    dy = dy*self.obj_weights
                diff.append(dy)
            if not fd_mode:
                with torch.no_grad(): #break if the error is too large
                    if torch.max(yhat).item()>sqr_cut_off:
                        print('WARNING: we hit the sqr_cut_off')
                        #LOOKOUT with possibly truncating the elements in the mean: it will change the weighing of the diff wrt. lambda_encoder_in_obj
                        break
            if not fd_mode and i>nf_internal-1: break
            x = self.fn(x,u)
            xfuture.append(x)

        if fd_mode:
            print('loss :: fft'); flush()
            rfft_yhats = torch.fft.rfft(yhats,dim=1)
            with torch.no_grad(): rfft_yfuture = torch.fft.rfft(yfuture,dim=1) 
            rfft_yhats_focus = rfft_yhats[:,0:10,:] #TUNEIT frequency range of interest is now 1:10 
            rfft_yfuture_focus = rfft_yfuture[:,0:10,:] 
            #plt.clf(); plt.plot(20*torch.log10(abs(rfft_yhats[0,0:20,0])).cpu().detach().numpy()); mkpng('rfft_yhats')
            #plt.clf(); plt.plot(20*torch.log10(abs(rfft_yfuture[0,0:20,0])).cpu().detach().numpy()); mkpng('rfft_yfuture')
            #plt.clf(); plt.plot(20*torch.log10(torch.abs(torch.fft.rfft(ufuture[0,:,0])[0:40])).cpu().detach().numpy()); mkpng('rfft_ufuture')

            #diff = torch.real(torch.transpose(diff_inner,1,2) @ diff_inner)

            # ways to compose the objective:
            #1) with abs() 
            #diff = torch.abs(rfft_yhats_focus-rfft_yfuture_focus)**2 

            #2) with A^T*A 
            #diff_inner = rfft_yhats_focus-rfft_yfuture_focus
            #diff = torch.real(torch.transpose(diff_inner,1,2) @ diff_inner) 
            #assert diff.shape[1] == 1 and diff.shape[2] == 1

            #3) with I^2+R^2 
            diff = (torch.real(rfft_yhats_focus)-torch.real(rfft_yfuture_focus))**2 + (torch.imag(rfft_yhats_focus)-torch.imag(rfft_yfuture_focus))**2 
            #diff = diff * (torch.randn_like(diff)>0)
            #h_diff_hook = diff.register_hook(lambda grad: printgrad(grad,0)) #GRADDEBUG
            #this was to downweight frequencies at the beginning, but it's incorrect as the FFT has two sides:
            #diffview = diff.view(-1,i)
            #diffview *= torch.linspace(min(epoch/500,1),1,i)
        else:
            diff = torch.stack(diff,dim=1)

        xfuture = torch.stack(xfuture, dim=1)
        print('loss :: encoder_in_obj'), flush()
        print('loss :: debug, yfuture.shape[1]: '+str(yfuture.shape[1])+' xfuture.shape[1]: '+str(xfuture.shape[1]))
        if self.lambda_encoder_in_obj is not None:
            #assert uhist.shape[1]==yhist.shape[1]
            #nfuture = min(nf_internal,yfuture.shape[1])
            nfuture = xfuture.shape[1] #this used to be yfuture.shape[1] but then it crashes if sqr_cut_off becomes active
            #uall = torch.cat((uhist,ufuture),dim=1)
            #yall = torch.cat((yhist,yfuture),dim=1)
            yall = yfuture
            uall = ufuture
            #nmax = max(self.na+self.na_right, self.nb+self.nb_right)
            obj_enc_x = []
            obj_enc_hn = []
            obj_enc_x_exp_scaling = 1
            for i in range(nfuture - self.na - self.na_right):
                #assert self.na+self.na_right == self.nb and self.nb_right == 0 and uhist.ndim == 2 #previously I just applied upast to the corresponding encoder argument (which was basically []).
                encoder_output = self.encoder(uall[:,(i+self.nb_right):(i+self.nb+self.nb_right*2)],yall[:,(i+self.na_right):(i+self.na+self.na_right*2),:])
                obj_enc_hn.append((self.hn(encoder_output)-yfuture[:,i+self.na])**2)
                obj_enc_x.append(obj_enc_x_exp_scaling*((encoder_output-xfuture[:,i+self.na])**2))
                obj_enc_x_exp_scaling *= 0.9 #CASE #TUNEIT the exponential scaling of obj_enc_x
                if obj_enc_x_exp_scaling < 1e-5: break #CASE
            #return torch.mean((torch.cat((torch.stack(diff,dim=1),torch.stack(obj_enc,dim=1)),dim=1)))
            #adaptive_weight_obj_enc = 1-max(min((epoch-100)/300,1),0)

            #TUNEIT adaptive weights
            adaptive_weight_obj_enc_hn = 1 #CASE
            #adaptive_weight_obj_enc_x = min(max((epoch)/1000,0),1)*0.3 #CASE
            adaptive_weight_obj_enc_x = 1 #CASE
            print('adaptive_weight_obj_enc_x:', adaptive_weight_obj_enc_x) #UNDO
            adaptive_weight_sim = 1
            #adaptive_weight_obj_enc_hn = 1 if epoch%200<100 else 0
            #adaptive_weight_obj_enc_x = 0
            #adaptive_weight_sim = 1-adaptive_weight_obj_enc_hn
            #if epoch<=30:
            #    adaptive_weight_obj_enc = 1
            #else:
            #    if epoch%10>7:
            #        adaptive_weight_obj_enc = 1
            #    else:
            #        adaptive_weight_obj_enc = 0
            #adaptive_weight_obj_enc= 1-max(min((epoch-100)/300,1),0)
            #print("adaptive_weight_obj_enc_hn",adaptive_weight_obj_enc_hn)

            #    adaptive_weight_obj_enc_hn*self.lambda_encoder_in_obj['hn']*torch.stack(obj_enc_hn,dim=1).reshape(-1),\
            if len(obj_enc_x)==0: obj_enc_x.append(torch.tensor([]))
            #print('adaptive_weight_obj_enc_x =', adaptive_weight_obj_enc_x, 'nf_internal =', nf_internal)
            print('loss :: we return in the next step'), flush()
            return torch.mean((torch.cat((\
                adaptive_weight_sim*diff.reshape(-1),\
                adaptive_weight_obj_enc_x*self.lambda_encoder_in_obj['x']*torch.stack(obj_enc_x,dim=1).reshape(-1).to(tgt_device),\
                )))) + reg_val
            #We needed to .to(tgt_device) there because sometimes the obj_enc_x is just an empty tensor, which will be on cpu if stacked.
        else:
            return torch.mean(diff.reshape(-1))

def mkpng(what, title=None):
    if title is not None:
        if title == True: title = what
        plt.title(title)
    plt.savefig(what+".png",dpi=(250), bbox_inches='tight')
    plt.close()
    print('mkpng: '+what)

def mkplt(x, which_file='temprect'): plt.clf(); plt.plot(x.cpu().detach()); mkpng(which_file)

## Helper functions & parameters
mat2onedarray = lambda x: x.reshape((x.shape[0]))
onedarray2mat = lambda x: x.reshape(x.shape[0],1)
measurable_states = np.array(range(12)) #TUNEIT

## Settings
generate_initial_fit_figures = False if not from_exp_dir else (True if really_from_exp_dir else True) #DEBUGME #IMPORTANT
show_input_on_start = False
fit_to_simulated_data = True if not from_exp_dir else not exp_real_data #IMPORTANT
filter_from = 200
mon_steps = [75, 150]

def compare(data_measurement, data_simulation, title='', plt_show_block=True, mkpng_filename=None):
        assert data_simulation.y.shape[1] == data_measurement.y.shape[1]
        num_axes = data_simulation.y.shape[1]
        fig, axs = plt.subplots(num_axes)
        fig.suptitle(title)
        nrms_per_ch = data_simulation.NRMS_per_channel(data_measurement, multi_average=False)
        for i in range(data_simulation.y.shape[1]):
            ax = num_axes == 1 and axs or axs[i]
            ax.plot(data_measurement.y[:,i])
            ax.plot(data_simulation.y[:,i])
            ax.legend(['measured', 'simulation (NRMS =  '+str(nrms_per_ch[i])+')'])
        if mkpng_filename:
            mkpng(mkpng_filename)
        elif plt_show_block:
            plt.show(block=plt_show_block)
        return nrms_per_ch

def show_periods_on_each_other(signal, n_samples_per_period, first_period=0):
    #function to plot periods of the signal on each other
    plt.clf()
    n_periods = int(len(signal)/n_samples_per_period)
    for i in range(first_period, n_periods):
        plt.plot(signal[i*n_samples_per_period:(i+1)*n_samples_per_period])
    plt.show(block=True)

demean=lambda x:x-np.mean(x)

def myfftresamp(x_H,to_n,fd_input=False, transpose_outputs = False, norm = "backward"):
    #for something similar, see: % https://www.dsprelated.com/showcode/54.php
    #assert x_H.ndim==2
    if x_H.ndim==1: x_H = np.array([x_H])
    xf_H=np.fft.fft(x_H,axis=1,norm=norm) if not fd_input else x_H
    to_np2=to_n//2
    assert to_n//2==to_n/2
    xfr_H=np.concatenate((xf_H[:,0:to_np2], xf_H[:,-to_np2:]), axis=1)
    td_output = np.real(np.fft.ifft(xfr_H,axis=1,norm=norm))*(to_n/x_H.shape[1])
    fd_output = xfr_H
    return (td_output, fd_output) if not transpose_outputs else (td_output.T, fd_output.T)
    #figure(1), clf, plot(y)
    #figure(2), clf, plot(mag2db(abs(fft(y))),'.-')

#print(myfftresamp(np.array([1,2,3,4,5,6,7,8,9,10,11,13,3,6,9,4,2,1]),4))
#In MATLAB it gives: [ 6.9834 24.3804 44.8676 27.7687].'

class System_data_FD(System_data): #this is used nowhere

    def __init__(self, fft_the_yfuture=False, **args):
        super(System_data_FD, self).__init__(**args)
        self.fft_the_yfuture = fft_the_yfuture

    def to_hist_future_data(self, na, nb, nf, na_right=0, nb_right=0, stride=1, force_multi_u=False, force_multi_y=False, online_construct=False):
        '''Transforms the system data to encoder structure as structure (uhist,yhist,ufuture,yfuture) of

        Made for simulation error and multi step error methods

        Parameters
        ----------
        na : int
            y history considered
        nb : int
            u history considered
        nf : int
            future inputs considered

        Returns
        -------
        uhist : ndarray (samples, nb, nu) or (sample, nb) if nu=None
            array of [u[k-nb],....,u[k - (nb_right + 1)]]
        yhist : ndarray (samples, na, ny) or (sample, na) if ny=None
            array of [y[k-na],....,y[k - (na_right + 1)]]
        ufuture : ndarray (samples, nf, nu) or (sample, nf) if nu=None
            array of [u[k],....,u[k+nf-1]]
        yfuture : ndarray (samples, nf, ny) or (sample, nf) if ny=None
            array of [y[k],....,y[k+nf-1]]
        '''
        assert online_construct == False #we don't support this because we don't know what it is
        #assert nf == self.u.shape[0] #we go round in a sample buffer that's the same size as nf
        assert nf <= self.u.shape[0]
        assert stride == 1
        u = np.tile(np.copy(self.u),(3,1))
        y = np.tile(np.copy(self.y),(3,1))
        #we need the repeat because let's say our u is of 600 elements, then we want to be able to take 500:700 as well
        yhist = []
        uhist = []
        ufuture = []
        yfuture = []
        for i in range(nf):
            yhist.append(y[nf+i-na:nf+i+na_right])
            uhist.append(u[nf+i-nb:nf+i+nb_right])
            ufuture.append(u[nf+i:nf*2+i])
            yfuture.append(y[nf+i:nf*2+i])
        uhist, yhist, ufuture, yfuture = np.array(uhist), np.array(yhist), np.array(ufuture), np.array(yfuture)
        if self.fft_the_yfuture:
            yfuture = torch.fft.rfft(yfuture, dim=1)
        if force_multi_u and uhist.ndim==2: #(N, time_seq, nu) #LOOKOUT not sure what this is, might cause problems
            assert False
            uhist = uhist[:,:,None]
            ufuture = ufuture[:,:,None]
        if force_multi_y and yhist.ndim==2: #(N, time_seq, ny)
            assert False
            yhist = yhist[:,:,None]
            yfuture = yfuture[:,:,None]
        return uhist, yhist, ufuture, yfuture

def diff_v_from_p_and_filter_yu(input_system_data): #this is used when loading the data from the matlab files indeed (like py_recording_*)
    with torch.no_grad():
        input_y = torch.tensor(input_system_data.y)
        input_u = torch.tensor(input_system_data.u)
        N = input_y.shape[0]
        w = ((((torch.arange(0,(N//2+1)))/N)*(2*torch.pi)))
        j = torch.tensor(complex(0,1))
        F_yfuture_p = torch.fft.rfft(input_y[:,0:6], dim=0)
        F_yfuture_v = (w*j)[None,:].T*F_yfuture_p
        F_yfuture_p[filter_from:,:] = 0
        F_yfuture_v[filter_from:,:] = 0
        #yfuture_p = input_y[:,0:6]
        yfuture_p = torch.fft.irfft(F_yfuture_p, dim=0)
        yfuture_v = torch.fft.irfft(F_yfuture_v,dim=0)*(1/data_dt)
        yfuture_p_v = torch.cat((yfuture_p,yfuture_v),dim=1)
        F_ufuture = torch.fft.rfft(input_u[:,:], dim=0)
        F_ufuture[filter_from:,:] = 0
        ufuture = torch.fft.irfft(F_ufuture, dim=0)
        output_system_data = System_data(u=ufuture.detach().numpy(), y=yfuture_p_v.detach().numpy(), dt=input_system_data.dt)
        return output_system_data

def diff_v_from_p(input_system_data): #this is used when loading the data from the matlab files indeed (like py_recording_*)
    with torch.no_grad():
        input_y = torch.tensor(input_system_data.y)
        N = input_y.shape[0]
        w = ((((torch.arange(0,(N//2+1)))/N)*(2*torch.pi)))
        j = torch.tensor(complex(0,1))
        F_yfuture_p = torch.fft.rfft(input_y[:,0:6], dim=0)
        F_yfuture_v = (w*j)[None,:].T*F_yfuture_p
        yfuture_p = input_y[:,0:6]
        yfuture_v = torch.fft.irfft(F_yfuture_v,dim=0)*(1/data_dt)
        yfuture_p_v = torch.cat((yfuture_p,yfuture_v),dim=1)
        output_system_data = System_data(u=input_system_data.u, y=yfuture_p_v.detach().numpy(), dt=input_system_data.dt)
        return output_system_data

#            _           _   the robot data
#           | |         | |
#  _ __ ___ | |__   ___ | |_
# | '__/ _ \| '_ \ / _ \| __|
# | | | (_) | |_) | (_) | |_
# |_|  \___/|_.__/ \___/ \__|
#

data_dt = 0.004 #it should be the same as in `diff(z.time)`

## FD fitting: load data
#robot_mat['realizations'][0][0]['periods'][0][0][0][0]['y_data_V'][0][0]
#plt.plot(20*np.log10(np.real(np.fft.ifft(robot_mat['realizations'][0][0]['y_data_f_avg_V'][0][0]))))
#plt.plot(20*np.log10(robot_mat['realizations'][0][0]['y_data_f_avg_V'][0][0]))
#plt.plot(20*np.log10(np.real(robot_mat['realizations'][0][0]['y_data_f_avg_V'][0][0])))
#plt.plot(np.fft.ifft(robot_mat['realizations'][0][0]['y_data_f_avg_V'][0][0], axis=0))
#plt.clf(), plt.plot(myfftresamp(np.fft.ifft(robot_mat['realizations'][0][0]['y_data_f_avg_V'][0][0], axis=0).T,600*2)[0].T)
#plt.clf(), plt.plot(myfftresamp(robot_mat['realizations'][0][0]['y_data_f_avg_V'][0][0].T,600*2,fd_input=True)[0].T)

#load mat files
robot_mat_29 = None
ref_trajectories = None
robot_mat_45 = loadmat('../common_data/py_recording_2021_12_15_20H_45M.mat')

#use just the original data (first period), without resampling:
u_training1_td   = robot_mat_45['realizations'][0][0]['periods'][0][0][0][0]['u_data_V'][0][0].astype(np.float32)
u_training2_td   = robot_mat_45['realizations'][0][1]['periods'][0][0][0][0]['u_data_V'][0][0].astype(np.float32)
u_training3_td   = robot_mat_45['realizations'][0][2]['periods'][0][0][0][0]['u_data_V'][0][0].astype(np.float32)
u_validation_td  = robot_mat_45['realizations'][0][1]['periods'][0][0][0][0]['u_data_V'][0][0].astype(np.float32)
u_test_td        = robot_mat_45['realizations'][0][2]['periods'][0][0][0][0]['u_data_V'][0][0].astype(np.float32)

y_training1_td   = (math.pi/180)*robot_mat_45['realizations'][0][0]['periods'][0][0][0][0]['y_data_V'][0][0].astype(np.float32)
y_training2_td   = (math.pi/180)*robot_mat_45['realizations'][0][1]['periods'][0][0][0][0]['y_data_V'][0][0].astype(np.float32)
y_training3_td   = (math.pi/180)*robot_mat_45['realizations'][0][2]['periods'][0][0][0][0]['y_data_V'][0][0].astype(np.float32)
y_validation_td  = (math.pi/180)*robot_mat_45['realizations'][0][1]['periods'][0][0][0][0]['y_data_V'][0][0].astype(np.float32)
y_test_td        = (math.pi/180)*robot_mat_45['realizations'][0][2]['periods'][0][0][0][0]['y_data_V'][0][0].astype(np.float32)

#use first period of data:
#u_training1_td, u_training1_fd   = myfftresamp(robot_mat_29['realizations'][0][0]['periods'][0][0][0][0]['u_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#u_training2_td, u_training2_fd   = myfftresamp(robot_mat_45['realizations'][0][1]['periods'][0][0][0][0]['u_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#u_training3_td, u_training3_fd   = myfftresamp(robot_mat_45['realizations'][0][2]['periods'][0][0][0][0]['u_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#u_validation_td, u_validation_fd = myfftresamp(robot_mat_29['realizations'][0][1]['periods'][0][0][0][0]['u_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#u_test_td, u_test_fd             = myfftresamp(robot_mat_29['realizations'][0][2]['periods'][0][0][0][0]['u_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)

#y_training1_td, y_training1_fd   = myfftresamp((math.pi/180)*robot_mat_29['realizations'][0][0]['periods'][0][0][0][0]['y_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#y_training2_td, y_training2_fd   = myfftresamp((math.pi/180)*robot_mat_45['realizations'][0][1]['periods'][0][0][0][0]['y_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#y_training3_td, y_training3_fd   = myfftresamp((math.pi/180)*robot_mat_45['realizations'][0][2]['periods'][0][0][0][0]['y_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#y_validation_td, y_validation_fd = myfftresamp((math.pi/180)*robot_mat_29['realizations'][0][1]['periods'][0][0][0][0]['y_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)
#y_test_td, y_test_fd             = myfftresamp((math.pi/180)*robot_mat_29['realizations'][0][2]['periods'][0][0][0][0]['y_data_V'][0][0].T,600, fd_input=False, transpose_outputs=True)

#use FD average of all periods of data:
#u_training1_td, u_training1_fd     = myfftresamp(robot_mat_29['realizations'][0][0]['u_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#u_training2_td, u_training2_fd     = myfftresamp(robot_mat_45['realizations'][0][1]['u_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#u_training3_td, u_training3_fd     = myfftresamp(robot_mat_45['realizations'][0][2]['u_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#u_validation_td, u_validation_fd   = myfftresamp(robot_mat_29['realizations'][0][1]['u_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#u_test_td, u_test_fd               = myfftresamp(robot_mat_29['realizations'][0][2]['u_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)

#y_training1_td, y_training1_fd   = myfftresamp((math.pi/180)*robot_mat_29['realizations'][0][0]['y_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#y_training2_td, y_training2_fd   = myfftresamp((math.pi/180)*robot_mat_45['realizations'][0][1]['y_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#y_training3_td, y_training3_fd   = myfftresamp((math.pi/180)*robot_mat_45['realizations'][0][2]['y_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#y_validation_td, y_validation_fd = myfftresamp((math.pi/180)*robot_mat_29['realizations'][0][1]['y_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)
#y_test_td, y_test_fd             = myfftresamp((math.pi/180)*robot_mat_29['realizations'][0][2]['y_data_f_avg_V'][0][0].T,600, fd_input=True, transpose_outputs=True)

data_fit_input_training   = System_data(u=u_training1_td, y=y_training1_td, dt=data_dt)
data_fit_input_training2  = System_data(u=u_training2_td, y=y_training2_td, dt=data_dt)
data_fit_input_training3  = System_data(u=u_training3_td, y=y_training3_td, dt=data_dt)
data_fit_input_validation = System_data(u=u_validation_td, y=y_validation_td, dt=data_dt)
data_fit_input_test       = System_data(u=u_test_td, y=y_test_td, dt=data_dt)

#plot data_fit_input_training.y on 6 subplots with plotly:
fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
for i in range(6):
    fig.add_trace(go.Scatter(y=data_fit_input_training.y[:,i], name='pos #'+str(i)), row=i+1, col=1)
fig.write_html('data_fit_input_training_y_unfiltered.html')

#calculate the unfiltered v from p and plot it:
data_fit_input_training_d_unfiltered = diff_v_from_p(data_fit_input_training)
fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
for i in range(6):
    fig.add_trace(go.Scatter(y=data_fit_input_training_d_unfiltered.y[:,i+6], name='vel #'+str(i)), row=i+1, col=1)
fig.write_html('data_fit_input_training_v_unfiltered.html')    

#plot data_fit_input_training.u on 6 subplots with plotly:
fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
for i in range(6):
    fig.add_trace(go.Scatter(y=data_fit_input_training.u[:,i], name='tau #'+str(i)), row=i+1, col=1)
fig.write_html('data_fit_input_training_u_unfiltered.html')    

import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Calculate the unfiltered v from p and plot it:
data_fit_input_training_d_unfiltered = diff_v_from_p(data_fit_input_training)

# Create a subplot with 6 rows, shared x-axes, and two y-axes (one for v and one for tau)
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, specs=[[{'secondary_y': True},]] * 1)

which = [5,]
for i in which: fig.add_trace(go.Scatter(y=data_fit_input_training_d_unfiltered.y[:, i ], name='pos unfilt #' + str(i), legendgroup='p', showlegend=(i == 0)), row=1, col=1, secondary_y=True)
for i in which: fig.add_trace(go.Scatter(y=data_fit_input_training_d_unfiltered.y[:, i + 6], name='vel unfilt #' + str(i), legendgroup='v', showlegend=(i == 0)), row=1, col=1, secondary_y=True)
for i in which: fig.add_trace(go.Scatter(y=data_fit_input_training.u[:, i], name='tau unfilt #' + str(i), legendgroup='tau', showlegend=(i == 0)), row=1, col=1, secondary_y=False)

N_samples = 15000 #LOOKOUT this is defined at 4 places, 3x here and also in fit_system.py in deepSI
N_samples_cutoff = 0 #LOOKOUT this is defined at 4 places, 3x here and also in fit_system.py in deepSI
#data_fit_input_training   = diff_v_from_p(data_fit_input_training) #note: v will be garbage because we didn't filter it
#data_fit_input_validation = diff_v_from_p(data_fit_input_validation)
#data_fit_input_test       = diff_v_from_p(data_fit_input_test)
data_fit_input_training   = diff_v_from_p_and_filter_yu(data_fit_input_training) #note: v will be garbage because we didn't filter it
data_fit_input_validation = diff_v_from_p_and_filter_yu(data_fit_input_validation)
data_fit_input_test       = diff_v_from_p_and_filter_yu(data_fit_input_test)

for i in which: fig.add_trace(go.Scatter(y=data_fit_input_training.y[:, i ], name='pos filt #' + str(i), legendgroup='p', showlegend=(i == 0)), row=1, col=1, secondary_y=True)
for i in which: fig.add_trace(go.Scatter(y=data_fit_input_training.y[:, i + 6], name='vel filt #' + str(i), legendgroup='v', showlegend=(i == 0)), row=1, col=1, secondary_y=True)
for i in which: fig.add_trace(go.Scatter(y=data_fit_input_training.u[:, i], name='tau filt #' + str(i), legendgroup='tau', showlegend=(i == 0)), row=1, col=1, secondary_y=False)

# Save the combined plot as an HTML file
fig.write_html('combined_data_fit_input_training_v_and_tau_unfiltered.html')




#save u_training1_td, y_training1_td, u_training2_td, y_training2_td, u_training3_td, y_training3_td, u_validation_td,  y_validation_td, u_test_td, y_test_td, data_dt into a .mat file with savemat:
#savemat('weigand_robot_data_export.mat', dict(
#    u_training1_td = u_training1_td,
#    y_training1_td = y_training1_td,
#    u_training2_td = u_training2_td,
#    y_training2_td = y_training2_td,
#    u_training3_td = u_training3_td,
#    y_training3_td = y_training3_td,
#    u_validation_td = u_validation_td,
#    y_validation_td = y_validation_td,
#    u_test_td = u_test_td,
#    y_test_td = y_test_td,
#    data_dt = data_dt))
#print('export done')
#sys.exit(0)



#  _       _ _   _       _        _                 _       _   _
# (_)     (_) | (_)     | |      (_)               | |     | | (_)
#  _ _ __  _| |_ _  __ _| |   ___ _ _ __ ___  _   _| | __ _| |_ _  ___  _ __
# | | '_ \| | __| |/ _` | |  / __| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \
# | | | | | | |_| | (_| | |  \__ \ | | | | | | |_| | | (_| | |_| | (_) | | | |
# |_|_| |_|_|\__|_|\__,_|_|  |___/_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|


#TUNEIT initial parameters
start_pars = dict(friction_params=torch.zeros(30), dyn_params=torch.zeros(66), hwc_params=torch.zeros(7), u_params=torch.zeros(6))
#start_pars = dict(friction_params=torch.ones(30)*10, dyn_params=torch.ones(66)*100, hwc_params=torch.zeros(7)*10, u_params=torch.zeros(6)*10)

SS_enc_dict = dict( \
    nx=12, na=0, nb=0, na_right=1, nb_right=0, \
    e_net=E_ypast, e_net_kwargs=dict(), \
    f_net=F_robust_nn, f_net_kwargs=dict(params=start_pars, normalization=None), \
    h_net=H_some_measurable, h_net_kwargs=dict(measurable_states=np.array(range(12))), f_norm=1.0, dt_base=1.0, intermediate_steps=1) #IMPORTANT

if fit_to_simulated_data:
    ftsd_file_version_counter = 20
    ftsd_fvc_loaded_from_file = -1
    try:
        previous_mat_data = loadmat('elabee_robot_ftsd.mat')
        ftsd_fvc_loaded_from_file = previous_mat_data['ftsd_file_version_counter']
    except:
        pass

    x_p_ref_training = torch.tensor(myfftresamp( torch.tensor(ref_trajectories['q_ref_1']).T, 15000 )[0].T, device=tgt_device, dtype=torch.float32)*(torch.pi/180) #[0] is there because we only want the TD result
    x_p_ref_validation = torch.tensor(myfftresamp( torch.tensor(ref_trajectories['q_ref_2']).T, 15000 )[0].T, device=tgt_device, dtype=torch.float32)*(torch.pi/180) #[0] is there because we only want the TD result
    x_p_ref_test = torch.tensor(myfftresamp( torch.tensor(ref_trajectories['q_ref_3']).T, 15000 )[0].T, device=tgt_device, dtype=torch.float32)*(torch.pi/180) #[0] is there because we only want the TD result
    
    """
    #plot x_p_ref_training with plotly, on 6 subplots:
    import plotly.graph_objects as go
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
    for i in range(6):
        fig.add_trace(go.Scatter(y=x_p_ref_training[:,i], mode='lines', name='ref. trajectory ({})'.format(i)), row=i+1, col=1)
        fig.add_trace(go.Scatter(y=previous_mat_data['y_training'][:,i], mode='lines', name='sim. trajectory ({})'.format(i)), row=i+1, col=1)
    fig.show()

    import plotly.graph_objects as go
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
    for i in range(6):
        fig.add_trace(go.Scatter(y=previous_mat_data['u_training'][:,i], mode='lines', name='torque for sim. trajectory ({})'.format(i)), row=i+1, col=1)
    fig.show()
    """

    """
    #plot x_p_ref_training with plotly, on 6 subplots:
    import plotly.graph_objects as go
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
    for i in range(6):
        fig.add_trace(go.Scatter(y=x_p_ref_training[:,i], mode='lines', name='x_p_ref_training[:,{}]'.format(i)), row=i+1, col=1)
        fig.add_trace(go.Scatter(y=previous_mat_data['y_training'][:,i], mode='lines', name='y_training'.format(i)), row=i+1, col=1)
    fig.show()

    #plot x_p_ref_validation with plotly, on 6 subplots:
    import plotly.graph_objects as go
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
    for i in range(6):
        fig.add_trace(go.Scatter(y=x_p_ref_validation[:,i], mode='lines', name='x_p_ref_validation[:,{}]'.format(i)), row=i+1, col=1)
        fig.add_trace(go.Scatter(y=previous_mat_data['y_validation'][:,i], mode='lines', name='y_validation'.format(i)), row=i+1, col=1)
    fig.show()

    #plot x_p_ref_test with plotly, on 6 subplots:
    import plotly.graph_objects as go
    fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
    for i in range(6):
        fig.add_trace(go.Scatter(y=x_p_ref_test[:,i], mode='lines', name='x_p_ref_test[:,{}]'.format(i)), row=i+1, col=1)
        fig.add_trace(go.Scatter(y=previous_mat_data['y_test'][:,i], mode='lines', name='y_test'.format(i)), row=i+1, col=1)
    fig.show()
    """

    if ftsd_fvc_loaded_from_file != ftsd_file_version_counter:
        print('ftsd_file_version_counter changed, so we will recompute the data and save it')

        #plot x_p_ref_training with plotly, on 6 subplots:
        #import plotly.graph_objects as go
        #fig = go.Figure()
        #for i in range(6):
        #    fig.add_trace(go.Scatter
        #        (x=np.arange(x_p_ref_training.shape[0]), y=x_p_ref_training[:,i],
        #        mode='lines', name='x_p_ref_training[:,{}]'.format(i)))
        #fig.show()
        
        ##plot x_p_ref_training in frequency domain, on dB scale, with plotly, on 6 subplots:
        #import plotly.graph_objects as go
        #fig = go.Figure()
        #for i in range(6):
        #    fig.add_trace(go.Scatter
        #        (x=np.arange(x_p_ref_training.shape[0]), y=20*np.log10(np.abs(np.fft.fft(x_p_ref_training[:,i]))),
        #        mode='lines', name='x_p_ref_training[:,{}]'.format(i)))
        #fig.show()

        #x_p_ref_training = torch.tensor(ref_trajectories['q_ref_1'], dtype=torch.float32)*(torch.pi/180)
        #x_p_ref_validation = torch.tensor(ref_trajectories['q_ref_2'], dtype=torch.float32)*(torch.pi/180)
        #x_p_ref_test = torch.tensor(ref_trajectories['q_ref_3'], dtype=torch.float32)*(torch.pi/180)

        N_resolution_upscale = 25
        x_p_ref_training = torch.tensor(myfftresamp( torch.tensor(ref_trajectories['q_ref_1']).T, 600*N_resolution_upscale )[0].T, device=tgt_device, dtype=torch.float32)*(torch.pi/180) #[0] is there because we only want the TD result
        x_p_ref_validation = torch.tensor(myfftresamp( torch.tensor(ref_trajectories['q_ref_2']).T, 600*N_resolution_upscale )[0].T, device=tgt_device, dtype=torch.float32)*(torch.pi/180) #[0] is there because we only want the TD result
        x_p_ref_test = torch.tensor(myfftresamp( torch.tensor(ref_trajectories['q_ref_3']).T, 600*N_resolution_upscale )[0].T, device=tgt_device, dtype=torch.float32)*(torch.pi/180) #[0] is there because we only want the TD result
        #plot the x_p_ref_training and x_p_ref_training2 on each other, with plotly:
        #import plotly.graph_objects as go
        #fig = go.Figure()
        #fig.add_trace(go.Scatter(x=torch.arange(x_p_ref_training.shape[0]), y=x_p_ref_training[:,0], mode='lines', name='x_p_ref_training'))
        #fig.add_trace(go.Scatter(x=torch.arange(x_p_ref_training2.shape[0]), y=x_p_ref_training2[:,0], mode='lines', name='x_p_ref_training2'))
        #fig.show()

        print('N_samples', N_samples)

        #calculate the derivative of x_p_ref_training in the frequency domain
        x_p_ref_training_reconstr, x_v_ref_training, x_a_ref_training, _ = diff_fd(x_p_ref_training, None, dim=0, dt=data_dt)
        x_p_ref_training_reconstr, x_v_ref_validation, x_a_ref_validation, _ = diff_fd(x_p_ref_validation, None, dim=0, dt=data_dt)
        x_p_ref_training_reconstr, x_v_ref_test, x_a_ref_test, _ = diff_fd(x_p_ref_test, None, dim=0, dt=data_dt)
        x_ref_training = torch.cat((x_p_ref_training, x_v_ref_training, x_a_ref_training), dim=1)
        x_ref_validation = torch.cat((x_p_ref_validation, x_v_ref_validation, x_a_ref_validation), dim=1)
        x_ref_test = torch.cat((x_p_ref_test, x_v_ref_test, x_a_ref_test), dim=1)

        sys_data_generation = SS_encoder_deriv_weighted_general(**SS_enc_dict)
        sys_data_generation.init_model(nu=6, ny=12, device=tgt_device)
        sys_data_generation.fn.integrator.dt = data_dt
        rclsim = robot_closed_loop_simulator(sys_data_generation.fn, sys_data_generation.derivn.robot)
        x0 = torch.zeros(12, device=tgt_device)
        #generate a tensor of 12 random numbers with mean 0 and std 1e-6

        #for debug: plot the reference trajectories
        #plt.clf(); plt.plot(x_p_ref_training.detach().numpy()); mkpng('x_p_ref_training', title=True)
        #plt.clf(); plt.plot(x_p_ref_validation.detach().numpy()); mkpng('x_p_ref_validation', title=True)
        #plt.clf(); plt.plot(x_p_ref_test.detach().numpy()); mkpng('x_p_ref_test', title=True)
        #plt.clf(); plt.plot(x_v_ref_training.detach().numpy()); mkpng('x_v_ref_training', title=True)
        #plt.clf(); plt.plot(x_v_ref_validation.detach().numpy()); mkpng('x_v_ref_validation', title=True)
        #plt.clf(); plt.plot(x_v_ref_test.detach().numpy()); mkpng('x_v_ref_test', title=True)
        #plt.clf(); plt.plot(x_a_ref_training.detach().numpy()); mkpng('x_a_ref_training', title=True)
        #plt.clf(); plt.plot(x_a_ref_validation.detach().numpy()); mkpng('x_a_ref_validation', title=True)
        #plt.clf(); plt.plot(x_a_ref_test.detach().numpy()); mkpng('x_a_ref_test', title=True)

        N_repeat_trajectory = 3
        sys_data_generation.derivn.robot.modifier = True
        sys_data_generation.derivn.robot.enable_parts_dict = False
        data_fit_input_training, data_fit_input_training_tau = rclsim.apply_experiment(x0, x_ref_training.repeat(N_repeat_trajectory, 1)) #TODO I did not implement the N_samples_cutoff here on x_ref_training_data
        """
        rolsim = robot_open_loop_simulator(sys_data_generation.fn)
        x0_rolcheck = torch.tensor([ 0.00069261, 0.01337057, 0.00159142, 0.00543189, -0.01210412, -0.00715754, 0.00750225, 0.00406193, 0.00648657, -0.00797173, -0.01799383, 0.01206385 ], device=tgt_device)
        #torch.randn(12, device=tgt_device)*1e-2
        data_fit_input_rolcheck_training = rolsim.apply_experiment(x0_rolcheck, data_fit_input_training_tau)
        plt.clf(); compare(data_fit_input_training, data_fit_input_rolcheck_training, title='data_fit_input_training_y_rolcheck', mkpng_filename='data_fit_input_training_y_rolcheck')
        """
        plt.clf(); plt.plot(data_fit_input_training.y[:,0:6]); plt.legend(['q1','q2','q3','q4','q5','q6']); mkpng('data_fit_input_training_y', title=True)
        plt.clf(); plt.plot(data_fit_input_training.u); mkpng('data_fit_input_training_u', title=True)
        plt.clf(); plt.plot(torch.tensor(data_fit_input_training.y[:,0:6]) - x_p_ref_training.repeat(N_repeat_trajectory, 1)); plt.legend(['q1','q2','q3','q4','q5','q6']); mkpng('data_fit_input_training_y_pos_error', title=True)
        plt.clf(); plt.plot(torch.tensor(data_fit_input_training.y[:,6:12]) - x_v_ref_training.repeat(N_repeat_trajectory, 1)); plt.legend(['q1','q2','q3','q4','q5','q6']); mkpng('data_fit_input_training_y_vel_error', title=True)

        print('training data: NRMSE of position: ', torch.norm(torch.tensor(data_fit_input_training.y[:,0:6]) - x_p_ref_training.repeat(N_repeat_trajectory, 1), dim=0)/torch.norm(x_p_ref_training, dim=0))
        print('training data: NRMSE of velocity: ', torch.norm(torch.tensor(data_fit_input_training.y[:,6:12]) - x_v_ref_training.repeat(N_repeat_trajectory, 1), dim=0)/torch.norm(x_v_ref_training, dim=0))

        data_fit_input_validation, data_fit_input_validation_tau = rclsim.apply_experiment(x0, x_ref_validation.repeat(N_repeat_trajectory, 1))
        data_fit_input_test, data_fit_input_test_tau = rclsim.apply_experiment(x0, x_ref_test.repeat(N_repeat_trajectory, 1))
        sys_data_generation.derivn.robot.modifier = False
        data_fit_input_training = data_fit_input_training[-N_samples:] #we get the last period
        data_fit_input_training_tau = data_fit_input_training_tau[-N_samples:,:]
        data_fit_input_validation = data_fit_input_validation[-N_samples:]
        data_fit_input_validation_tau = data_fit_input_validation_tau[-N_samples:,:]
        data_fit_input_test = data_fit_input_test[-N_samples:]
        data_fit_input_test_tau = data_fit_input_test_tau[-N_samples:,:]

        plt.clf(); plt.plot(20*torch.fft.rfft(torch.tensor(data_fit_input_training.y, dtype=torch.float32), dim=0).abs().log10()); mkpng('data_fit_input_training_fd', title=True)
        plt.clf(); plt.plot(data_fit_input_training.y[:,0:6]);    plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_training_yp', title=True)
        plt.clf(); plt.plot(data_fit_input_training.y[:,6:12]);   plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_training_yv', title=True)
        plt.clf(); plt.plot(data_fit_input_training.u);           plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_training_u', title=True)
        plt.clf(); plt.plot(data_fit_input_validation.y[:,0:6]);  plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_validation_yp', title=True)
        plt.clf(); plt.plot(data_fit_input_validation.y[:,6:12]); plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_validation_yv', title=True)
        plt.clf(); plt.plot(data_fit_input_validation.u);         plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_validation_u', title=True)
        plt.clf(); plt.plot(data_fit_input_test.y[:,0:6]);        plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_test_yp', title=True)
        plt.clf(); plt.plot(data_fit_input_test.y[:,6:12]);       plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_test_yv', title=True)
        plt.clf(); plt.plot(data_fit_input_test.u);               plt.legend(['1','2','3','4','5','6']); mkpng('data_fit_input_test_u', title=True)

        mat_data = dict(u_training=data_fit_input_training.u, y_training=data_fit_input_training.y, \
                        u_validation=data_fit_input_validation.u, y_validation=data_fit_input_validation.y, \
                        u_test=data_fit_input_test.u, y_test=data_fit_input_test.y, \
                        ftsd_file_version_counter=ftsd_file_version_counter)
        savemat('elabee_robot_ftsd.mat', mat_data)
        print('savedmat')
        sys.exit(0)
    else:
        print('ftsd_file_version_counter did not change, so we will load the data')
        data_fit_input_training = System_data(previous_mat_data['u_training'], previous_mat_data['y_training'], dt=data_dt)
        data_fit_input_validation = System_data(previous_mat_data['u_validation'], previous_mat_data['y_validation'], dt=data_dt)
        data_fit_input_test = System_data(previous_mat_data['u_test'], previous_mat_data['y_test'], dt=data_dt)
        data_fit_input_training   = diff_v_from_p_and_filter_yu(data_fit_input_training) #note: v will be garbage because we didn't filter it
        data_fit_input_validation = diff_v_from_p_and_filter_yu(data_fit_input_validation)
        data_fit_input_test       = diff_v_from_p_and_filter_yu(data_fit_input_test)

"""
#plot data_fit_input_training with plotly on 6 subplots
fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
for i in range(6):
    fig.add_trace(go.Scatter(y=data_fit_input_test.y[:,i], name='y'+str(i)), row=i+1, col=1)
fig.show()
fig = make_subplots(rows=6, cols=1, shared_xaxes=True)
for i in range(6):
    fig.add_trace(go.Scatter(y=data_fit_input_test.u[:,i], name='u'+str(i)), row=i+1, col=1)
fig.show()
"""

#view my dataset
if show_input_on_start:
    compare(data_fit_input_training, data_fit_input_training, 'data_fit_input_training', mkpng_filename='data_fit_input_training')
    compare(data_fit_input_validation, data_fit_input_validation, 'data_fit_input_validation', mkpng_filename='data_fit_input_validation')
    compare(data_fit_input_test, data_fit_input_test, 'data_fit_input_test', mkpng_filename='data_fit_input_test')

#Here is the place to slightly change the parameters, if fit_to_simulated_data was active #CASE
#start_pars = dict(friction_params=friction_params_goodepoch, dyn_params=0.95*dyn_param_guess_goodepoch, hwc_params=hwc_params_goodepoch, u_params=u_params_goodepoch) #TUNEIT #CASE
#SS_enc_dict['f_net_kwargs']['params'] = start_pars

"""
#what are we doing here? we are actually plotting the data and the model output, in case the a*tanh(b*v) friction is switched on or not
sys_initial_simulation = SS_encoder_deriv_weighted_general(**SS_enc_dict)
sys_initial_simulation.init_model(nu=6, ny=12, device=tgt_device) #the only difference between this and sys_trained should be that here e_net=E_specify_initial_states
sys_initial_simulation.fn.integrator.dt = data_dt
with torch.no_grad():
    rolsim = robot_open_loop_simulator(sys_initial_simulation.fn)
    x0 = torch.zeros(12, device=tgt_device)
    rolsim_output  = rolsim.apply_experiment(x0, torch.tensor(data_fit_input_training.u, dtype=torch.float32))
    sys_initial_simulation.derivn.robot.modifier = True
    #once we set this true, the we will switch on the last term in the friction
    rolsim_output_2  = rolsim.apply_experiment(x0, torch.tensor(data_fit_input_training.u, dtype=torch.float32))
torch.save(rolsim_output, 'arolsim_output.torchsave')
torch.save(rolsim_output_2, 'arolsim_output_2.torchsave')

#create figure with 6 subplots with plotly express
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.001)
for i in range(6):
    #fig.add_trace(go.Scatter(y=data_fit_input_training.y[:,i], name='measurement', line=dict(color="#000000")), row=i+1, col=1)
    fig.add_trace(go.Scatter(y=rolsim_output.y[:,i], name='sim. without a*tanh(b*v)', line=dict(color="#0000ff")), row=i+1, col=1)
    fig.add_trace(go.Scatter(y=rolsim_output_2.y[:,i], name='sim. with a*tanh(b*v)', line=dict(color="#ff0000")), row=i+1, col=1)
fig.show()
"""

if generate_initial_fit_figures:
    #2023_big_rewrite: this needs to be changed to N-step-NRMS. It doesn't make sense to generate these single simulations anymore. 
    sys_initial_simulation = SS_encoder_deriv_weighted_general(**SS_enc_dict)
    sys_initial_simulation.init_model(nu=6, ny=12, device=tgt_device) #the only difference between this and sys_trained should be that here e_net=E_specify_initial_states
    sys_initial_simulation.fn.integrator.dt = data_dt
    initial_analysis_results = plot_full_system_analysis(sys_initial_simulation, data_fit_input_training, data_fit_input_validation, data_fit_input_test, state_name='initial', plot_loss_history=False, with_nn=False, extra_plots=True)

#                                _  _                   _
#                               | |(_)              _  (_)   normalization
#  ____   ___   ____ ____  _____| | _ _____ _____ _| |_ _  ___  ____
# |  _ \ / _ \ / ___)    \(____ | || (___  |____ (_   _) |/ _ \|  _ \
# | | | | |_| | |   | | | / ___ | || |/ __// ___ | | |_| | |_| | | | |
# |_| |_|\___/|_|   |_|_|_\_____|\_)_(_____)_____|  \__)_|\___/|_| |_|

tonorm_u = torch.tensor(data_fit_input_training.u, dtype=torch.float32)
tonorm_pos = torch.tensor(data_fit_input_training.y[:,0:6], dtype=torch.float32)
tonorm_posf, tonorm_v, tonorm_a, tonorm_uf = diff_fd(tonorm_pos, tonorm_u, dim=0, dt=data_dt) 
torch.save(tonorm_a, 'tonorm_a.torchsave')

u_mean = torch.mean(tonorm_uf, dim=0)
u_std = torch.std(tonorm_uf, dim=0)
pos_mean = torch.mean(tonorm_posf, dim=0)
pos_std = torch.std(tonorm_posf, dim=0)
v_mean = torch.mean(tonorm_v, dim=0)
v_std = torch.std(tonorm_v, dim=0)
a_mean = torch.mean(tonorm_a, dim=0)
a_std = torch.std(tonorm_a, dim=0)
normalization = dict(u_mean=u_mean, u_std=u_std, pos_mean=pos_mean, pos_std=pos_std, v_mean=v_mean, v_std=v_std, a_mean=a_mean, a_std=a_std)
SS_enc_dict['f_net_kwargs']['normalization']=normalization

#plot tonorm_a with plotly on 6 subplots:
"""
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.001)
for i in range(6):
    fig.add_trace(go.Scatter(y=tonorm_posf[:,i], name='tonorm_posf '+str(i)), row=i+1, col=1)
fig.show()
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.001)
for i in range(6):
    fig.add_trace(go.Scatter(y=tonorm_v[:,i], name='tonorm_v '+str(i)), row=i+1, col=1)
fig.show()
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.001)
for i in range(6):
    fig.add_trace(go.Scatter(y=tonorm_a[:,i], name='tonorm_a '+str(i)), row=i+1, col=1)
fig.show()
fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.001)
for i in range(6):
    fig.add_trace(go.Scatter(y=tonorm_uf[:,i], name='tonorm_uf '+str(i)), row=i+1, col=1)
fig.show()
"""

#   __ _ _   _   _
#  / _(_) | | | (_)
# | |_ _| |_| |_ _ _ __   __ _
# |  _| | __| __| | '_ \ / _` |
# | | | | |_| |_| | | | | (_| |
# |_| |_|\__|\__|_|_| |_|\__, |
#                         __/ |
#                        |___/

# sys_trained.checkpoint_load_system('_last')

SS_enc_dict['e_net']=E_ypast
SS_enc_dict['e_net_kwargs']=dict()

#TUNEIT
#SS_enc_dict['na']=11 #ENCODER
#SS_enc_dict['na_right']=1 #ENCODER
#SS_enc_dict['nb']=12 #ENCODER
#SS_enc_dict['nb_right']=0 #ENCODER
#SS_enc_dict['e_net']=E_normalized_nn #ENCODER


#SS_enc_dict['lambda_encoder_in_obj']=dict(hn=2,x=0.3) #TUNEIT #ENCODER
#SS_enc_dict['lambda_encoder_in_obj']=dict(hn=0,x=0) #TUNEIT
#SS_enc_dict['lambda_encoder_in_obj']=None #TUNEIT

#not: "k1_nlspring","fv1", "fv2"
#SS_enc_dict['f_net_kwargs']=dict(params=start_pars) #TUNEIT param_guess_v initial parameters
SS_enc_dict['f_net_kwargs']=dict(params=start_pars, normalization=normalization) #TUNEIT param_guess_v initial parameters
#key:(item*0.9 if isinstance(item,float) else item) for key,item in start_pars.items() #working reference of making 0.9* all the parameters
#SS_enc_dict['obj_weights']=torch.tensor([1., 1., 10., 10.])[measurable_states] #TUNEIT
SS_enc_dict['h_net_kwargs']=dict(measurable_states=measurable_states)
SS_enc_dict['n_step_nrms_mode']=False if not from_exp_dir else (exp_obj_counter==0)  #PAPER

def readict(filename):
    #read dictionary from text file
    with open(filename, 'r') as f:
        readicted = json.load(f)
    return readicted

def scheduler_fn_phy_lr_phy(epoch):
    if not from_exp_dir: return 0
    else: 
        if exp_model_counter in [2,3,7]: 
            return 1
        else: 
            if epoch < epoch_patience_min: return 1e0
            else: return 0
    #if epoch < 1000: return 0
    #else: return 1e-1
    #return readict('lr.txt')['fn_phy']

def scheduler_fn_ann_lr_phy(epoch):
    if not from_exp_dir: return 1e-4
    else: 
        if epoch < epoch_patience_min: 
            return 0
        else:
            if exp_model_counter in [1,13,12,14,15]: return 1e-5
            else: return 1e-4
    #return readict('lr.txt')['fn_ann']

scheduler_kwargs = dict(scheduler=optim.lr_scheduler.LambdaLR, lr_lambda=[ scheduler_fn_phy_lr_phy, scheduler_fn_ann_lr_phy ], verbose=False) #TUNEIT adjustable learning rate #PRINTIT verbose
nf_value = 15000

sys_trained = SS_encoder_deriv_weighted_general(**SS_enc_dict)
sys_trained.init_model(nu=6, ny=12, device=tgt_device)
file_phy_params = open('phy_params.txt','w')
file_phy_params.write('========================== initial')
sys_trained.dump_params(file_phy_params)
batch_size = int(os.environ['BATCHSIZE']) if 'BATCHSIZE' in os.environ else 512
print('batch_size = ', batch_size)

del sys_trained
sys_trained = SS_encoder_deriv_weighted_general(**SS_enc_dict)
sys_trained.init_nets(6, 12)
sys_trained.loss(None, None, \
    torch.cat((torch.tensor(data_fit_input_training.u)[None,:,:],torch.tensor(data_fit_input_validation.u)[None,:,:],torch.tensor(data_fit_input_test.u)[None,:,:]),dim=0),\
    torch.cat((torch.tensor(data_fit_input_training.y)[None,:,:],torch.tensor(data_fit_input_validation.y)[None,:,:],torch.tensor(data_fit_input_test.y)[None,:,:]),dim=0),\
    epoch=0, initial_std_mean=True, param_groups = [])
#base_ann_state_dict = torch.load('best_model_state_dict_base_ann.torch_save')
sys_trained.epoch_patience = 1000
sys_trained.epoch_patience_min = epoch_patience_min
sys_trained.fit(data_fit_input_training, data_fit_input_validation, epochs=50000, validation_measure=('andras-nstep-10' if (SS_enc_dict['n_step_nrms_mode'] if not from_exp_dir else (exp_obj_counter==0 or exp_obj_counter==3)) else 'andras-derivn'), loss_kwargs=dict(nf=nf_value), batch_size=batch_size, optimizer_kwargs=dict(lr=1,betas=(0.99,0.999)), auto_fit_norm=False, scheduler_kwargs=scheduler_kwargs, cuda=cuda_on, sqrt_train=False) #TUNEIT
#default optimizer_kwargs: dict(lr=1e-3, betas=(0.9, 0.999))

#      _                 _       _   _
#     (_)               | |     | | (_)
#  ___ _ _ __ ___  _   _| | __ _| |_ _  ___  _ __
# / __| | '_ ` _ \| | | | |/ _` | __| |/ _ \| '_ \
# \__ \ | | | | | | |_| | | (_| | |_| | (_) | | | |
# |___/_|_| |_| |_|\__,_|_|\__,_|\__|_|\___/|_| |_|

last_model_state_dict = {'optimizer_state_dict': sys_trained.optimizer.state_dict(), 'fn_state_dict': sys_trained.fn.state_dict(), 'decorrelation_matrix': sys_trained.derivn.decorrelation_matrix}
torch.save(last_model_state_dict, 'last_model_state_dict.torch_save')
print('sys_trained.i_am_bestmodel:',sys_trained.i_am_bestmodel)
#with open('last_model_state_dict.txt', 'w') as f: f.write(str(last_model_state_dict))

#print("Identified parameters (rescaled):", sys_trained.derivn.not_scaled_parameters)
#sys.trained.hn.measurable_states = np.array([1,2,3,4])
if cuda_on: sys_trained.cuda()
#sys_trained.encoder.cuda()
print('finished training, calculating simulations with the final parameters...')
print(' === lastmodel ===')
file_phy_params.write('========================== lastmodel')
sys_trained.dump_params(file_phy_params)
lastmodel_analysis_results = plot_full_system_analysis(sys_trained, data_fit_input_training, data_fit_input_validation, data_fit_input_test, state_name='after_fit_lastmodel')
exp_epoch_counter = sys_trained.epoch_counter
exp_bestmodel_epoch = sys_trained.bestmodel_epoch

if sys_trained.bestmodel is not None:
    print('Loading the best model sys_trained.bestmodel')
    sys_trained.__dict__ = sys_trained.bestmodel
    sys_trained.finalize_model()
    print('sys_trained.i_am_bestmodel:',sys_trained.i_am_bestmodel)
    best_model_state_dict = {'optimizer_state_dict': sys_trained.optimizer.state_dict(), 'fn_state_dict': sys_trained.fn.state_dict(), 'decorrelation_matrix': sys_trained.derivn.decorrelation_matrix}
    torch.save(best_model_state_dict, 'best_model_state_dict.torch_save')

    print ('=== bestmodel ===')
    file_phy_params.write('========================== bestmodel')
    sys_trained.dump_params(file_phy_params)
    bestmodel_analysis_results = plot_full_system_analysis(sys_trained, data_fit_input_training, data_fit_input_validation, data_fit_input_test, state_name='after_fit_bestmodel', extra_plots=True)
    we_have_bestmodel = True
else:
    bestmodel_analysis_results = lastmodel_analysis_results
    we_have_bestmodel = False

try:
    torch.save(sys_trained, 'sys_trained.torch_save')
    print('saving sys_trained succeeded')
except: print('saving sys_trained failed')

if from_exp_dir: #then write the CSV:
    with open('result.csv', 'w') as f:
        #write the CSV header based on the line after:
        f.write(f"the_expid,exp_real_data,exp_real_data_explanation,exp_model_counter,exp_model_counter_explanation,"+
                f"exp_obj_counter,exp_obj_counter_explanation,"+
                f"initial_results_with_nn,bestmodel_results_with_nn,lastmodel_results_with_nn,"+
                f"initial_training_{mon_steps[0]}_step_nrms,initial_validation_{mon_steps[0]}_step_nrms,initial_test_{mon_steps[0]}_step_nrms,"+
                f"bestmodel_training_{mon_steps[0]}_step_nrms,bestmodel_validation_{mon_steps[0]}_step_nrms,bestmodel_test_{mon_steps[0]}_step_nrms,"+
                f"lastmodel_training_{mon_steps[0]}_step_nrms,lastmodel_validation_{mon_steps[0]}_step_nrms,lastmodel_test_{mon_steps[0]}_step_nrms,"+
                f"bestmodel_without_nn_training_{mon_steps[0]}_step_nrms,bestmodel_without_nn_validation_{mon_steps[0]}_step_nrms,bestmodel_without_nn_test_{mon_steps[0]}_step_nrms,"+
                f"initial_training_{mon_steps[1]}_step_nrms,initial_validation_{mon_steps[1]}_step_nrms,initial_test_{mon_steps[1]}_step_nrms,"+
                f"bestmodel_training_{mon_steps[1]}_step_nrms,bestmodel_validation_{mon_steps[1]}_step_nrms,bestmodel_test_{mon_steps[1]}_step_nrms,"+
                f"lastmodel_training_{mon_steps[1]}_step_nrms,lastmodel_validation_{mon_steps[1]}_step_nrms,lastmodel_test_{mon_steps[1]}_step_nrms,"+
                f"bestmodel_without_nn_training_{mon_steps[1]}_step_nrms,bestmodel_without_nn_validation_{mon_steps[1]}_step_nrms,bestmodel_without_nn_test_{mon_steps[1]}_step_nrms,"+
                f"initial_training_r_squared,initial_validation_r_squared,initial_test_r_squared,"+
                f"bestmodel_training_r_squared,bestmodel_validation_r_squared,bestmodel_test_r_squared,"+
                f"lastmodel_training_r_squared,lastmodel_validation_r_squared,lastmodel_test_r_squared,"+
                f"bestmodel_first_loss_val,bestmodel_last_loss_val,bestmodel_num_loss_val,"+
                f"bestmodel_first_loss_train,bestmodel_last_loss_train,bestmodel_num_loss_train,"+
                f"cvl_initial_loss_train,cvl_initial_loss_validation,cvl_initial_loss_test,"+
                f"cvl_best_loss_train,cvl_best_loss_validation,cvl_best_loss_test,"+
                f"cvl_last_loss_train,cvl_last_loss_validation,cvl_last_loss_test,"+
                f"epochs,epoch_bestmodel,we_have_bestmodel,exp_weight_decay,exp_nn_size\n")
        f.write(f"{the_expid},{exp_real_data},{exp_real_data_explanation},{exp_model_counter},{exp_model_counter_explanation},"+
                f"{exp_obj_counter},{exp_obj_counter_explanation},"+
                f"{initial_analysis_results['with_nn']},{bestmodel_analysis_results['with_nn']},{lastmodel_analysis_results['with_nn']},"+
                f"{initial_analysis_results['training'][mon_steps[0]]},{initial_analysis_results['validation'][mon_steps[0]]},{initial_analysis_results['test'][mon_steps[0]]},"+
                f"{bestmodel_analysis_results['training'][mon_steps[0]]},{bestmodel_analysis_results['validation'][mon_steps[0]]},{bestmodel_analysis_results['test'][mon_steps[0]]},"+
                f"{lastmodel_analysis_results['training'][mon_steps[0]]},{lastmodel_analysis_results['validation'][mon_steps[0]]},{lastmodel_analysis_results['test'][mon_steps[0]]},"+
                f"{bestmodel_analysis_results['without_nn_results']['training'][mon_steps[0]]},{bestmodel_analysis_results['without_nn_results']['validation'][mon_steps[0]]},{bestmodel_analysis_results['without_nn_results']['test'][mon_steps[0]]},"+
                f"{initial_analysis_results['training'][mon_steps[1]]},{initial_analysis_results['validation'][mon_steps[1]]},{initial_analysis_results['test'][mon_steps[1]]},"+
                f"{bestmodel_analysis_results['training'][mon_steps[1]]},{bestmodel_analysis_results['validation'][mon_steps[1]]},{bestmodel_analysis_results['test'][mon_steps[1]]},"+
                f"{lastmodel_analysis_results['training'][mon_steps[1]]},{lastmodel_analysis_results['validation'][mon_steps[1]]},{lastmodel_analysis_results['test'][mon_steps[1]]},"+
                f"{bestmodel_analysis_results['without_nn_results']['training'][mon_steps[1]]},{bestmodel_analysis_results['without_nn_results']['validation'][mon_steps[1]]},{bestmodel_analysis_results['without_nn_results']['test'][mon_steps[1]]},"+
                f"{initial_analysis_results['training']['rsq']},{initial_analysis_results['validation']['rsq']},{initial_analysis_results['test']['rsq']},"+
                f"{bestmodel_analysis_results['training']['rsq']},{bestmodel_analysis_results['validation']['rsq']},{bestmodel_analysis_results['test']['rsq']},"+
                f"{lastmodel_analysis_results['training']['rsq']},{lastmodel_analysis_results['validation']['rsq']},{lastmodel_analysis_results['test']['rsq']},"+
                f"{bestmodel_analysis_results['loss_history']['first_loss_val']},{bestmodel_analysis_results['loss_history']['last_loss_val']},{bestmodel_analysis_results['loss_history']['num_loss_val']},"+
                f"{bestmodel_analysis_results['loss_history']['first_loss_train']},{bestmodel_analysis_results['loss_history']['last_loss_train']},{bestmodel_analysis_results['loss_history']['num_loss_train']},"+
                f"{initial_analysis_results['training']['loss']},{initial_analysis_results['validation']['loss']},{initial_analysis_results['test']['loss']},"+
                f"{bestmodel_analysis_results['training']['loss']},{bestmodel_analysis_results['validation']['loss']},{bestmodel_analysis_results['test']['loss']},"+
                f"{lastmodel_analysis_results['training']['loss']},{lastmodel_analysis_results['validation']['loss']},{lastmodel_analysis_results['test']['loss']},"+
                f"{exp_epoch_counter},{exp_bestmodel_epoch},{we_have_bestmodel},{exp_weight_decay},{exp_nn_size}\n")

if not from_exp_dir:
    import code
    try:
        from ptpython.repl import embed
    except ImportError:
        print("ptpython is not available: falling back to standard prompt")
        code.interact(local=dict(globals(), **locals()))
    else:
        embed(globals(), locals())
else:
    portlo.release()

print('every ending is a new beginning')
