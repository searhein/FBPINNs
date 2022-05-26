#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 23:48:06 2021

@author: bmoseley
"""

# This script reproduces all of the paper results

import sys

import numpy as np

import problems
from active_schedulers import AllActiveSchedulerND, PointActiveSchedulerND, LineActiveSchedulerND, PlaneActiveSchedulerND
from constants import Constants, get_subdomain_xs, get_subdomain_ws
from main import FBPINNTrainer, PINNTrainer
from trainersBase import train_models_multiprocess

sys.path.insert(0, '../shared_modules/')
import multiprocess



# constants constructors

def run_PINN():
    sampler = "r" if random else "m"
    c = Constants(
                  RUN="final_PINN_%s_%sh_%sl_%sb_%ss_%s"%(P.name, n_hidden, n_layers, batch_size[0], n_steps, sampler),
                  P=P,
                  SUBDOMAIN_XS=subdomain_xs,
                  BOUNDARY_N=boundary_n,
                  Y_N=y_n,
                  N_HIDDEN=n_hidden,
                  N_LAYERS=n_layers,
                  BATCH_SIZE=batch_size,
                  RANDOM=random,
                  N_STEPS=n_steps,
                  BATCH_SIZE_TEST=batch_size_test,
                  PLOT_LIMS=plot_lims,
                  MODEL_SAVE_FREQ=save_frequency
                  )
    return c, PINNTrainer

def run_FBPINN():
    sampler = "r" if random else "m"
    c = Constants(
                  RUN="final_FBPINN_%s_%sh_%sl_%sb_%s_%sw_%s_%su"%(P.name, n_hidden, n_layers, batch_size[0], sampler, width, A.name, n_update_every_iterations),
                  P=P,
                  SUBDOMAIN_XS=subdomain_xs,
                  SUBDOMAIN_WS=subdomain_ws,
                  BOUNDARY_N=boundary_n,
                  Y_N=y_n,
                  ACTIVE_SCHEDULER=A,
                  ACTIVE_SCHEDULER_ARGS=args,
                  N_HIDDEN=n_hidden,
                  N_LAYERS=n_layers,
                  BATCH_SIZE=batch_size,
                  RANDOM=random,
                  N_STEPS=n_steps,
                  BATCH_SIZE_TEST=batch_size_test,
                  PLOT_LIMS=plot_lims,
                  )
    return c, FBPINNTrainer


# DEFINE PROBLEMS


# below uses 200 points per w


runs = []

plot_lims = (1.1, False)
random = False


# # 1D PROBLEMS
#
# # Cos w=1
#
# P = problems.Cos1D_1(w=1, A=0)
# subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
# boundary_n = (1/P.w,)
# y_n = (0,1/P.w)
# batch_size_test = (5000,)
#
# n_steps = 50000
# for n_hidden, n_layers, batch_size in [
#                                        (16, 2, (100,)),
#                                        (16, 2, (500,)),
#                                        (16, 2, (1000,)),
#                                        (16, 2, (5000,)),
#                                        (32, 3, (100,)),
#                                        (32, 3, (500,)),
#                                        (32, 3, (1000,)),
#                                        (32, 3, (5000,)),
#                                        (64, 4, (100,)),
#                                        (64, 4, (500,)),
#                                        (64, 4, (1000,)),
#                                        (64, 4, (5000,)),
#                                        (128, 5, (100,)),
#                                        (128, 5, (500,)),
#                                        (128, 5, (1000,)),
#                                        (128, 5, (5000,))]:
#     runs.append(run_PINN())
#
# # Cos w=5
#
# P = problems.Cos1D_1(w=5, A=0)
# subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
# boundary_n = (1/P.w,)
# y_n = (0,1/P.w)
# batch_size_test = (5000,)
#
# n_steps = 50000
# for n_hidden, n_layers, batch_size in [
#                                        (16, 2, (100,)),
#                                        (16, 2, (500,)),
#                                        (16, 2, (1000,)),
#                                        (16, 2, (5000,)),
#                                        (32, 3, (100,)),
#                                        (32, 3, (500,)),
#                                        (32, 3, (1000,)),
#                                        (32, 3, (5000,)),
#                                        (64, 4, (100,)),
#                                        (64, 4, (500,)),
#                                        (64, 4, (1000,)),
#                                        (64, 4, (5000,)),
#                                        (128, 5, (100,)),
#                                        (128, 5, (500,)),
#                                        (128, 5, (1000,)),
#                                        (128, 5, (5000,))]:
#     runs.append(run_PINN())
#
# # Cos w=10
#
# P = problems.Cos1D_1(w=10, A=0)
# subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
# boundary_n = (1/P.w,)
# y_n = (0,1/P.w)
# batch_size_test = (5000,)
#
# n_steps = 50000
# for n_hidden, n_layers, batch_size in [
#                                        (16, 2, (100,)),
#                                        (16, 2, (500,)),
#                                        (16, 2, (1000,)),
#                                        (16, 2, (5000,)),
#                                        (32, 3, (100,)),
#                                        (32, 3, (500,)),
#                                        (32, 3, (1000,)),
#                                        (32, 3, (5000,)),
#                                        (64, 4, (100,)),
#                                        (64, 4, (500,)),
#                                        (64, 4, (1000,)),
#                                        (64, 4, (5000,)),
#                                        (128, 5, (100,)),
#                                        (128, 5, (500,)),
#                                        (128, 5, (1000,)),
#                                        (128, 5, (5000,))]:
#     runs.append(run_PINN())
#
# # Cos w=15
#
# P = problems.Cos1D_1(w=15, A=0)
# subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
# boundary_n = (1/P.w,)
# y_n = (0,1/P.w)
# batch_size_test = (5000,)
#
# n_steps = 50000
# for n_hidden, n_layers, batch_size in [
#                                        (16, 2, (100,)),
#                                        (16, 2, (500,)),
#                                        (16, 2, (1000,)),
#                                        (16, 2, (5000,)),
#                                        (32, 3, (100,)),
#                                        (32, 3, (500,)),
#                                        (32, 3, (1000,)),
#                                        (32, 3, (5000,)),
#                                        (64, 4, (100,)),
#                                        (64, 4, (500,)),
#                                        (64, 4, (1000,)),
#                                        (64, 4, (5000,)),
#                                        (128, 5, (100,)),
#                                        (128, 5, (500,)),
#                                        (128, 5, (1000,)),
#                                        (128, 5, (5000,))]:
#     runs.append(run_PINN())


# Cos w=1 and w=15

# P = problems.Cos_multi1D_1(w1=1, w2=15, A=0)
# subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
# boundary_n = (1/P.w2,)
# y_n = (0,2)
# batch_size_test = (5000,)
#
# for n_hidden, n_layers, batch_size, n_steps in [
#                                        (20, 2, (1001,), 3000),
#                                        (20, 2, (1001,), 3100),
#                                        (20, 2, (1001,), 3200),
#                                        (20, 2, (1001,), 3300),
#                                        (20, 2, (1001,), 3400),
#                                        (20, 2, (1001,), 3500),
#                                        (20, 2, (1001,), 3600),
#                                        (20, 2, (1001,), 3700),
#                                        (20, 2, (1001,), 3800),
#                                        (20, 2, (1001,), 3900),
#                                        (20, 2, (1001,), 4000)]:
#     save_frequency = n_steps
#     runs.append(run_PINN())

# Cos w=15

P = problems.Cos1D_1(w=15, A=0)
subdomain_xs = get_subdomain_xs([np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])], [2*np.pi])
boundary_n = (1/P.w,)
y_n = (0,1/P.w)
batch_size = (3000,)
batch_size_test = (5000,)

n_hidden, n_layers = 16, 2
width = 0.7
subdomain_ws = get_subdomain_ws(subdomain_xs, width)
args = ()
for A,n_update_every_iterations,n_steps in [(AllActiveSchedulerND, 1, 50000),
                                            (AllActiveSchedulerND, 5, 50000),
                                            (AllActiveSchedulerND, 10, 50000),
                                            (AllActiveSchedulerND, 50, 50000),
                                            (AllActiveSchedulerND, 100, 50000)]:
    runs.append(run_FBPINN())

if __name__ == "__main__":# required for multiprocessing

    import socket
    
    # GLOBAL VARIABLES
    
    # parallel devices (GPUs/ CPU cores) to run on
    DEVICES = ["cpu"]*5
    
    
    # RUN
    
    for i,(c,_) in enumerate(runs): print(i,c)
    print("%i runs\n"%(len(runs)))
    
    if "local" not in socket.gethostname().lower():
        jobs = [(DEVICES, c, t, i) for i,(c,t) in enumerate(runs)]
        with multiprocess.Pool(processes=len(DEVICES)) as pool:
            pool.starmap(train_models_multiprocess, jobs)
        
        