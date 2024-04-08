# -*- coding: utf-8 -*-


# !/usr/bin/env python

# -*- coding: utf-8 -*-

from __future__ import absolute_import

from __future__ import print_function

from __future__ import division

import numpy as np

from scipy import sparse

import scipy
import pickle

import argparse
import json

# -----------------------------------------------------------------------------

from pathlib import Path
import sys
import os 

# Obtain path
path = os.getenv('HANANLAB_PATH')
if not path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(path)

import geolab as geo
import geopt

# -----------------------------------------------------------------------------

from orthogonal_mapper import OrthoOptimizer

# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------


if __name__ == '__main__':

    # -------------------------------------------------------------------------
    # Open the file containing the pickled object. The 'rb' parameter denotes 'read binary'
    # Open a file for writing. The 'wb' parameter denotes 'write binary'
   

    # Crear el analizador de argumentos
    parser = argparse.ArgumentParser(description='Get pickle data of experiment.')
    # Añadir un argumento posicional
    parser.add_argument('pkl_name'  , type=str, help='Pickle file diretion')
    parser.add_argument('w_fairnes' , type=str, help='Weight of the fairness energy')
    parser.add_argument('w_closenes', type=str, help='Weight of the closeness energy')
    parser.add_argument('it', type=str, help='Number of iterations')


    
    # Analizar los argumentos de la línea de comandos
    args = parser.parse_args()
    
    
    with open(args.pkl_name, 'rb') as file:
        data  = pickle.load(file)

    V = data["V"]
    F = data["F"]
    t1 = data["t1"]
    t2 = data["t2"]

    # print("args.w_fairness: ", args)
    # print("args.w_closeness: ", args.w_closeness)

    # Scale factor
    v0, v1, v2, v3 = V[F[:,0]], V[F[:,1]], V[F[:,2]], V[F[:,3]]

    du = v1 - v0
    dv = v3 - v0

    # Get mean diagonals
    mean_diag =  np.min(np.vstack([du, dv]).mean(axis=1) )

    scale = 0.05


    w_f, w_c= float(args.w_fairnes), float(args.w_closenes)
    name = args.pkl_name.split('/')[-1].split('.')[0]

    save = 1

    

    # per face directions
    # D1 = np.loadtxt(dir_data+'tor_D1.dat', delimiter=',', dtype=np.float64)

    # D2 = np.loadtxt(dir_data+'tor_D2.dat', delimiter=',', dtype=np.float64)
    D1 = t1 
    D2 = t2

    # -------------------------------------------------------------------------
    plotter = geo.plotter()

    H = geo.halfedges(F)
    B = geo.faces_centroid(V, H)

    plotter.plot_faces(V, H, opacity=0.8, color='white')
    plotter.plot_edges(V, H, width=2, color='black')
    plotter.plot_vectors(D1, anchor=B, position='center', scale_factor=scale, color='black')
    plotter.plot_vectors(D2, anchor=B, position='center', scale_factor=scale, color='black')

    opt = OrthoOptimizer(V, H, D1, D2)
    #opt.iterations = int(args.it)
    opt.iterations = 300
    opt.step = 0.20
    opt.set_weight('mesh_fairness', w_f)
    opt.set_weight('closeness', w_c)
    opt.optimize()

    if save:
        opt.save_mesh('{}_deformed'.format(name), field_scale=1, save_all=True)
        geo.save_mesh_obj(V, F, '{}_start'.format(name), overwrite=True)

    v1, v2 = opt.vectors()

    V = opt.optimized_vertices()
    #V += np.array([0,0,1])

    B1 = geo.faces_centroid(V, H)
    

    plotter.plot_faces(V, H, color='b')
    plotter.plot_edges(V, H, color='white', width=4)
    plotter.plot_vectors(v1, anchor=B1, position='center', scale_factor=scale)
    plotter.plot_vectors(v2, anchor=B1, position='center', scale_factor=scale)
    plotter.show()
