# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse
import time
from pickle import load
import subprocess

# Obtain the path HananLab; this is the parent directory of the hananLab/hanan directory
# <Here you can manually add the path direction to the hananLab/hanan directory>
# Linux 
path = os.getenv('HANANLAB_PATH')
if not path:
    raise EnvironmentError("HANANLAB_PATH environment variable not set")
sys.path.append(path)

# Verification of the path
print(path)


# Import the necessary libraries for visualization and computation
import igl
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
import matplotlib.pyplot as plt
import splipy as sp
import json
import pickle



# Import the necessary classes and functions from the hananLab/hanan directory

# Geometry classes
from geometry.mesh import Mesh
from geometry.utils import *

# Local files
from utils.bsplines_functions import *
from utils.visualization import *

# Optimization classes
#from energies.Support import Supp
from energies.Sphere import Sphere
from energies.QM_Fairness import QM_Fairness
from energies.Proximity_Gen import Proximity
from energies.Proximity_C import Proximity_C
from energies.Support import Supp
from energies.Support_Fix import Supp_F
from energies.Sphere_Fix_V import Sphere_Fix
from energies.Sph_unit import Sph_Unit
from energies.Tor_Planarity import Tor_Planarity
from energies.Reg_E import Reg_E

from optimization.Optimizer import Optimizer




# Create the parser
parser = argparse.ArgumentParser(description="Post Optimization")

# Add an argument
parser.add_argument('file_name', type=str, help='File name to load')

# Parse the command line arguments
name = parser.parse_args().file_name
pickle_name = name+'.pickle'
save = 1

# Working directory
working_path = os.getcwd()

# Experiments folder
exp_dir = os.path.join(working_path, 'experiments')

# Remeshing data folder
remeshing_dir = os.path.join(working_path, 'data', 'Remeshing', name)

# Reports folder
reports_dir = os.path.join(working_path, "data", "Reports")


# Load picke information
def load_data():
    """ Function to load the data from a pickle file
    """
    with open(os.path.join(exp_dir, pickle_name), 'rb') as f:
        data = load(f)
    return data

data = load_data()

if "projected" not in data.keys():
    print("Projecting the mesh")
    # Replace 'script_to_run.py' with the path to your Python script
    script_path = 'foot_point_bspline.py '+name+' 0'
    # Run the script using subprocess
    subprocess.run(['python', script_path])



# Get u and v points 
u_pts = data['u_pts']
v_pts = data['v_pts']

# Get the B-spline and its derivatives
BSurf = data['surf']
rsurf = data['r_uv']

# Get information from the pickle file
f_f_adj = data['f_f_adj']
e_f_f = data['e_f_f']
e_v_v = data['e_v_v']
adj_v = data['adj_v']
inn_f = data['inn_f']
inn_v = data['inn_v']
bd_v = data['bd_v']
cc = data['cc']
vc = data['vc']
nd = data['nd']
f_pts = data['f_pts']
rads = data['rads']
dual_top = data['dual_top']
ffF = data['ffF']
ffV = data['ffV']
ref_V = data['ref_V']
ref_F = data['ref_F']
ref_C = data['ref_C']
VR = data['VR']
l_f = data['l_dir']

A, B, C = get_Implicit_sphere(cc, rads)
dual_edges = np.array(list(extract_edges(dual_top)))
mesh_edges = e_v_v
tuples_edges = np.vstack((mesh_edges[0], mesh_edges[1])).T

w_proximity = 0.2
w_proximity_c = 0.1
w_fairness = 0.2
w_sphericity = 1
w_supp = 0.5
w_lap = 0.1
iter_per_opt = 1
state = 0
state2 = 0
counter = 0
init_opt = False
w_tor = 0.01
idx_sph = 0
times = []
step = 0.5
name_saved = "Post_optimization"
def optimization():

    global w_proximity, w_fairness, w_sphericity, w_supp, iter_per_opt, init_opt, state, counter, vc, nd, f_pts, rads, dual_top, ffF, ref_V, ref_F, ref_u, ref_v, cc, vc, opt, inn_v, bd_v, adj_v, name_saved, e_f_f, e_v_v, w_lap, state2, init_opt_2, A, B, C, w_proximity_c, idx_sph, w_tor, step, l_f, counter2, time, opt

    # Title
    psim.TextUnformatted("Post Optimization GUI")

    if psim.Button("Stop"):
        state = 0
        state2 = 0
        

    psim.PushItemWidth(150)
    changed, iter_per_opt = psim.InputInt("Num of Iterations per Run: ", iter_per_opt)
    changed, step = psim.InputFloat("Setp per Optimization: ", step)
    
    psim.Separator()

        # State handler
    if counter%iter_per_opt == 0 :
        state = 0

       
    # Inputs Opt 1
    if psim.CollapsingHeader("Optimization 1:"):
                
        changed, w_proximity  = psim.InputFloat("Proximity M", w_proximity)
        changed, w_proximity_c  = psim.InputFloat("Proximity C", w_proximity_c)
        changed, w_fairness   = psim.InputFloat("Fairness W", w_fairness)
        changed, w_sphericity = psim.InputFloat("Sphericity W", w_sphericity)
        changed, w_supp       = psim.InputFloat("Support W", w_supp)
        changed, w_tor        = psim.InputFloat("Tor Planarity W", w_tor)
        


        if psim.Button("Init First opt"):

            # Initialize Optimizer
            init_opt = True

            for _ in range(30):
                # Get gradients
                opt.get_gradients()
                opt.optimize()

            opt.get_energy_per_constraint()

            A, B, C = opt.uncurry_X("A", "B", "C")
        
            opt = Optimizer()

            # Add variables to the optimizer
            opt.add_variable("A" , len(vc)  ) # Centers of spheres
            opt.add_variable("B" , len(vc)*3  ) # Centers of spheres
            opt.add_variable("C" , len(vc)  ) # Centers of spheres

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", step, 0)

            # Initialize variables
            opt.init_variable("A"  , A) 
            opt.init_variable("B"  , B.flatten())
            opt.init_variable("C"  , C)

            opt.add_variable("nd", len(l_f)*3   ) # Normals of dual faces
            opt.add_variable("v" , len(f_pts)*3) # Vertices of mid mesh
            opt.add_variable("n_l" , len(mesh_edges[0])*3   ) # Normal or the supp planes
            #opt.add_variable("r" , len(rads)   ) # Radii of spheres
            #opt.add_variable("mu", len(dual_edges))

            opt.fix_variable("v", bd_v)

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", step, 1)

            # Initialize variables
            opt.init_variable("A"  , A) 
            opt.init_variable("B"  , B)
            opt.init_variable("C"  , C)
            opt.init_variable("v"  , f_pts.flatten())
            opt.init_variable("nd" , l_f.flatten())
            
                        
            # Constraints ==========================================
            # # Line congruence l.cu, l.cv = 0
            Supp_E = Supp()
            opt.add_constraint(Supp_E, args=(dual_top, inn_v), w=w_supp, ce=1)

            # Sphericity
            Sph_E = Sphere()
            opt.add_constraint(Sph_E, args=(ffF, 2), w=w_sphericity, ce=1)

            Sph_U = Sph_Unit()
            opt.add_constraint(Sph_U, args=(), w=10)

            # # Fairness
            Fair_M = QM_Fairness()
            opt.add_constraint(Fair_M, args=(adj_v, "v", 3), w=w_fairness)

            # Torosal Planarity
            Tor_P = Tor_Planarity()
            opt.add_constraint(Tor_P, args=(mesh_edges, inn_v), w=w_tor, ce=1)

            # # Proximity
            Prox_M = Proximity()
            opt.add_constraint(Prox_M, args=("v", ref_V, ref_F, 0.0001), w=w_proximity, ce =1)

            Prox_C = Proximity_C()
            opt.add_constraint(Prox_C, args=(ref_C, ref_F, 0.0001), w=w_proximity_c, ce =1)

            # # Define unit variables
            opt.unitize_variable("nd", 3, 10)
            opt.unitize_variable("n_l", 3, 10)

            opt.control_var("v" , 0.4)
            opt.control_var("nd", 0.5)
            #opt.control_var("c", 0.0001)

    if psim.Button("Optimize 1"):
        if init_opt:
            state = 1
        else:
            ps.warning("First Optimization not initialized")
            state = 0

    if state and not opt.stop:    
        counter += 1
        i_t = time.time()
        opt.get_gradients() # Compute J and residuals
        opt.optimize_step() # Solve linear system and update variables
        f_t = time.time()
        times.append(f_t - i_t)

        opt.stop_criteria()
        
        # Get Variables
        vk, oA, oB, oC, nd = opt.uncurry_X("v", "A", "B", "C", "nd")
        
        vk = vk.reshape(-1,3)
        oB = oB.reshape(-1,3)
        nd = nd.reshape(-1,3)

        vc = np.sum(vk[ffF], axis=1)/4

        sph_c, rf = Implicit_to_CR(oA,oB,oC)

        v0, v1, v2, v3 = vk[ffF[:, 0]], vk[ffF[:, 1]], vk[ffF[:, 2]], vk[ffF[:, 3]]

        dif0 = np.linalg.norm(v0 - sph_c, axis=1)**2 - rf**2
        dif1 = np.linalg.norm(v1 - sph_c, axis=1)**2 - rf**2
        dif2 = np.linalg.norm(v2 - sph_c, axis=1)**2 - rf**2
        dif3 = np.linalg.norm(v3 - sph_c, axis=1)**2 - rf**2


        planarity1 = np.zeros(len(dual_top))
        planarity  = np.zeros(len(dual_top))
        for i in range(len(dual_top)):

            if len(dual_top[i]) == 4:
                c0, c1, c2, c3 = sph_c[dual_top[i]]
                planarity1[i] =  compute_volume_of_tetrahedron(c0, c1, c2, c3)
                planarity[i]  =  compute_planarity(c0, c1, c2, c3)


        sphericity =  dif0**2 + dif1**2 + dif2**2 + dif3**2

    
        mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        mesh.add_scalar_quantity("Sphericity", np.array(sphericity), defined_on="faces", enabled=True )
        mesh.add_vector_quantity("lc", nd, enabled=True, length=0.12)
        c_surf = ps.register_surface_mesh("Opt_C", sph_c, dual_top)
        c_surf.add_scalar_quantity("Planarity", planarity, defined_on='faces', enabled=True)
        # c_surf.add_scalar_quantity("Face_Vol" , planarity1, defined_on='faces', enabled=True)
        c_surf.add_vector_quantity("Normals", nd[inn_v], defined_on='faces', enabled=True)
        ps.register_curve_network("Edges", sph_c, dual_edges, radius=0.001)
        ps.register_point_cloud("cc", sph_c, radius=0.002)

        centers = ps.register_point_cloud("cc", sph_c[465:468])

        centers.add_vector_quantity("Dir", 1.2*(vc[465:468] - sph_c[465:468]), enabled=True, vectortype="ambient")
    
    elif opt.stop:
        ps.warning("Optimization 1 has stopped")
        state = 0



    if psim.Button("Time"):
        print("Time: ", np.mean(times))
    psim.Separator()

    if psim.Button("Report"):
        #opt.get_energy_per_constraint()
        opt.get_norm_energy_per_constraint()

    psim.SameLine()    
    
    if psim.Button("Save Report"):

        report_data = {
            "Vertices": len(ffV),
            "Faces": len(ffF),
            "W_proximity": w_proximity,
            "W_proximity_c": w_proximity_c,
            "W_fairness": w_fairness,
            "W_sphere": w_sphericity,
            "W_supp": w_supp,
            "W_tor": w_tor,
            "Iterations": opt.it,
            "Time": np.mean(times),
            "Energy": sum(e_i for e_i in opt.norm_energy_dic.values())
        }
        
        # Open file to write
        file_path = os.path.join(reports_dir, name + "_Final_Report.json")

        # Save the variable to a json
        with open(file_path, 'w') as file:
            json.dump(report_data, file)

    psim.Separator()

    psim.TextUnformatted("Visualization Sphere")

    change, idx_sph = psim.InputInt("Index", idx_sph)

    if psim.Button("Draw Sphere"):

        # Get Variables
        oA, oB, oC = opt.uncurry_X( "A", "B", "C")
        
        oB = oB.reshape(-1,3)
        
        sph_c, rf = Implicit_to_CR(oA,oB,oC)


        sph = ps.register_point_cloud("s_"+str(idx_sph), np.array([sph_c[idx_sph]]), enabled=True, transparency=0.3, color=(1, 0, 0))
        sph.set_radius(rf[idx_sph], relative=False)
    
    
    if psim.Button("Draw Support"):

        # # Get results
        vk, n_l = opt.uncurry_X("v",  "n_l")

        n_l = n_l.reshape(-1,3)
        vk  = vk.reshape(-1,3)


        # # Get vertices per edge
        vi, vj = vk[mesh_edges[0]], vk[mesh_edges[1]]


        vij = vj - vi
        dist = np.linalg.norm(vij, axis=1)

        vij = vij - np.einsum('ij,ij->i', vij, n_l)[:, None]*n_l
        b1 = vij/np.linalg.norm(vij, axis=1)[:, None]

        print(b1.shape, n_l.shape)

        # Compute orthogonal vector
        b2 = np.cross(n_l, b1)

        supp_obj = open(os.path.join(remeshing_dir, name+'_supp_opt.obj'), 'w')


        # delta value of widht for plane
        delta = 0.01


        for i in range(len(mesh_edges[0])):
            
            p0 = vi[i] - delta*b1[i] - delta*b2[i]
            p1 = vi[i] + (dist[i] + delta)*b1[i] - delta*b2[i]
            p2 = vi[i] + (dist[i] + delta)*b1[i] + (dist[i] + delta)*b2[i]
            p3 = vi[i] - delta*b1[i] + (dist[i] + delta)*b2[i]
            
            verts = np.array([p0,p1,p2,p3])
            #draw_polygon(verts, (0.1, 0, 0.2), name="plane"+str(i))

            supp_obj.write(f"v {p0[0]} {p0[1]} {p0[2]}\n")
            supp_obj.write(f"v {p1[0]} {p1[1]} {p1[2]}\n")
            supp_obj.write(f"v {p2[0]} {p2[1]} {p2[2]}\n")
            supp_obj.write(f"v {p3[0]} {p3[1]} {p3[2]}\n")
            supp_obj.write(f"f {4*i + 1} {4*i + 2} {4*i + 3} {4*i + 4} \n")

        supp_obj.close()
        
        # mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        # mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        # ps.register_surface_mesh("Opt_C", sph_c, dual_top)
    psim.Separator()

    psim.TextUnformatted("Save Results")
    if psim.Button("Save"):

        # Get results
                  
        # Get Variables
        vk, oA, oB, oC, nl, nd = opt.uncurry_X("v", "A", "B", "C", "n_l", "nd")


        vk = vk.reshape(-1,3)
        oB = oB.reshape(-1,3)
        nl = nl.reshape(-1,3)
        nd = nd.reshape(-1,3)

        # Save opt into pickle
        data['vk'] = vk
        data['oA'] = oA
        data['oB'] = oB
        data['oC'] = oC
        data['nl'] = nl
        data['nd'] = nd
        with open(os.path.join(exp_dir, pickle_name), 'wb') as f:
            pickle.dump(data, f)

        # Draw the spheres

        sph_c, rf = Implicit_to_CR(oA,oB,oC)

        v0, v1, v2, v3 = vk[ffF[:, 0]], vk[ffF[:, 1]], vk[ffF[:, 2]], vk[ffF[:, 3]]

        # save mesh
        write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

        # Save spheres ======================================================================
        # Open file
        sph_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')
        sph_cut_file = open(os.path.join(remeshing_dir, name+'_sphs_p_cut.dat'), 'w')
        annuli_file = open(os.path.join(remeshing_dir, name+'_annuli.dat'), 'w')

        # Write inn_f in the first line
        for i in inn_f:
            sph_file.write(f"{i} ")
        sph_file.write("\n")

        # Get list of faces of each edge
        edges_faces = np.c_[e_f_f].flatten()
        for i in edges_faces:
            annuli_file.write(f"{i} ")
        annuli_file.write("\n")
    
        for i in range(len(e_f_f[0])):

            # Get the normal of the edge
            n = nl[i]

            # Get the vertices of the edge
            vi, vj = vk[e_v_v[0][i]], vk[e_v_v[1][i]]

            # Compute the mid point of the edge
            mid = (vi + vj)/2
            annuli_file.write(f"{mid[0]} {mid[1]} {mid[2]} {n[0]} {n[1]} {n[2]} {vi[0]} {vi[1]} {vi[2]} {vj[0]} {vj[1]} {vj[2]}\n")


        for i in range(len(ffF)):
            
            sph_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

            string_data_sph_panel = f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} "

            annuli_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]}\n")

            for j in range(len(ffF[i])):

                #edge
                e = [ffF[i][j], ffF[i][(j+1) % len(ffF[i])]]

                # find index of edge
                idx = search_edge(tuples_edges, e)

                # Get the normal of the edge
                n = nl[idx]

                # Get the vertices of the edge
                vi, vj = vk[e[0]], vk[e[1]]

                # Compute the mid point of the edge
                mid = (vi + vj)/2
                string_data_sph_panel += f"{vi[0]} {vi[1]} {vi[2]} {mid[0]} {mid[1]} {mid[2]} {n[0]} {n[1]} {n[2]} "
        
            sph_cut_file.write(string_data_sph_panel + "\n")
            

            for j in f_f_adj[i]:
                sph_file.write(f"{j} ")

            sph_file.write("\n")

            
        annuli_file.close()
        sph_cut_file.close()
        sph_file.close()


         
        # Save Optimized line congruence
        Node_path = os.path.join(remeshing_dir, name+'_opt_NodesAxis.obj')

        # Compute line congruence
        v_lc = np.vstack((vk - 0.3*nd, vk + 0.3*nd))

        f_lc = np.array([[i, i+len(vk)] for i in range(len(vk))])
        # Write info
        with open(Node_path, 'w') as f:
            for v in v_lc:
                f.write('v {} {} {}\n'.format(v[0], v[1], v[2]))
            for l in f_lc:
                f.write('l {} {}\n'.format(l[0]+1, l[1]+1))


#print("Mesh", ffV[:10])
opt = Optimizer()

# Add variables to the optimizer
opt.add_variable("A" , len(vc)  ) # Centers of spheres
opt.add_variable("B" , len(vc)*3  ) # Centers of spheres
opt.add_variable("C" , len(vc)  ) # Centers of spheres

# Initialize Optimizer ("Method", step, verbosity)
opt.initialize_optimizer("LM", step, 1)

# Initialize variables
opt.init_variable("A"  , A) 
opt.init_variable("B"  , B.flatten())
opt.init_variable("C"  , C)

# # Line congruence l.cu, l.cv = 0
Supp_E = Supp_F()
opt.add_constraint(Supp_E, args=(dual_top, l_f), w=0.001, ce=1)

# Sphericity
Sph_E = Sphere_Fix()
opt.add_constraint(Sph_E, args=(ffF, f_pts), w=10, ce=1)

Sph_U = Sph_Unit()
opt.add_constraint(Sph_U, args=(), w=10, ce=1)

Prox_C = Proximity_C()
opt.add_constraint(Prox_C, args=(ref_C, ref_F, 0.0001), w=0.1)
         
ps.init()

mesh = ps.register_surface_mesh("mesh", f_pts, ffF)
#mesh.add_vector_quantity("lc", l_f, enabled=True)
#ps.register_surface_mesh("S_uv", ref_V, ref_F)
#ps.register_surface_mesh("Remeshed", ffV, ffF)
#ps.register_surface_mesh("Dual mesh", vc, dual_top)
#centers = ps.register_point_cloud("cc", cc[465:468])

#centers.add_vector_quantity("Dir", 1.2*(vc[465:468] - cc[465:468]), enabled=True, vectortype="ambient")


#ps.register_point_cloud("Boundary", f_pts[bd_v], enabled=True, color=(0.1, 0.1, 0.1))
#ps.register_surface_mesh("C_uv", ref_C, ref_F, enabled=False)
#ps.register_surface_mesh("Mid_mesh", VR, ffF)


ps.set_user_callback(optimization)
ps.show()



