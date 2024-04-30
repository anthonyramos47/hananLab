# Import the necessary libraries
import os
import sys
from pathlib import Path
import argparse
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
from energies.Support import Supp
from energies.Sphericity import Sphericity
from energies.Proximity_Gen import Proximity
from energies.Edge_Lenght import Edge_L
from energies.QM_Fairness import QM_Fairness
from energies.Laplacian import Laplacian
from optimization.Optimizer import Optimizer




# Create the parser
parser = argparse.ArgumentParser(description="Visualizer Parser")

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


u_vals = np.linspace(0, 1, 300)
v_vals = np.linspace(0, 1, 300)

anal_l = line_congruence_uv(BSurf, rsurf, u_vals, v_vals).reshape(-1, 3)


dual_edges = np.array(list(extract_edges(dual_top)))

w_proximity = 0.01
w_fairness = 1
w_sphericity = 1/max(rads)
w_supp = 1
iter_per_opt = 1
w_lap = 0.1
state = 0
counter = 0
init_opt = False
name_saved = "Post_optimization"
def optimization():

    global w_proximity, w_fairness, w_sphericity, w_supp, iter_per_opt, init_opt, state, counter, vc, nd, f_pts, rads, dual_top, ffF, ref_V, ref_F, ref_u, ref_v, cc, vc, opt, inn_v, bd_v, adj_v, name_saved,e_f_f, e_v_v, w_lap

    # Title
    psim.TextUnformatted("Post Optimization GUI")

    if psim.Button("Stop"):
        state = 0
        

    psim.PushItemWidth(150)
    changed, iter_per_opt = psim.InputInt("Num of Iterations per Run: ", iter_per_opt)
    psim.Separator()
       
    # Inputs Opt 1
    if psim.CollapsingHeader("Optimization 1:"):
        #changed, w_proximity  = psim.InputFloat("Proximity W", w_proximity)
        #changed, w_fairness   = psim.InputFloat("Fairness W", w_fairness)
        changed, w_sphericity = psim.InputFloat("Sphericity W", w_sphericity)
        changed, w_supp       = psim.InputFloat("Support W", w_supp)
        #changed, w_lap        = psim.InputFloat("Laplacian W", w_lap)

        # State handler
        if counter%iter_per_opt == 0:
            state = 0

        if psim.Button("Init First opt"):

            # Set init to True
            init_opt = True


            opt = Optimizer()

            # Add variables to the optimizer
            opt.add_variable("c" , len(vc)*3   ) # Centers of spheres
            opt.add_variable("r" , len(rads)   ) # Radii of spheres
            

            # Initialize Optimizer ("Method", step, verbosity)
            opt.initialize_optimizer("LM", 0.25, 1)

            # Initialize variables
            opt.init_variable("c"  , cc.flatten()) 
            #opt.init_variable("nd" , nd.flatten())
            opt.init_variable("r"  , rads)

                        
            # Constraints ==========================================
            # Line congruence l.cu, l.cv = 0
            Supp_E = Supp()
            opt.add_constraint(Supp_E, args=(dual_top, l_f[inn_v]), w=w_supp)

            # Sphericity
            Sph_E = Sphericity()
            opt.add_constraint(Sph_E, args=(ffF, f_pts), w=w_sphericity)

            # # Fairness
            # Fair_M = QM_Fairness()
            # opt.add_constraint(Fair_M, args=(bd_v, adj_v, "v", 3), w=w_fairness)

            # # Proximity
            # Prox_M = Proximity()
            # opt.add_constraint(Prox_M, args=("v", ref_V, ref_F, 0.001), w=w_proximity)

            # # Edge Length
            # Edge_l = Edge_L()
            # opt.add_constraint(Edge_l, args=("c", dual_edges, 3), w=1)

            # Lap = Laplacian()
            #opt.add_constraint(Lap, args=(f_f_adj, "c", 3), w=w_lap)

            Prox_C = Proximity()
            opt.add_constraint(Prox_C, args=("c", ref_C, ref_F, 0.001), w=w_proximity)

            # Fair_C = QM_Fairness()
            # opt.add_constraint(Fair_C, args=(in_v_c, ad_v_c, "c", 3), w=0.002)

            # Define unit variables
            #opt.unitize_variable("nd", 3, 10)

            #opt.control_var("v", 0.001)
            #opt.control_var("c", 0.001)

            ps.info("Finished Initialization of Optimization 1")


        if psim.Button("Optimize 1"):
            if init_opt:
                state = 1
            else:
                ps.warning("First Optimization not initialized")
                state = 0
            
        if state:    
            counter += 1
            # Optimize
            opt.get_gradients() # Compute J and residuals
            opt.optimize() # Solve linear system and update variables
            
            # Get Line congruence
            sph_c, rf = opt.uncurry_X("c", "r")
            sph_c = sph_c.reshape(-1, 3)
            #vk = vk.reshape(-1,3)
            #nd = nd.reshape(-1,3)

            v0, v1, v2, v3 = f_pts[ffF[:, 0]], f_pts[ffF[:, 1]], f_pts[ffF[:, 2]], f_pts[ffF[:, 3]]

            dif0 = np.linalg.norm(v0 - sph_c, axis=1)**2 - rf**2
            dif1 = np.linalg.norm(v1 - sph_c, axis=1)**2 - rf**2
            dif2 = np.linalg.norm(v2 - sph_c, axis=1)**2 - rf**2
            dif3 = np.linalg.norm(v3 - sph_c, axis=1)**2 - rf**2


            planarity1 = np.zeros(len(dual_top))
            planarity  = np.zeros(len(dual_top))
            for i in range(len(dual_top)):
                if len(dual_top[i]) != 4:
                    print(dual_top[i])
                if len(dual_top[i]) == 4:
                    c0, c1, c2, c3 = sph_c[dual_top[i]]
                    planarity1[i] =  compute_volume_of_tetrahedron(c0, c1, c2, c3)
                    planarity[i]  =  compute_planarity(c0, c1, c2, c3)


            sphericity =  dif0**2 + dif1**2 + dif2**2 + dif3**2

            idx_max = np.argmax(sphericity)

            max_sph = ps.register_point_cloud("Max_Sphericity", np.array([sph_c[idx_max]]), enabled=True, transparency=0.2, color=(0.1, 0.1, 0.1))
            max_sph.set_radius(rf[idx_max], relative=False)
            #mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
            mesh = ps.register_surface_mesh("mesh", f_pts, ffF)
            mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
            mesh.add_scalar_quantity("Sphericity", np.array(sphericity), defined_on="faces", enabled=True )
            c_surf = ps.register_surface_mesh("Opt_C", sph_c, dual_top)
            c_surf.add_scalar_quantity("Planarity", planarity, defined_on='faces', enabled=True)
            c_surf.add_scalar_quantity("Face_Vol" , planarity1, defined_on='faces', enabled=True)
            #c_surf.add_vector_quantity("Normals", nd, defined_on='faces', enabled=True)
            ps.register_curve_network("Edges", sph_c, dual_edges)
            ps.register_point_cloud("cc", sph_c)

    
    psim.Separator()


    if psim.Button("Report"):
        opt.get_energy_per_constraint()

    psim.Separator()

    if psim.Button("Draw Support Structure"):

        # Get results
        sph_c, rf = opt.uncurry_X("c", "r")
        sph_c = sph_c.reshape(-1, 3)
        vk = f_pts

        # Get vertices per edge
        vi, vj = vk[e_v_v[0]], vk[e_v_v[1]]



        # Get sphere indices per edge
        c_i, c_j = sph_c[e_f_f[0]], sph_c[e_f_f[1]]

        dir = c_j - c_i
        d = np.linalg.norm(dir, axis=1)
        dir /= d[:, None]

        ri, rj = rf[e_f_f[0]], rf[e_f_f[1]]

        mc = c_i + ((ri**2 - rj**2 + d**2)/(2*d))[:, None]*dir


        # Direction from center to vertices
        dvimc = vi - mc
        dvimc /= np.linalg.norm(dvimc, axis=1)[:, None]
        # Project dvimc to the plane
        dvimc = dvimc - np.sum(dvimc*dir, axis=1)[:, None]*dir
        dvimc /= np.linalg.norm(dvimc, axis=1)[:, None]

        # Direction from center to vertices
        dvjmc = vj - mc
        dvjmc /= np.linalg.norm(dvjmc, axis=1)[:, None]
        # Project dvjmc to the plane
        dvjmc = dvjmc - np.sum(dvjmc*dir, axis=1)[:, None]*dir
        dvjmc /= np.linalg.norm(dvjmc, axis=1)[:, None]

        # Define basis for the plane
        b1 = dvimc
        b2 = np.cross(dir, dvimc)

        # Compute angle between vectors
        a = np.arccos(np.sum(dvimc*dvjmc, axis=1))

        # radius of intersection
        ri = np.sqrt(ri**2 - ((ri**2 - rj**2 + d**2)/(2*d))**2)

        # Delta
        delta = 0.01*np.pi/180

        rin = ri - 0.01
        rot = ri + 0.01

        # Define points to draw plane
        sp0 = (rin*np.cos(0))[:, None] * b1 + (rin*np.sin(0))[:, None]*b2 + mc
        sp1 = (rin*np.cos(np.pi/2))[:, None] * b1 + (rin*np.sin(np.pi/2))[:, None]*b2 + mc
        sp2 = (rot*np.cos(np.pi/2))[:, None] * b1 + (rot*np.sin(np.pi/2))[:, None]*b2 + mc
        sp3 = (rot*np.cos(0))[:, None] * b1 + (rot*np.sin(0))[:, None]*b2 + mc


        # # Dir vector between spheres
        # dij  = c_j - c_i
        # dij /= np.linalg.norm(dij, axis=1)[:, None]

        # # Point of bisecting plane between spheres
        # mc =  c_i +  np.einsum('ij,ij->i', (vi - c_i), dij)[:, None]*dij

        # v1mc = vi - mc
        # v2mc = vj - mc

        supp_obj = open(os.path.join(remeshing_dir, name+'_supp_opt.obj'), 'w')

        for i in range(len(mc)):
            verts = np.array([sp0[i], sp1[i], sp2[i], sp3[i]])
            #draw_polygon(verts, (0.1, 0, 0.2), name="plane"+str(i))

            
            supp_obj.write(f"v {verts[0][0]} {verts[0][1]} {verts[0][2]}\n")
            supp_obj.write(f"v {verts[1][0]} {verts[1][1]} {verts[1][2]}\n")
            supp_obj.write(f"v {verts[2][0]} {verts[2][1]} {verts[2][2]}\n")
            supp_obj.write(f"v {verts[3][0]} {verts[3][1]} {verts[3][2]}\n")

            supp_obj.write(f"f {4*i + 1} {4*i + 2} {4*i + 3} {4*i + 4} \n")

        supp_obj.close()
        
        mesh = ps.register_surface_mesh("Opt_Vertices", vk, ffF)
        mesh.add_scalar_quantity("Radii", rf, defined_on='faces', enabled=True)
        ps.register_surface_mesh("Opt_C", sph_c, dual_top)

    psim.Separator()
    psim.TextUnformatted("Save Results")
    if psim.Button("Save"):

        # Get results
        sph_c, rf  = opt.uncurry_X('c', 'r')
        
        sph_c = sph_c.reshape(-1, 3)

        # save mesh
        #write_obj(os.path.join(remeshing_dir, name+'_mesh_opt.obj'), vk, ffF)

        # Save spheres ======================================================================
        # Open file
        exp_file = open(os.path.join(remeshing_dir, name+'_sphs_opt.dat'), 'w')

        # Write inn_f in the first line
        for i in inn_f:
            exp_file.write(f"{i} ")
        exp_file.write("\n")

        for i in range(len(ffF)):

            exp_file.write(f"{sph_c[i][0]} {sph_c[i][1]} {sph_c[i][2]} {rf[i]} {vc[i][0]} {vc[i][1]} {vc[i][2]} ")

            for j in f_f_adj[i]:
                exp_file.write(f"{j} ")

            exp_file.write("\n")

        exp_file.close()


#print("Mesh", ffV[:10])
         
ps.init()

mesh = ps.register_surface_mesh("mesh", f_pts, ffF)
mesh.add_vector_quantity("l", l_f, defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.1, 0.0, 0.0))
ref_M = ps.register_surface_mesh("S_uv", ref_V, ref_F)
#ref_M.add_vector_quantity("A l", anal_l, defined_on="vertices", vectortype='ambient',  enabled=True, color=(0.0, 0.1, 0.0))
ps.register_surface_mesh("Remeshed", ffV, ffF)
ps.register_curve_network("Edges", cc, dual_edges)
ps.register_point_cloud("cc", cc)
#ps.register_point_cloud("Boundary", f_pts[bd_v], enabled=True, color=(0.1, 0.1, 0.1))
#ps.register_surface_mesh("C_uv", ref_C, ref_F)
ps.register_surface_mesh("Mid_mesh", VR, ffF)


ps.set_user_callback(optimization)
ps.show()



