# Planarity constraint implementation
import numpy as np
from hanan.optimization.constraint import Constraint
from hanan.geometry.utils import vec_dot


class Center_abs(Constraint):

    def __init__(self) -> None:
        """ Sphericity constraint
        Energy that penalizes the deviation of both envelopes from a sphere
        Energy : \sum_{f \in F} || \sum_{v \in V} [(c_f - v_i)^2 - r_f^2]  \sum_{j \in U} [(c_f - v_j)^2 - r_f^2] ||^2
        where   c_f is the center of the sphere, 
                r_f is the radius of the sphere, 
                V is the set of vertices of the face f, 
                U is the set of vertices of the second envelope face f
        """
        super().__init__()
        self.name = "Absolute Center" # Name of the constraint
        self.normals = None
        self.lengths = None
        self.weights = None
        self.vertices = None

    def initialize_constraint(self, X, var_idx, v, n, l, w) -> None:
        # Input
        # X : Variables
        # F: List of faces
        # V: List of vertices

        # Set if initalization or not
        self.normals = n
        self.lengths = l
        self.weights = w
        self.vertices = v
        # Add energy
        self.add_constraint("E1", len(n))

        


    def compute(self, X, var_idx) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
        """

        # Get variables
        m = self.uncurry_X(X, var_idx, "m")      

        m = np.array(m)

        # Get lenghts
        p_lengths = self.lengths
        # Get weights
        w = self.weights
        # Get normals
        n = self.normals

        # Get vertices
        v = self.vertices

        # dp E1 =>
        self.add_derivatives(self.const_idx["E1"].repeat(3), np.tile(var_idx["m"], len(v)), (-(vec_dot(v - m, n))[:, None] * (-2 * (vec_dot(v - m, v - m))[:, None] * n + 2 * (vec_dot(v-m, n))[:,None] * (v - m))/ ((vec_dot(v - m, v - m) ** 2)[:, None])).flatten())  
        
        # Set r
        self.set_r(self.const_idx["E1"], 1 - (vec_dot(n, v- m ) * vec_dot(n, v - m)) / vec_dot(v - m, v - m))

        #angle = np.arccos(vec_dot(n, v - m)/p_lengths)

        #self.weights= np.tan(angle)

        #self.lengths = np.linalg.norm( v-m, axis = 1 )