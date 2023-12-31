# Optimizer class
import numpy as np
import time as tm
import pandas as pd
from hanan.optimization.Unit import Unit
from scipy.sparse import csc_matrix,diags, vstack
from scipy.sparse.linalg import splu, spsolve


class Optimizer():
    def __init__(self) -> None:
        """
            The optimizer is a class that allows to solve non-linear least squares problems
            using two different methods: Levenberg-Marquardt and Projected Gauss-Newton, of the form
                J^TJ dx = -J^Tr (LM) or J^TJ (X + l I) = J^T.X - r dx (PG)

            The optimizer is initialized with the variables to optimize (X) and the step size.

            The optimizer is then updated with constraints. Each constraint is a class that inherits from the Constraint class.
            The constraints are added to the optimizer using the add_constraint(constraint, *args) method. 
            The arguments of the constraint are passed as *args.

            The optimizer is then solved using the optimize(name_solv) method. The name_solv parameter is a string that can be
            either "LM" for Levenberg-Marquardt or "PG" for Projected Gauss-Newton.
        """
        self.J = None # Jacobian matrix
        self.r = None # Residual vector
        self.X = None # Variable
        self.bestX = None # Best variable
        self.bestit = None # Best iteration
        self.prevdx = None # Previous dx
        self.H = None # Hessian matrix
        self.it = None # Iteration 
        self.step = None # Step size
        self.method = None # Method used to solve the problem
        self.energy = [] # Energy vector
        self.var_idx = None # Variable indices

    def unitize_variable(self, var_name, dim) -> None:
        """
            Method to add a unit constraint to the optimizer. The unit constraint is a constraint that forces the variable to be unit.
            Input:
                var_name: Name of the variable
                var_indices: Indices of the variable
                dim: Dimension of the variable
        """
        # Initialize constraint
        unit = Unit()
        unit.initialize_constraint(self.X, self.var_idx, var_name, dim)

        # Add constraint
        self.get_gradients(unit)

    def fix_vertices(self, fix_vertices) -> None:
        """
            **-- Warning: Method not ready yet --**
            Method to fix vertices of the mesh. Not used in the current implementation
        """

        # The structure of our Jacobian is J = [ V | aux variables]
        # If we want to fix a vertex, we need to set the corresponding column of the Jacobian to zero
        # self.J[:, fix_vertices*3    ] = 0
        # self.J[:, fix_vertices*3 + 1] = 0
        # self.J[:, fix_vertices*3 + 2] = 0
        pass



    def initialize_optimizer(self, X, var_dic, method= "LM", step = 0.8) -> None:
        """
        Initialize the optimizer( variables, step size)
        """
        # Initialize variables
        self.X = X
        self.var_idx = var_dic
        self.X0 = X.copy()
        self.it = 0
        self.step = step
        self.method = method

    def get_gradients(self, constraint) -> None:
        """ Add constraint to the optimizer
            Input:
                constraint: Constraint class
                args: arguments of the constraint
        """
        
        # Add J, r to the optimizer
        if constraint.w != 0:
            
            # Compute J, r for the constraint
            constraint._compute(self.X)

            # Add J, r to the optimizer
            if self.J is None:
                self.J =  np.sqrt(constraint.w) * constraint.J
                self.r =  np.sqrt(constraint.w) * constraint.r
            else:
                self.J = vstack((self.J, np.sqrt(constraint.w) * constraint.J))
                self.r = np.concatenate((self.r, np.sqrt(constraint.w) * constraint.r))
        

 
    def optimize(self):
        
        if self.prevdx is None or self.prevdx > 1e-8:
            if self.method == 'LM': # Levenberg-Marquardt
                self.LM()
            elif self.method == 'PG': # Projected Gauss-Newton
                self.PG()
            else:
                print("Error: Solver not implemented or not specified")
            

    def LM(self):
        # Levenberg-Marquardt method for non-linear least squares
        # https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm
        # Solve for (J^TJ + lambda*I) dx = -J^Tr, lambda = max(diag(J^TJ))*1e-8

        # Get J 
        J = self.J

        # Compute pseudo Hessian
        H = (J.T * J).tocsc()
        
        # Calculate the value to add to the diagonal
        add_value = H.max() * 1e-8

        # Create a diagonal matrix with the values to add
        diagonal_values = np.array([add_value] * H.shape[0])
        diagonal_matrix = diags(diagonal_values, 0, format='csc')

        # Add the diagonal_matrix to H
        H = H + diagonal_matrix
       
        b = -J.T@self.r

        dx = spsolve(H, b)

        # Store previous dx norm
        self.prevdx = np.linalg.norm(dx)

        # Compute energy
        energy = self.r.T@self.r

        # Append energy
        self.energy.append(energy)

        # Update variables
        self.update_variables(dx)

        # Store best X and iteration
        if self.it == 0:
            self.bestX = self.X
            self.bestit = self.it
        else:
            if energy < self.energy[self.bestit]:
                self.bestX = self.X
                self.bestit = self.it
        
        # Update iteration
        self.it +=1
        
        # Print energy
        print(f" E {self.it}: {energy}\t dx: {self.prevdx}")

        # Clear constraints
        self.clear_constraints()

    def get_variables(self):
        # Return variables
        print(f"Best iteration: {self.bestit + 1}\t Best energy: {self.energy[self.bestit]}")
        return self.bestX
        
    def PG(self):
        # To be implemented
        pass 
    
    def log_simple(self, filename) -> None:
        # Create a log file with the total energy of the system

        # Open file
        log_file = open(filename, "w")

        # Write energy
        for i in range(len(self.energy)):
            log_file.write(f"{i} {self.energy[i]}\n")
        
        log_file.close()


    def print_log(self) -> None:

        # Print energy
        df = pd.DataFrame(self.energy)

        df.columns = ["Energy"]
        
        print(df)
    
    def update_variables(self, arg) -> None:
        # Update variables

        if self.it%10 ==0:
            self.step *= 0.9
        
        if self.method == "LM":
            self.X += self.step*arg
        elif self.method == "PG":
            pass
        
    def get_variables(self) -> np.array:
        # Return best variables
        print(f"Best iteration: {self.bestit + 1}\t Best energy: {self.energy[self.bestit]}")
        return self.bestX

    def clear_constraints(self):
        # Clear Jacobian and residual
        self.J = None
        self.r = None

    def reset(self):
        """ Function that resets the optimizer to the initial state.
        """

        self.X = self.X0.copy()
        self.prevdx = None
        self.energy = []
        self.it = 0
        self.bestX = None
        self.bestit = None
        self.clear_constraints()



