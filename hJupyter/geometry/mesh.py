"""
    Mesh Data Structure
    ===================
    This module contains the mesh data structure and its associated methods.
    The mesh data structure is a collection of vertices, edges, and faces.
    The mesh data structure is used to represent the geometry of a 3D object.

    The mesh data is stored in half-edge data structure in a matrix form.
    Half-Edge r-th row : | Origin | Twin | Face | Next | Previous | Edge |
                             0        1     2      3       4          5

    The code is based on the code implemented by Davide Pellis. 

"""
__author__ = "Anthony Ramos"

import numpy as np
import copy
import scipy as sp
from scipy.sparse import coo_matrix

class Mesh():

    def __init__(self):

        # Number of Vertices
        self._V = 0

        # Number of Faces
        self._F = 0

        # Number of Edges
        self._E = 0

        # Half-Edge Data Structure
        self._halfedges = None

        # Vertices
        self._vertices = None

        # Boundary Half-Edges
        self._boundary_halfedges = None

    # Properties: ------------------------------------------------------------------

    @property
    def V(self):
        return self._V
    
    @property
    def F(self):
        return self._F
    
    @property
    def E(self):
        return self._E
    
    @property
    def BH(self):
        return self._boundary_halfedges
    
    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, vertices):
        vertices = np.asarray(vertices, dtype=np.float64)
        # Assert that vertices are 3D 
        assert vertices.ndim == 2, "vertices must be an array of shape (n,3)"
        assert vertices.shape[1] == 3, "vertices must be an array of shape (n,3)"
        
        # Set the vertices
        self._vertices = vertices
        self.update_dimensions()        

    @property
    def H(self):
        return self._halfedges
    
    @H.setter
    def H(self, halfedges):
        halfedges = np.asarray(halfedges, dtype=np.int64)
        # Assert that halfedges are 6D 
        assert halfedges.ndim == 2, "halfedges must be an array of shape (n,6)"
        assert halfedges.shape[1] == 6, "halfedges must be an array of shape (n,6)"
        
        # Set the halfedges
        self._halfedges = halfedges
        self.update_dimensions()


    # Updates Functions: ------------------------------------------------------------------ 

    def update_dimensions(self):
        self._V =  np.amax(self._halfedges[:,0]) + 1
        self._F =  np.amax(self._halfedges[:,2]) + 1
        self._E =  np.amax(self._halfedges[:,5]) + 1

    # Methods: ------------------------------------------------------------------

    def __str__(self):
        string = "Mesh Data Structure: |V| = {}, |F| = {}, |E| = {}".format(self.V, self.F, self.E)
        return string
    
    def copy_mesh(self):
        copyMesh = Mesh()
        copyMesh._vertices = self.vertices.copy()
        copyMesh.H = self.H.copy()
        copyMesh._V = self.V
        copyMesh._F = self.F
        copyMesh._E = self.E
        return copyMesh

    # Mesh Functions: ------------------------------------------------------------------
    def make_mesh(self, vertices, faces):
        try:
            self._make_mesh(vertices, faces)
        except:
            try:
                faces = self.orient_faces(vertices, faces)
                self._make_mesh(vertices, faces)
                print("Faces were oriented")
            except:
                raise Exception("Mesh not orientable")
        print(self)

    def _make_mesh(self, vertices, faces):
        vertices = np.array(vertices, dtype=np.float64)
        faces = np.array(faces, dtype=np.int16)
        
        self._vertices = vertices
        self._V = vertices.shape[0]
        self._F = len(faces)
        
        # Init Columns of Half-Edge Data Structure
        origin = []
        twin_i = []
        twin_j = []
        face_idxs = []
        next = []
        previous = []
        

        # halfedge counter
        h_count = 0
        for f in range(self.F):
            # First half-edge of the face
            # ---------------------------
            n = len(faces[f]) # Number of vertices in face f
            origin.append(faces[f, 0]) # First vertex in the face
            face_idxs.append(f) # Face of the corresponding half-edge
            next.append(h_count + 1) # Next vertex of the face
            previous.append(h_count + n - 1) # Last vertex of the face
            
            # Record half edge direction of twin ij -> ji
            twin_i.append(faces[f, 1])
            twin_j.append(faces[f, 0])
            
            h_count += 1
            # ---------------------------

            # Loop over the remaining vertices in the face
            for i in range(1, n - 1):
                # Add next origin
                origin.append(faces[f,i])
                # Face to next half-edge
                face_idxs.append(f)
                next.append(h_count  + 1)
                previous.append(h_count - 1)
                
                # Record half edge direction of twin ij -> ji
                twin_i.append(faces[f, i + 1])
                twin_j.append(faces[f, i])

                h_count += 1

            # Last half-edge of the face
            # ---------------------------
            origin.append(faces[f, n - 1])
            face_idxs.append(f)
            next.append(h_count - n + 1)
            previous.append(h_count - 1)
            twin_i.append(faces[f, 0])
            twin_j.append(faces[f, n - 1])

            h_count += 1

        # Set twin half-edges
        # Create a sparse matrix with the twin half-edges (hafl-edge index) = (i,j)
        twin = coo_matrix((np.arange(h_count) + 1, (twin_i, twin_j)), shape=(h_count, h_count)).tocsc()
        # Set half-edges data structure
        H = np.zeros((h_count, 6), dtype=np.int16)
        H[:, 0] = origin
        H[:, 2] = face_idxs
        H[:, 3] = next
        H[:, 4] = previous

        H[:, 1] = twin[origin, H[H[:,3], 0] ] - 1
        
        # Boundary half-edges
        # Indices of half-edges with no twin
        b = np.where(H[:, 1] == -1)[0]
        # Boundary half-edges
        boundary = H[b, :]
        boundary[:, 0] = H[H[b,3],0]
        boundary[:, 2] = -1
        boundary[:, 1] = b

        print("H: ", H)

        B = len(boundary)
        if B > 0:
            # Indices for the boundary half-edges
            Bh = np.arange(h_count, h_count + B)

            # Auxiliary zero array
            zeros = np.zeros(B, dtype=np.int8)

            # Set Twins for boundary half-edges
            H[b, 1] = Bh

            # Dictionary for the boundary half-edges
            indices = H[b, 0]
            values = Bh
            dic = coo_matrix((values, (indices, zeros)), shape=(self.V, 1)).tocsc()
            
            
            # Set previus for boundary half-edges
            boundary[:, 4] = dic[boundary[:,0], zeros]

            # i 
            nextindices = boundary[boundary[:,4] - h_count, 0]         

            # Dictionary for next boundary half-edges
            indices = nextindices
            values = Bh
            dic = coo_matrix((values, (indices, zeros)), shape=(self.V, 1)).tocsc()

            # Set next for boundary half-edges
            boundary[:, 3] = dic[boundary[:,0], zeros]

            # Add boundary half-edges to the half-edges data structure
            H = np.vstack((H, boundary))
        
        # Extract (prev, twin) columns
        aux = H[:, (4,1)]

        # Set first column to array of indices of vertices
        aux[:, 0] = np.arange(H.shape[0]) 

        # Get the minimun of each row
        m = np.amin(aux, axis=1)
        u = np.unique(m)

        imap = np.arange(np.max(u)+1, dtype=np.int16)
        imap[u] = np.arange(u.shape[0], dtype=np.int16)

        # Set edges
        H[:, 5] = imap[m]

        # Set half-edges data structure
        self._halfedges = H
        self._E = int(H.shape[0]/2)
        self.update_dimensions()

        
    def orient_faces(self, vertices_list, faces_list):
        """ 
        Orientation of faces code made by Davide Pellis and commented using ChatGPT 3.
        """
        # Get the number of faces and vertices
        F = len(faces_list)
        V = len(vertices_list)
        
        # Initialize data structures
        fmap = -np.ones((V,V), dtype=np.int16)  # Face mapping between vertices
        inconsistent = np.zeros((V,V), dtype=np.int16)  # Inconsistent vertices tracker
        flipped = np.zeros(F, dtype=bool)  # Array to track flipped faces
        oriented = np.zeros(F, dtype=bool)  # Array to track oriented faces
        oriented_faces = copy.deepcopy(faces_list)  # Create a deep copy of faces_list
        
        # Iterate over each face and each vertex to update mappings and track inconsistencies
        for f in range(F):
            face = faces_list[f]
            for j in range(len(face)):
                v0 = face[j-1]
                v1 = face[j]
                if fmap[v0,v1] == -1:
                    fmap[v0,v1] = f
                else:
                    fmap[v1,v0] = f
                    inconsistent[v0,v1] = True
                    inconsistent[v1,v0] = True
        
        # Begin the face orientation process
        ring = [0]  # Start with the first face
        oriented[0] = True  # Mark the first face as oriented
        i = 1  # Counter for the number of oriented faces
        
        while len(ring) > 0:
            next_ring = []  # Store the indices of the next set of faces to be processed
            
            # Iterate over each face index in the current ring
            for f in ring:
                face = faces_list[f]
                for j in range(len(face)):
                    flip = False
                    v0 = face[j-1]
                    v1 = face[j]
                    
                    # Determine the correct order of vertices based on mappings
                    if fmap[v0,v1] == f:
                        v2 = v1
                        v3 = v0
                    else:
                        v2 = v0
                        v3 = v1
                    
                    # Check for inconsistencies and flipping of faces
                    if inconsistent[v2,v3] and not flipped[f]:
                        flip = True
                    if not inconsistent[v2,v3] and flipped[f]:
                        flip = True
                    
                    fi = fmap[v2,v3]  # Get the face index from the mapping
                    
                    # Orient the face if it's not already oriented
                    if fi != -1 and not oriented[fi]:
                        if fi not in next_ring:
                            next_ring.append(fi)
                        if flip:
                            oriented_faces[fi].reverse()  # Reverse the vertex order
                            flipped[fi] = True  # Mark the face as flipped
                        i += 1
                        oriented[fi] = True  # Mark the face as oriented
                        
                        if i == F:  # If all faces have been oriented, return the result
                            return oriented_faces
            
            ring = next_ring  # Update the current ring of faces
            
            # If the current ring is empty, find the index of the first unoriented face
            if len(ring) == 0:
                try:
                    ring = [np.where(oriented == False)[0][0]]
                except:
                    return  # All faces have been oriented, so return
            




    
    
    
