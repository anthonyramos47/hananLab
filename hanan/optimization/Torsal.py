# Planarity constraint implementation

import numpy as np
from hanan.geometry.utils import vec_dot
from hanan.optimization.constraint import Constraint


class Torsal(Constraint):

    def __init__(self) -> None:
        """ 
        Initializes a new instance of the Torsal class.

        Torsal directions constraint energy, we assume nt are unitary vectors.
        E = \sum{f \in F} || n_t1 . e_c ||^2 + || n_t1 . t1 ||^2 + || n_f1 . tt1 ||^2 
            +             || n_t2 . e_c ||^2 + || n_t2 . t2 ||^2 + || n_f2 . tt2 ||^2 
        where, 
            n_t .- normal of torsal plane.
            t   .- torsal direction in the face of triangular mesh T.
            tt  .- torsal direction in the face of triangular mesh \tilde{T} [second envelope]
            e_c .- direction of the line congruence at the barycenter of the face.
        """
        super().__init__()
        self.nV = None # Number of vertices
        self.F = None # Faces
        self.nF = None # Number of faces
        self.ncf = None # List of circumcenter axis direction
        self.bf = None # List of circumcenters
        self.fvij = None # List of the edge vectors per each face
        self.v = None # List of the vertices per each face
        self.vc= None # List og the barycenters of the faces
        self.ecnorms = None # List of the norms of the line congruence directions
        self.ttnorms1 = None # List of the norms of the torsal directions
        self.ttnorms2 = None # List of the norms of the torsal directions


    
    def initialize_constraint(self, X, var_indices, V, F, bf, ncf) -> np.array:
        # Input
        # X: variables [ e   | a | b | n_t  | d_i ] 
        # X  size      [ 3*V | F | F | 3*F  | F   ]
        # V: Vertices
        # F: Faces
        # bf: circumcenters of the faces
        # ncf: normals of the circumcenters

        self.F = F
        # Set circumcenters
        self.bf = bf

        # Set circumcenters axis
        self.ncf = ncf

        # Number of vertices
        self.nV = len(V)
        
        # Number of faces
        self.nF = len(F)

        # Number of constraints 4*|F|
        self.const = 8*self.nF

        # Define indices indices
        self.var_idx = var_indices
    
        self.const_idx = {  "nt1.ec"  : np.arange( 0                  , self.nF),
                            "nt1.t1"  : np.arange( self.nF            , 2*self.nF),
                            "nt1.tt1" : np.arange( 2*self.nF          , 3*self.nF),
                            "nt2.ec"  : np.arange( 3*self.nF          , 4*self.nF),
                            "nt2.t2"  : np.arange( 4*self.nF          , 5*self.nF),
                            "nt2.tt2" : np.arange( 5*self.nF          , 6*self.nF),
                            "ut1"    : np.arange( 6*self.nF          , 7*self.nF),
                            "ut2"    : np.arange( 7*self.nF          , 8*self.nF)                          
                        }
        
        # Number of variables
        self.var = len(X)

        # Get df 
        df = X[var_indices["df"]]

        # Get ei
        e_i = X[var_indices["e"]].reshape(-1, 3)
        
        # Get vertices of the faces
        vi, vj, vk = V[F[:,0]], V[F[:,1]], V[F[:,2]]
        self.v = vi, vj, vk
        self.vc = (vi + vj + vk)/3

        # Compute the edge vectors per each face
        self.fvij = np.empty((self.nF, 2, 3), dtype=float)

        # Compute the edge vectors per each face
        self.fvij[:,0] = vj - vi
        self.fvij[:,1] = vk - vi

        # Compute the direction of the line congruence at the barycenters
        ec = self.compute_ec(df, e_i, F)

        self.ecnorms = np.linalg.norm(ec, axis=1)

        # Set initial a1 
        X[self.var_idx["a1"]] = np.linalg.norm(vj-vi, axis=1)
        X[self.var_idx["b1"]] = np.linalg.norm(vk-vi, axis=1)

        t1 = self.compute_t(X[self.var_idx["a1"]], X[self.var_idx["b1"]])

        X[self.var_idx["a1"]] /= np.linalg.norm(t1, axis=1)
        X[self.var_idx["b1"]] /= np.linalg.norm(t1, axis=1)

        # Set initial b
        X[self.var_idx["a2"]] = np.linalg.norm(vj-vi, axis=1)
        X[self.var_idx["b2"]] = -np.linalg.norm(vk-vi, axis=1)

        t2 = self.compute_t(X[self.var_idx["a2"]], X[self.var_idx["b2"]])

        X[self.var_idx["a2"]] /= np.linalg.norm(t2, axis=1)
        X[self.var_idx["b2"]] /= np.linalg.norm(t2, axis=1)

    
        vvi, vvj, vvk, _, _, _ = self.compute_second_env(df, e_i, F)
        

        tt, _, _ = self.compute_tt(X[self.var_idx["a1"]], X[self.var_idx["b1"]], vvi, vvj, vvk)

        self.ttnorms1 = np.linalg.norm(tt, axis=1)

        tt, _, _ = self.compute_tt(X[self.var_idx["a2"]], X[self.var_idx["b2"]], vvi, vvj, vvk)

        self.ttnorms2 = np.linalg.norm(tt, axis=1)

        # Compute initial directions of normals of torsal plane
        nt1 = np.cross(ec, t1)
        nt1 /= np.linalg.norm(nt1, axis=1)[:, None]

        nt2 = np.cross(ec, nt1)
        nt2 /= np.linalg.norm(nt2, axis=1)[:, None]
        
        # Set initial directions of normals of torsal plane
        X[self.var_idx["nt1"]] = nt1.flatten()
        X[self.var_idx["nt2"]] = nt2.flatten()





    def compute(self, X) -> None:
        """ Compute the residual and the Jacobian of the constraint
            Input:
                X: Variables
                F: Faces
        """
        e, a1, b1, nt1, a2, b2, nt2, df = self.uncurry_X(X, "e", "a1", "b1", "nt1", "a2", "b2", "nt2", "df")

        e   = e.reshape(-1, 3)
        nt1 = nt1.reshape(-1, 3)
        nt2 = nt2.reshape(-1, 3)

        
        # Get vertices of second envelope
        vvi, vvj, vvk, dcvi, dcvj, dcvk = self.compute_second_env(df, e, self.F)

        # Compute ec
        ec = self.compute_ec(df, e, self.F)
        

        # Get vij, vik
        vij = self.fvij[:,0]
        vik = self.fvij[:,1]

        # Init indices for sparse J matrix
        self.fill_J_t(e, ec, a1, b1, nt1, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, self.F, 1)
        self.fill_J_t(e, ec, a2, b2, nt2, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, self.F, 2)     

        self.ecnorms = np.linalg.norm(ec, axis=1)
        

    def fill_J_t(self, e, ec, a1, b1, nt1, vij, vik, vvi, vvj, vvk, dcvi, dcvj, dcvk, F, torsal=1):
        """ Function to define the values of J per each torsal direction
        """

        if torsal == 1:
            nt_ec = "nt1.ec"
            nt_t = "nt1.t1"
            nt_tt = "nt1.tt1"
            nt = "nt1"
            va = "a1"
            vb = "b1"
            ut = "ut1"
            ttnorms = self.ttnorms1
            
        else:
            nt_ec = "nt2.ec"
            nt_t = "nt2.t2"
            nt_tt = "nt2.tt2"
            nt = "nt2"
            va = "a2"
            vb = "b2"
            ut = "ut2"
            ttnorms = self.ttnorms2
            
        # Compute derivatives of de (nt.ec)
        deix, deiy, deiz, dejx, dejy, dejz, dekx, deky, dekz = self.compute_decnt(dcvi, dcvj, dcvk, e, nt1, F)
        
        # Compute derivatives of d df (nt.ec)
        d_df, eicf_ei, ejcf_ej, ekcf_ek = self.compute_d_df(e, nt1, F)

        # Compute t
        t = self.compute_t(a1, b1)

        # Compute tt
        tt, vvij, vvik = self.compute_tt(a1, b1, vvi, vvj, vvk)

        # constrain indices
        c_idx = self.const_idx
        v_idx = self.var_idx

        # Fill J for || nt.ec ||^2
        # Set derivatives de (nt.ec)
        # J[c_idx[nt_ec], 3*F[:,0] ]   = 2/3*deix
        # J[c_idx[nt_ec], 3*F[:,0]+1 ] = 2/3*deiy
        # J[c_idx[nt_ec], 3*F[:,0]+2 ] = 2/3*deiz
        # J[c_idx[nt_ec], 3*F[:,1] ]   = 2/3*dejx
        # J[c_idx[nt_ec], 3*F[:,1]+1 ] = 2/3*dejy
        # J[c_idx[nt_ec], 3*F[:,1]+2 ] = 2/3*dejz
        # J[c_idx[nt_ec], 3*F[:,2] ]   = 2/3*dekx
        # J[c_idx[nt_ec], 3*F[:,2]+1 ] = 2/3*deky
        # J[c_idx[nt_ec], 3*F[:,2]+2 ] = 2/3*dekz
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,0]]  , 2/3*deix/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,0]]+1, 2/3*deiy/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,0]]+2, 2/3*deiz/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,1]]  , 2/3*dejx/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,1]]+1, 2/3*dejy/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,1]]+2, 2/3*dejz/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,2]]  , 2/3*dekx/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,2]]+1, 2/3*deky/self.ecnorms)
        # self.add_derivatives(c_idx[nt_ec], 3*v_idx["e"][F[:,2]]+2, 2/3*dekz/self.ecnorms)    

        # Normalize ec
        ecnor = (ec/self.ecnorms[:,None])
        

        # Set derivateives dnt (nt.ec)
        #J[c_idx[nt_ec].repeat(3), v_idx[nt] ] = ec.flatten()
        self.add_derivatives(c_idx[nt_ec].repeat(3), v_idx[nt], ecnor.flatten())
        
        # Set derivatives d(df) (nt.ec)
        #J[c_idx[nt_ec], v_idx["df"]] = d_df
        #self.add_derivatives(c_idx[nt_ec], v_idx["df"], d_df/self.ecnorms)

        # Set r
        # r[c_idx[nt_ec]] = np.sum(nt1*ec, axis=1)
        self.set_r(c_idx[nt_ec], np.sum(nt1*ecnor, axis=1))

        
        # Fill J for || nt.t ||^2; t = a vij + b vik
                
        # Set derivatives dnt (nt.t)
        # J[c_idx[nt_t].repeat(3), v_idx[nt]    ] = tnor.flatten()
        self.add_derivatives(c_idx[nt_t].repeat(3), v_idx[nt], t.flatten())
        
        # Set derivatives da (nt.t)
        # J[c_idx[nt_t], v_idx[va]] = vec_dot(vij, nt1)/tnorms 
        self.add_derivatives(c_idx[nt_t], v_idx[va], vec_dot(vij, nt1))

        # Set derivatives db (nt.t)
        #J[c_idx[nt_t], v_idx[vb]] = vec_dot(vik, nt1)/tnorms
        self.add_derivatives(c_idx[nt_t], v_idx[vb], vec_dot(vik, nt1))

        # Set r 
        #self.r[c_idx[nt_t]] = np.sum(tnor*nt1, axis=1)
        self.set_r(c_idx[nt_t], np.sum(t*nt1, axis=1))

        # Fill J for || nt.tt ||^2; tt = a vvij + b vvik
        ttnor = tt/ttnorms[:, None]
        
        # Set derivatives dnt (nt.tt)
        # J[c_idx[nt_tt].repeat(3), v_idx[nt]   ] = ttnor.flatten()
        self.add_derivatives(c_idx[nt_tt].repeat(3), v_idx[nt], ttnor.flatten())

        # Set derivatives da (nt1.tt1)
        # J[c_idx[nt_tt], v_idx[va]] = vec_dot(vvij, nt1)/ttnorms
        self.add_derivatives(c_idx[nt_tt], v_idx[va], vec_dot(vvij, nt1)/ttnorms)

        # Set derivatives db (nt1.tt1)
        # J[c_idx[nt_tt], v_idx[vb]] = vec_dot(vvik, nt1)/ttnorms
        self.add_derivatives(c_idx[nt_tt], v_idx[vb], vec_dot(vvik, nt1)/ttnorms)

        # Set derivatives d e (nt.tt)
        #J[c_idx[nt_tt], 3*F[:,0] ]   = -2*(a1+b1)*deix/ttnorms
        #J[c_idx[nt_tt], 3*F[:,0]+1 ] = -2*(a1+b1)*deiy/ttnorms
        #J[c_idx[nt_tt], 3*F[:,0]+2 ] = -2*(a1+b1)*deiz/ttnorms
        # self.add_derivatives(c_idx[nt_tt],   3*F[:,0], -2*(a1+b1)*deix/ttnorms)
        # self.add_derivatives(c_idx[nt_tt], 3*F[:,0]+1, -2*(a1+b1)*deiy/ttnorms)
        # self.add_derivatives(c_idx[nt_tt], 3*F[:,0]+2, -2*(a1+b1)*deiz/ttnorms)

        # #J[c_idx[nt_tt], 3*F[:,1] ]   = 2*a1*dejx/ttnorms
        # #J[c_idx[nt_tt], 3*F[:,1]+1 ] = 2*a1*dejy/ttnorms
        # #J[c_idx[nt_tt], 3*F[:,1]+2 ] = 2*a1*dejz/ttnorms
        # self.add_derivatives(c_idx[nt_tt],   3*F[:,1], 2*a1*dejx/ttnorms)
        # self.add_derivatives(c_idx[nt_tt], 3*F[:,1]+1, 2*a1*dejy/ttnorms)
        # self.add_derivatives(c_idx[nt_tt], 3*F[:,1]+2, 2*a1*dejz/ttnorms)

        # #J[c_idx[nt_tt], 3*F[:,2] ]   = 2*b1*dekx/ttnorms
        # #J[c_idx[nt_tt], 3*F[:,2]+1 ] = 2*b1*deky/ttnorms
        # #J[c_idx[nt_tt], 3*F[:,2]+2 ] = 2*b1*dekz/ttnorms
        # self.add_derivatives(c_idx[nt_tt],   3*F[:,2], 2*b1*dekx/ttnorms)
        # self.add_derivatives(c_idx[nt_tt], 3*F[:,2]+1, 2*b1*deky/ttnorms)
        # self.add_derivatives(c_idx[nt_tt], 3*F[:,2]+2, 2*b1*dekz/ttnorms)

        # Set derivatives d df (nt.tt)
        # J[c_idx[nt_tt], v_idx["df"]] = vec_dot((2*a1[:,None]*(ejcf_ej - eicf_ei) + 2*b1[:,None]*(ekcf_ek - eicf_ei)), nt1)/ttnorms
        #self.add_derivatives(c_idx[nt_tt], v_idx["df"], vec_dot((2*a1[:,None]*(ejcf_ej - eicf_ei) + 2*b1[:,None]*(ekcf_ek - eicf_ei)), nt1)/ttnorms)
        
        # Set r
        # r[c_idx[nt_tt]] = np.sum(ttnor*nt1, axis=1)
        self.set_r(c_idx[nt_tt], np.sum(ttnor*nt1, axis=1))

        # Normalization of t;  t = a vij + b vik
        # da ||t.t -1||^2 = 2 vij. t 
        self.add_derivatives(c_idx[ut], v_idx[va], 2*vec_dot(vij, t))
        # db ||t.t -1||^2 = 2 vik. t
        self.add_derivatives(c_idx[ut], v_idx[vb], 2*vec_dot(vik, t))

        self.set_r(c_idx[ut], vec_dot(t, t) - 1)

        if torsal== 1:
            self.ttnorms1 = np.linalg.norm(tt, axis=1)
        else:
            self.ttnorms2 = np.linalg.norm(tt, axis=1)




    def compute_second_env(self, df, ei, F):
        """ Compute the second envelope of the mesh
        Input:
            df .- distance to center np.array
            vi .- Vertices np.array
            ei .- direction vectors np.array
        Output:
            vvi, vvj, vvk, dcvi, dcvj, dcvk
        """

        cf = self.compute_sphere_centers(df)

        # Get vertices of the faces
        vi, vj, vk = self.v 
        
        # Direction from vi to cf
        dcvi = cf - vi
        # Direction from vj to cf
        dcvj = cf - vj
        # Direction from vk to cf
        dcvk = cf - vk

        ue = ei/np.linalg.norm(ei, axis=1)[:, None]

        # compute the second envelope as vi + 2*(dcv.ei) uei; uei = ei/||ei||
        vvi = 2*np.sum(dcvi*ue[F[:,0]], axis=1)[:, None] * ue[F[:,0]] + vi
        vvj = 2*np.sum(dcvj*ue[F[:,1]], axis=1)[:, None] * ue[F[:,1]] + vj
        vvk = 2*np.sum(dcvk*ue[F[:,2]], axis=1)[:, None] * ue[F[:,2]] + vk

        return vvi, vvj, vvk, dcvi, dcvj, dcvk


    def compute_sphere_centers(self, df) ->  np.array:
        """ Compute sphere centers
        Input:
            bf .- Circumcenters
            df .- Distance to center
            ncf .- Normal of circumcenters
        """
        return self.bf + df[:, None] * self.ncf
    
    def compute_ec(self, df, e_i, F) -> np.array:
        """ Compute the direction of the line congruence at the barycenters
        Input:
            df .- Distance to center
            e_i .- direction vectors np.array
            F.- Faces
        Output:
            ec .- direction of the line congruence at the barycenters
        """
        # Get first envelope
        vc = self.vc

        # Compute second envelope
        vvi, vvj, vvk, _, _, _ = self.compute_second_env(df, e_i, F)

        # Compute varycenter in second envelope
        vvc = (vvi + vvj + vvk)/3

        # Compute ec as differencen between second and first envelope
        ec = vvc - vc

        if len(ec.shape) == 1:
            ec = np.array([ec])

        return ec


    
    def compute_decnt(self, dcvi, dcvj, dcvk, e, nt, F):
        """ Function to compute the derivative of nt.ec with respect to e_i
            (nt.ec) = (nt.(ei + ej + ek)/3) = (nt).(vvc - vc);
            vc = (vi + vj + vk)/3
            vvi = vi + 2*(dcv.ei) ei
        """
        # Get directions per vertex
        ei = e[F[:,0]]
        ej = e[F[:,1]]
        ek = e[F[:,2]]  

        # Compute derivatives of de (nt.ec)
        deix = vec_dot( ( dcvi[:,0][:, None]*ei +  vec_dot(ei, dcvi)[:, None]*np.array([1,0,0]) ), nt)
        deiy = vec_dot( ( dcvi[:,1][:, None]*ei +  vec_dot(ei, dcvi)[:, None]*np.array([0,1,0]) ), nt)
        deiz = vec_dot( ( dcvi[:,2][:, None]*ei +  vec_dot(ei, dcvi)[:, None]*np.array([0,0,1]) ), nt)

        dejx = vec_dot( ( dcvj[:,0][:, None]*ej +  vec_dot(ej, dcvj)[:, None]*np.array([1,0,0]) ), nt)
        dejy = vec_dot( ( dcvj[:,1][:, None]*ej +  vec_dot(ej, dcvj)[:, None]*np.array([0,1,0]) ), nt)
        dejz = vec_dot( ( dcvj[:,2][:, None]*ej +  vec_dot(ej, dcvj)[:, None]*np.array([0,0,1]) ), nt)

        dekx = vec_dot( ( dcvk[:,0][:, None]*ek +  vec_dot(ek, dcvk)[:, None]*np.array([1,0,0]) ), nt)
        deky = vec_dot( ( dcvk[:,1][:, None]*ek +  vec_dot(ek, dcvk)[:, None]*np.array([0,1,0]) ), nt)
        dekz = vec_dot( ( dcvk[:,2][:, None]*ek +  vec_dot(ek, dcvk)[:, None]*np.array([0,0,1]) ), nt)

        return deix, deiy, deiz, dejx, dejy, dejz, dekx, deky, dekz


    def compute_d_df(self, e, nt, F):
        """ Compute the derivative of nt.ec with respect to df
            (nt.ec) = (nt.(ei + ej + ek)/3) = (nt).(vvc - vc);
            vc = (vi + vj + vk)/3
            vvi = vi + 2(dcv.ei)/||enor|| ei/||enor||; dcv = cf - vi 
            cf = bf + df ncf
            => d df (nt.ec) = 2/3 [nt.( (ei.ncf)* ei + (ej.ncf)*ej   + (ek.ncf)*ek )]/||enor||**2
        """
        # Get directions per vertex
        ei = e[F[:,0]]
        ej = e[F[:,1]]
        ek = e[F[:,2]] 

        # Get normals of the circumcenters
        ncf = self.ncf 

        # Compute dot produts
        eicf = vec_dot(ei, ncf)
        ejcf = vec_dot(ej, ncf)
        ekcf = vec_dot(ek, ncf)

        eicf_ei = eicf[:,None]*ei
        ejcf_ej = ejcf[:,None]*ej
        ekcf_ek = ekcf[:,None]*ek

        # Compute the derivative of nt.ec with respect to df
        d_df = 2/3*vec_dot( (eicf_ei + ejcf_ej + ekcf_ek ), nt)

        return d_df, eicf_ei, ejcf_ej, ekcf_ek   


    
    def compute_t(self, a, b):
        """ Compute t = a vij + b vik
        """
        return a[:, None]*self.fvij[:,0] + b[:, None]*self.fvij[:,1]

    def compute_tt(self, a, b, vvi, vvj, vvk):
        """ Compute tt = a vvij + b vvik
        """
        vvij = vvj - vvi
        vvik = vvk - vvi
        return a[:, None]*vvij + b[:, None]*vvik, vvij, vvik
    
       
 
    






        
       