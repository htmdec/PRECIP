# -*- coding: utf-8 -*-

################################################################################
#title          : PIG_E.py
#description    : Calculate elastic interfacial energy between matrix and precipitate, high-throuputtingly
#authors        : H. Murdoch, B. Szajewski, E. Hernandez-Rivera, M. Guziewski, D. Magagnosc (ARL)
#revised        : 28JUL2025
#version        : 1.0
#usage          : Steel_Info.py
#notes          : 
################################################################################

#%% Necessary Modules
import numpy as np
import matplotlib.pyplot as plt
import math 
from scipy.optimize import minimize_scalar
from copy import deepcopy
import itertools as it
#
# Importing materials specific information
# functions from this module will be commented with 'from {Material}_Info.py' for reference
#
from Steel_Info import * 

#%% Global variables / functions

#
# Conversion variables
#
AngstromLength=1.e-10
GigaPascal = 1.e9
Nav = 6.0221408e+23 # Avogadros number used in molar volume - > cell volume calculation


#
# Transformation Matrix / Crystallography functions
#

normalize = lambda x: x/np.linalg.norm(x)
unity = lambda size : np.ones((size,1))


def ConvertHCP4toHCP3(HCP):
    """
    changes from Miller-Bravais to Miller indices for HCP latice
               
    Parameters
    ----------
    HCP : Miller-Bravais indices (4)
        
    Returns
    -------
    Miller inidices (3)
    """
    
    HCP3 = np.array([2.*HCP[0] + HCP[1],
                     HCP[0] + 2.*HCP[1],
                     HCP[3]])
    
    return HCP3/np.gcd.reduce(HCP3.astype('int'))

def TensorSort(M):
    """
    Sorts a tensor according to the diagonal terms
               
    Parameters
    ----------
    M : a symmetric tensor
        
    Returns
    -------
    Sorted tensor
    """
    order = np.argsort(np.diag(M))
    c = np.array([order,order,order]).astype('int')
    r = np.array([np.ones(3)*order[0],np.ones(3)*order[1],np.ones(3)*order[2]]).astype('int')
    
    return M[r.flatten(),c.flatten()].reshape(M.shape) 


#%% CrystalProperties Class

class CrystalProperties:
    """
    Class to contain the necessary crystallographic and elastic information about a phase. 
    Each phase is a separate instance of the class.
    
    Attributes
    ----------
    CrystalStructure   : crystal structure of the phase in Thermo-Calc notation
    MolarVolume        : Molar volume of the phase, in m^3/mol, not used for manual option
    PhaseComposition   : composition of the phase in ATOMIC FRACTION, not used for manual option
    Temperature        : Temperature at which the lattice and elastic information is calculated, in KELVIN, not used for manual option
        
    a, b, c            : lattice parameters in ANGSTROMS
    alpha, beta, gamma : cell angles in DEGREES
    CBtoRB             : Crystal Basis to Reference Basis matrix
    g                  : Crystal metric tensor 

    ShearModulus       : shear modulus of the phase in GPa
    Poisson            : Poissons ratio of the phase
    
    CellVolume         : optional, if using manual lattice input, in Angstroms^3
    
    """
    
    def __init__(self, Crystal, Composition=None, Temperature=None, MolarVolume=None):
        """
        Initialize CrystalProperties class.  
        Composition/Temperature/MolarVolume are optional because you could manually input lattice parameters.

        Parameters
        ----------
        Crystal (str)       : crystal structure IN THERMO-CALC NOTATION for name of the phase
        Composition (dict)  : composition of the PHASE should be in ATOMIC FRACTION
        Temperature (float) : temper temperature of calculation, in KELVIN
        MolarVolume (float) : optional, should be in m^3/mol
              
        """
        
        CurrentCrystals = GetMatrixPhase() + GetPrecipitateList()   # from {Material}_Info.py
        
        #
        # Error checking
        #
        
        # making sure crystal structure is in the materials properties dictionaries  
        assert Crystal in CurrentCrystals, f'Input {Crystal} not in current phase list. Current phases are {CurrentCrystals}'
        
        # checking that composition is atomic fraction
        if Composition:
            assert sum(Composition.values()) < 1.1, 'Is this composition in atomic FRACTION?'
         
        #
        # Initializing information
        #
        self.CrystalStructure = Crystal
        self.MolarVolume = MolarVolume
        self.PhaseComposition = Composition
        self.Temperature = Temperature
           

    
    #
    # LATTICE INFORMATION
    #
         
    def Angles(self, alpha, beta, gamma): # angles in degrees
        """
        Function to calculate the angle dependent term of the cell volume
        CellVolume = a * b * c (1-cos^2⁡α-cos^2⁡β-cos^2⁡γ+2 cos⁡α  cos⁡β  cos⁡γ)
        
        Parameters
        ----------
        alpha , beta, gamma : cell angles IN DEGREES
        
        Returns
        -------
        angle dependent term of the cell volume
        """
        # convert angles from degrees to radians
        alphaR = alpha * np.pi/180.
        betaR = beta * np.pi/180.
        gammaR = gamma * np.pi/180.
            
        return (1. - np.cos(alphaR)**2. - np.cos(betaR)**2. - np.cos(gammaR)**2. + 2. * np.cos(alphaR) * np.cos(betaR) * np.cos(gammaR))**(1./2.)

    
    def InputLattice(self, a, b, c, alpha = 90, beta = 90, gamma = 90):
        """
        Defining the lattice MANUALLY, rather than from ThermoCalc
        Assigns lattice parameters to class.
        Assigns Cell Volume to class in same dimension as lattice parameter input, e.g. Angstroms^3.
        Generates matrix to orthogonalize crystal lattice to a reference lattice, CBtoRB. 
        
        Parameters
        ----------
        a, b, c            : lattice parameters IN ANGSTROMS
        alpha, beta, gamma : cell angles IN DEGREES
        
        """
        # assign lattice parameters to class
        self.a = a
        self.b = b
        self.c = c

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
            
        self.CellVolume = a * b * c * self.Angles(alpha, beta, gamma) 

        self.CBtoRB = self.Orthogonalize() 
        self.g = self.MetricTensor()
        
        return
    
    def LatticeParameters(self):
        """
        Calculating lattice spacing from molar volume (e.g. input from ThermoCalc output)
        Assigns lattice parameters and cell angles to class.
        Generates matrix to orthogonalize crystal lattice to a reference lattice, CBtoRB. 
        
        Uses precompiled dictionaries from {Materials}_Info.py describing the unit cells of each phase:
            Unit       : # atoms/unit cell
            Ratio      : ratios of lattice parameters in the cell, b:a and c:a
            CellAngles : angles of the cell in degrees
            
        MolarVolume [m^3/mol](mol/N_AV ) ((10^(-10) Å)/m)^3 (atoms/(unit cell)) -> CellVolume  [Å]
        CellVolume = a * b * c (1-cos^2⁡α-cos^2⁡β-cos^2⁡γ+2 cos⁡α  cos⁡β  cos⁡γ)

        """
        
        Unit, Ratio, CellAngles = LatticeParameterInfo(self.CrystalStructure) # from {Material}_Info.py
            
        
        # 
        # Calculating the lattice parameters from the volume
        #
        
        self.a = (self.MolarVolume / Nav / 
                   ((10**-10)**3) * Unit
                                   / ( Ratio['b:a'] * Ratio['c:a']
                                       * self.Angles(CellAngles[0], 
                                              CellAngles[1],
                                              CellAngles[2])
                                     ))**(1/3)
            
        self.b = self.a * Ratio['b:a']
        self.c = self.a * Ratio['c:a']

        self.alpha = CellAngles[0]
        self.beta = CellAngles[1]
        self.gamma = CellAngles[2]

        self.CBtoRB = self.Orthogonalize()
        self.g = self.MetricTensor()
                   
        return
    
    def Orthogonalize(self):
        """
        Generates matrix to orthogonalize crystal lattice to a reference lattice, CBtoRB

        Returns
        -------
        Crystal Basis to Reference Basis matrix (CBtoRB)

        """
        # convert angles from degrees to radians
        alphaR = self.alpha * np.pi/180.
        betaR = self.beta * np.pi/180.
        gammaR = self.gamma * np.pi/180.

        # calculate reciprocal cell angle corresponding to alpha
        CosineAlphaStar = (np.cos(betaR) * np.cos(gammaR) - np.cos(alphaR)) / (np.sin(betaR) * np.sin(gammaR))
        
        return np.array([[self.a, self.b*np.cos(gammaR), self.c*np.cos(betaR)],
                         [0, self.b*np.sin(gammaR), -self.c*np.sin(betaR)*CosineAlphaStar],
                         [0, 0, self.c*np.sin(betaR)*np.sqrt(1-CosineAlphaStar**2)]])

    def MetricTensor(self):
        """
         Generates the crystal metric tensor.
         
         Returns
         -------
         Metric tensor
        
        """
        # convert angles from degrees to radians
        alphaR = self.alpha * np.pi/180.
        betaR = self.beta * np.pi/180.
        gammaR = self.gamma * np.pi/180.
        
        g = np.array([[self.a**2, self.a*self.b*np.cos(gammaR), self.a*self.c*np.cos(betaR)],
                      [0,self.b**2,self.b*self.c*np.cos(alphaR)],
                      [0,0,self.c**2]])
        
        g = g+g.T - np.diag(g.diagonal())
        
        return g            
    
    #
    # ELASTIC INFORMATION
    #
    
    def InputElastic(self, Shear, Poisson):
        """
        Manually input elastic constants rather than calculate from precompiled information

        Parameters
        ----------
        Shear   : Shear Modulus of the phase in GPa.
        Poisson : Poissons ratio of the phase.

        Assigns
        -------
        ShearModulus and Poissons attributes to class.

        """
        
        self.ShearModulus = Shear
        self.Poisson = Poisson
        
        return
    
    def ElasticConstants(self, Martensite=True, Curie=False):
        """
        Assigns shear modulus and poisson values to class based on crystal structure, composition, and temperature

        Parameters
        ----------
        Martensite : Whether to use the equations for alpha'-martensite or alpha-ferrite. The default is True.
        Curie      : Whether to use composition dependent Curie temperature for matrix shear modulus equation. The default is False.

        Assigns
        -------
        ShearModulus - in units of GPa
        Poisson - dimensionless

        """
        
        # MATRIX    
        if self.CrystalStructure == GetMatrixPhase()[0]:    # from {Material}_Info.py
            
            ShearModulus, Poisson = ElasticConstantsMATRIX(self, Martensite=Martensite, Curie=Curie)  # from {Material}_Info.py     
    
        # PRECIPITATES      
        else: 
            ShearModulus, Poisson = ElasticConstantsPRECIP(self)    # from {Material}_Info.py
                     
    
        #
        # Assign final values to class
        #
        self.ShearModulus = ShearModulus
        self.Poisson = Poisson
        
        return 

#%% CrystalInterface Class

class CrystalInterface(NewOrientationRelationships):    # NewOrientationRelationships from {Material}_Info.py
    """
    Calculates information necessary to describe the orientation relationship (OR) between the matrix
    and precipitate phases. Pre-populated with some common ORs; new ORs can be added through the 
    NewOrientationRelatioonships class which should be located in the {Material}_Info module.
    
    Attributes
    ----------
    a_Matrix, b_Matrix, c_Matrix  : lattice parameters of the matrix
    Precipitate                   : name of the precipitate, in Thermo-Calc notation
    a1, a2, a3                    : lattice parameters of the precipitate
    
    CurrentORs                    : list of OR functions already in the class
    bCount                        : Number of burgers vectors in the interface plane for the OR
    ReferenceFrame                : what crystallographic reference frame the reference burgers vectors are in
    InterfaceSubUnits             : a scaling factor for unit cells which contain more than one precipitate structural unit relative to a matrix structural unit on the interface plane
       
    FBToA                         : dictionary of results of transformation matrices in global reference frame
    b                             : dictionary of results of burgers vectors
    """
    #
    # Initializing by passing the instance of the CrystalProperties class for 
    # the Matrix (MatrixPhase) and Precipitate (PrecipPhase)
    #
    def __init__(self, MatrixPhase, PrecipPhase):
        
        self.Matrix = deepcopy(MatrixPhase)

        # lattice parameters of matrix in Angstroms
        self.a_Matrix = MatrixPhase.a * AngstromLength
        self.b_Matrix = MatrixPhase.b * AngstromLength
        self.c_Matrix = MatrixPhase.c * AngstromLength
        
        
        self.Precipitate = deepcopy(PrecipPhase)
        
        # lattice parameters of precpitate in Angstroms
        self.a1 = PrecipPhase.a * AngstromLength
        self.a2 = PrecipPhase.b * AngstromLength
        self.a3 = PrecipPhase.c * AngstromLength
                
        # Dictionaries for results
        self.FBToA = {}
        self.b     = {}
        
        #
        # Initialize with information about ORs that have existing functions within the class
        #
        self.CurrentORs = ['Bain', 'KurdjumovSachs', 'NishiyamaWasserman','Pitsch',
                           'PitschSchrader','Burgers',
                           'Bagaryatski','Isaichev']
        # number of reference vectors in the OR
        self.bCount= {'Bain' :2,
                           'KurdjumovSachs' :3,
                           'NishiyamaWasserman' :3,
                           'Pitsch' :2,
                           'PitschSchrader' :3,
                           'Burgers' :3,
                           'Bagaryatski' :2,
                           'Isaichev' :2,
                            }

        #
        # some OR planes are sub units of the cell...; GetSubUnits() should be in {Material}_Info module
        #
        self.InterfaceSubUnits = GetSubUnits()  # from {Material}_Info.py

        #
        # Adding new Orientation Relationships from {Material}_Info.py
        #
        NewOrientationRelationships.__init__(self)
        self.bCount.update(NewOrientationRelationships.bCount)

        
        
    def GetInterfaces(self):
        """
        Function to calculate the Transformation matrix (FBToA) and Burgers vectors (b) for the interface

        Returns
        -------
        Populates FBToA and b dictionaries with results using the OR name as keys

        """
        # get dict of ORs for each precipitate fron {Material}_info
        PrecipitateInterfaces = GetPrecipitateORs()     # from {Material}_Info.py        
       
        try:
            for Interface in PrecipitateInterfaces[self.Precipitate.CrystalStructure]:
                # check if particular precipitate / OR combination requires subunits
                
                if self.Precipitate.CrystalStructure in self.InterfaceSubUnits.keys() and Interface in self.InterfaceSubUnits[self.Precipitate.CrystalStructure]:
                    #print('using subcell')
                    self.SubUnitDict = self.InterfaceSubUnits[self.Precipitate.CrystalStructure][Interface] 
                    
                else: 
                    self.SubUnitDict = {'matrix' : unity(3),
                                        'precip' : unity(3),
                                        'ref' : unity(self.bCount[Interface])
                                        }
                """
                Structure of each function is as follows where alpha refers to matrix and beta refers to precipitate:
                    OR_alpha    : vectors for orientation relationship, matrix phase
                    OR_beta     : vectors for orientation relationship, precipitate phase
                    Qalpha      : matrix for matrix phase
                    Qbeta       : matrix for precipitate phase
                    alpha       : Qalpha x OR_alpha
                    beta        : Qbeta x OR_beta
                    b0          : reference burgers vectors
                    burgers     : transformed burgers vectors   
                """
                # 
                # Get the information from the OR specific function
                #
                OR_alpha, OR_beta, b0 = eval(f'self.{Interface}')()

                #
                # same for all ORs
                #
                Qalpha = self.GetRotationMatrix( OR_alpha[0], OR_alpha[2], self.Matrix.CBtoRB)
                
                alpha = np.array([np.matmul( Qalpha, OR_alpha[0]),
                                  np.matmul( Qalpha, OR_alpha[1]),
                                  np.matmul( Qalpha, OR_alpha[2])]).T

                Qbeta = self.GetRotationMatrix( OR_beta[0], OR_beta[2], self.Precipitate.CBtoRB)
                
                beta = np.array([np.matmul( Qbeta, OR_beta[0]),
                                np.matmul( Qbeta, OR_beta[1]),
                                np.matmul( Qbeta, OR_beta[2])]).T

                #
                # Get the final info needed
                #
                FBToA = self.GetDeformationGradient(beta, alpha)
                
                burgers = np.array([np.matmul(np.linalg.inv(np.eye(3)), np.matmul(Qalpha, b)) for b in b0])

                self.FBToA[Interface] = FBToA
                self.b[Interface]     = burgers
                
        except:
            # this error message comes up no matter what the problem is, so its not really unidentified structure maybe?
            raise Exception('Unidentified Crystal Structure')

    #
    # Transformation Functions
    #
    def GetRotationMatrix(self, X, Z, M=np.eye(3)):
        """
        # Functions to get rotation matrix
        # from e to m
        #   
        # sigma_m = Q * sigma_e *Q^T
        # v_m = Q * v_e
        # C' = R \cdot R \cdot C R^T \cdot R^T
        #   
        # Equation 11a - 11b
        # 

        Parameters
        ----------
        X : Vector in the interface plane expressed in local crystallographic coordinates.
        Z : Vector normal to X (interface plane normal) expressed in local crystallgraphic coordinates.
        M : basis transformation matrix

        Returns
        -------
        Rotation matrix

        """
        Minv = np.linalg.inv(M)
        X = np.matmul(Minv,X)
        Z = np.matmul(Minv,Z)

        ETensor           = lambda X, Z: np.array([normalize(X),
                                                normalize(E2Tensor(X,Z)),
                                                normalize(Z)]).T
        E2Tensor          = lambda X, Z: np.matmul(np.eye(3),
                                                np.cross(normalize(Z),
                                                        normalize(X)))

        arr = np.linalg.inv(ETensor(X,Z))
        
        return np.array([normalize(np.matmul(M,arr[0])),
                        normalize(np.matmul(M,arr[1])),
                        normalize(np.matmul(M,arr[2]))])



    def GetDeformationGradient(self, beta, alpha):
        """
        # Function to get deformation gradient
        # alpha = F{B->A} beta
        #
        # Equation 12
        #

        Parameters
        ----------
        beta  : List of three precipitate vectors in the common reference frame.
        alpha : List of three matrix vectors in the common reference frame.

        Returns
        -------
        F_{B->A} : Linear map mapping precipitate vectors to matrix vectors (all in common reference frame).

        """
        return np.dot(alpha,np.linalg.inv(beta))

    def PlaneNormalVector(self, hkl,g):
        """
        Function to calculate Miller indices of a crystal plane normal vector
        
        Parameters
        ----------
        
        hkl : Miller indices of the crystal plane
        g   : Metric tensor for the crystal 
        
        Returns
        -------
        Normal vector
        """

        n = np.matmul(hkl,np.linalg.inv(g))
        
        #return n/np.min(n[np.nonzero(n.round(15))])
        return n/np.min(np.abs(n[np.nonzero(n.round(15))]))
   
    def PlaneSpacing(self, hkl, g):
        """
        Function to calculate the d-spacing of a given lattice plane
        
        Parameters
        ----------
        
        hkl : Miller indices of the crystal plane
        g   : Metric tensor for the crystal 
        
        Returns
        -------
        d-spacing
        """
    
        return np.sqrt(1/np.matmul(hkl,np.matmul(np.linalg.inv(g),hkl)))

    #
    # Individual OR functions 
    #
    """
    Following functions are named after the Orientation Relationship
    They return the components necessary to describe the specific orientation relationship                
       
    Returns
    -------
    OR_alpha : scaled matrix vectors for the OR 
    OR_beta  : scaled precipitate vectors for the OR
    b0       : reference burgers vectors

    """
  
    def Bain(self):
        #
        # Baker - Nutting (Bain) OR
        #
        if 'BCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a_Matrix
            a_FCC = self.a1
            bcc_is = 'matrix'
            fcc_is = 'precip'
        elif 'FCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a1
            a_FCC = self.a_Matrix
            bcc_is = 'precip'
            fcc_is = 'matrix'
        else:
            print('Incorrect crystal stucture for this OR. Must be "BCC" or "FCC"')

        #
        # Parallel Orientations
        #
        
        BCC = np.array([[0.,1.,0.],
                        [1.,1.,0.],
                        [0.,0.,1.]]) 
        FCC = np.array([[1.,1.,0.],
                        [1.,0.,0.],
                        [0.,0.,1.]]) 
        
        if bcc_is == 'matrix':
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                                np.matmul(self.Matrix.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,FCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,FCC[1]),
                                self.PlaneSpacing(FCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(FCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip'] 

            # for Bain, the [1 1 0] direction for the FCC precipitate is scaled by a factor of 1/2. 
            # This accounts for the number of atomic positions that lie in that direction
            OR_beta[0] /= 2.
                                
            b0 = a_BCC*np.array([[0.,1.,0.],[-1.,0.,0.]])
        
        else:
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,FCC[0]),
                                 np.matmul(self.Matrix.CBtoRB,FCC[1]),
                                 self.PlaneSpacing(FCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(FCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix']
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,BCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(BCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip']
            
            OR_alpha[0] /= 2

            b0 = a_FCC/2.*np.array([[1.,1.,0.],[1.,-1.,0.]]) 
        
        return OR_alpha, OR_beta, b0
    
    def KurdjumovSachs(self):

        #
        # Kurdjumov - Sachs OR
        #
        if 'BCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a_Matrix
            a_FCC = self.a1
            bcc_is = 'matrix'
            fcc_is = 'precip'
        elif 'FCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a1
            a_FCC = self.a_Matrix
            bcc_is = 'precip'
            fcc_is = 'matrix'
        else:
            print('Incorrect crystal stucture for this OR. Must be "BCC" or "FCC"')
    
        #
        # Parallel Orientations
        #
        BCC = np.array([[1.,1.,-1.],  
                        [1.,-1.,1.],
                        [0., 1.,1.]]) 
        
        FCC = np.array([[1.,0.,-1.],
                        [0.,-1.,1.],
                        [1.,1.,1.]]) 

    
        if bcc_is == 'matrix':
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                                np.matmul(self.Matrix.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,FCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,FCC[1]),
                                self.PlaneSpacing(FCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(FCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip'] 
            
            OR_alpha[0] /= 2.
            OR_alpha[1] /= 2. # interplanar spacing
            OR_beta[0] /= 2.
            OR_beta[1] /= 2.
            
            b0 = a_BCC*np.array([[0.5,0.5,1.]]).T * np.array([[1.,1.,-1],[1.,-1.,1.],[1.,0.,0.]])                    
    
        else:
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,FCC[0]),
                                 np.matmul(self.Matrix.CBtoRB,FCC[1]),
                                 self.PlaneSpacing(FCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(FCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix']
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,BCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(BCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip']                               

            OR_alpha[0] /= 2.
            OR_alpha[1] /= 2. # interplanar spacing
            OR_beta[0] /= 2.
            OR_beta[1] /= 2.

            b0 = a_FCC/2.*np.array([[-1.,0.,1.],[1.,-1.,0.],[0.,-1.,1.]]) 
    
        return OR_alpha, OR_beta, b0

    def NishiyamaWasserman(self):
    
        #
        # Nishiyama - Wasserman OR
        #
        if 'BCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a_Matrix
            a_FCC = self.a1
            bcc_is = 'matrix'
            fcc_is = 'precip'
        elif 'FCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a1
            a_FCC = self.a_Matrix
            bcc_is = 'precip'
            fcc_is = 'matrix'
        else:
            print('Incorrect crystal stucture for this OR. Must be "BCC" or "FCC"') 
        
        #
        # Parallel Orientations
        #       
        BCC = np.array([[-1.,0.,0.],  
                        [0.,1.,-1.],
                        [0., 1.,1.]]) 
        
        FCC = np.array([[-1.,1.,0.],
                        [1.,1.,-2.],
                        [1.,1.,1.]]) 

    
        if bcc_is == 'matrix':
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                                np.matmul(self.Matrix.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,FCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,FCC[1]),
                                self.PlaneSpacing(FCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(FCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip'] 

            OR_beta[0] /= 2.
            OR_beta[1] /= 2.

            b0 = a_BCC*np.array([[0.5,0.5,1.]]).T * np.array([[1.,1.,-1],[1.,-1.,1.],[1.,0.,0.]])
        
        else:
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,FCC[0]),
                                 np.matmul(self.Matrix.CBtoRB,FCC[1]),
                                 self.PlaneSpacing(FCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(FCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix']
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,BCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(BCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip']

            OR_alpha[0] /= 2.
            OR_alpha[1] /= 2. # interplanar spacing
             

            b0 = a_FCC/2.*np.array([[-1.,0.,1.],[1.,-1.,0.],[0.,-1.,1.]]) 
       
        return OR_alpha, OR_beta, b0
    
    def Pitsch(self):
    
        #
        # Pitsch OR
        #
        if 'BCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a_Matrix
            a_FCC = self.a1
            bcc_is = 'matrix'
            fcc_is = 'precip'
        elif 'FCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a1
            a_FCC = self.a_Matrix
            bcc_is = 'precip'
            fcc_is = 'matrix'
        else:
            print('Incorrect crystal stucture for this OR. Must be "BCC" or "FCC"')
    
        #
        # Parallel Orienations
        #
        BCC = np.array([[1.,1.,-1.],  
                        [1.,-1.,1.],
                        [0., 1.,1.]]) 


        FCC = np.array([[0.,1.,-1.],
                        [0.,-1.,-1.],
                        [1.,0.,0.]]) 

    
        if bcc_is == 'matrix':
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                                np.matmul(self.Matrix.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,FCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,FCC[1]),
                                self.PlaneSpacing(FCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(FCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip'] 
            OR_alpha[0] /= 2.
            OR_alpha[1] /= 2.
            
            OR_beta[0] /= 2.
            OR_beta[1] /= 2.

            b0 = a_BCC/2 * np.array([[1.,1.,-1.],[1.,-1.,1.]])
        
        else:
            OR_alpha = np.array([np.matmul(self.MatrixCBtoRB,FCC[0]),
                                 np.matmul(self.Matrix.CBtoRB,FCC[1]),
                                 self.PlaneSpacing(FCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(FCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix']
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,BCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(BCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip']
            
            OR_alpha[0] /= 2.
            OR_alpha[1] /= 2.
            
            OR_beta[0] /= 2.
            OR_beta[1] /= 2.

            b0 = a_FCC/2*np.array([[0.,1.,-1.], [0.,-1.,-1.]]) 
    
        return OR_alpha, OR_beta, b0
    
    def Isaichev(self):
        
        a_BCC = self.a_Matrix

        self.Precipitate.g = TensorSort(self.Precipitate.g)
        self.Precipitate.CBtoRB = TensorSort(self.Precipitate.CBtoRB)

        # 
        # Parallel Orientations
        #
        BCC = np.array([[1., 1., 1.], 
                        [0., -1., 1.], 
                        [2., -1., -1.]])  
                       
        Ortho = np.array([[0.,  1., 0.],
                          [-1., 0., -1.],
                          [-1., 0., 1.]])
        
        OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                            np.matmul(self.Matrix.CBtoRB,BCC[1]),
                            self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                            ]) * self.SubUnitDict['matrix'] 
        # OR_alpha[0] /= 2                    
        
        OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,Ortho[0]),
                            np.matmul(self.Precipitate.CBtoRB,Ortho[1]),
                            self.PlaneSpacing(Ortho[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(Ortho[2],self.Precipitate.g)))
                            ]) * self.SubUnitDict['precip'] 
    
        #   
        # Burgers vectors in BCC Reference Frame
        #   
        b0 = a_BCC*np.array([[1.,1.,1.],[0.,-1.,1.]])/np.array([[2],[1]])
    
        return OR_alpha, OR_beta, b0
    
    def Bagaryatski(self):

        a_BCC = self.a_Matrix

        self.Precipitate.g = TensorSort(self.Precipitate.g)
        self.Precipitate.CBtoRB = TensorSort(self.Precipitate.CBtoRB)
        
        #
        # Parallel Orientations
        #
        BCC = np.array([[1., 1., 1.], 
                        [1., -1., 0.],
                        [1., 1., -2.]])
                        
        Ortho = np.array([[0., 1., 0.],
                          [1., 0., 0.],
                          [0., 0., -1.]])
                          
        OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                            np.matmul(self.Matrix.CBtoRB,BCC[1]),
                            self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                            ]) * self.SubUnitDict['matrix'] 
        # OR_alpha[0] /= 2
                                           
        OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,Ortho[0]),
                            np.matmul(self.Precipitate.CBtoRB,Ortho[1]),
                            self.PlaneSpacing(Ortho[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(Ortho[2],self.Precipitate.g)))
                            ]) * self.SubUnitDict['precip']       
                          
    
        #   
        # Burgers vectors in BCC Reference Frame
        #   
        b0 = a_BCC*np.array([[1.,1.,1.],[1.,-1.,0.]])/np.array([[2],[1]])
       
        return OR_alpha, OR_beta, b0
    
    def Burgers(self):
    
        #
        # Burgers OR
        #
        if 'BCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a_Matrix
            a_Hex = self.a1
            c_Hex = self.a3 
            bcc_is = 'matrix'
            hcp_is = 'precip'
        elif 'HCP' in self.Matrix.CrystalStructure :
            a_BCC = self.a1
            a_Hex = self.a_Matrix
            c_Hex = self.c_Matrix
            bcc_is = 'precip'
            hcp_is = 'matrix'
        else:
            print('Incorrect crystal stucture for this OR. Must be "BCC" or "HCP"')
        
        #
        # Parallel Orientations
        #       
        BCC = np.array([[1., -1., 1.],
                        [1., 1., -1.],
                        [0., 1., 1.]]) 
                              
        HCP = np.array([ConvertHCP4toHCP3(np.array([1., 1., -2., 0.])),
                        ConvertHCP4toHCP3(np.array([-2., 1., 1., 0.])),
                        ConvertHCP4toHCP3(np.array([0., 0., 0., 1.]))])

        #
        # 
        #
        if bcc_is == 'matrix':
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                                np.matmul(self.Matrix.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 

            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,HCP[0]),
                                np.matmul(self.Precipitate.CBtoRB,HCP[1]),
                                self.PlaneSpacing(HCP[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(HCP[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip'] 
            
            OR_alpha[0] /= 2.
            OR_alpha[1] /= 2.

            b0 = a_BCC * np.array([[1.,-1.,1.],[1.,1.,-1.],[1.,0.,0.]]) / np.array([[2.,2.,1.]])
            
        else:
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,HCP[0]),
                                np.matmul(self.Matrix.CBtoRB,HCP[1]),
                                self.PlaneSpacing(HCP[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(HCP[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,BCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(BCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip'] 
            
            OR_beta[0] /= 2.
            OR_beta[1] /= 2.

            b0 = a_Hex * np.array([ConvertHCP4toHCP3(np.array([1.,1.,-2.,0])),
                                   ConvertHCP4toHCP3(np.array([-2.,1.,1.,0])),
                                   ConvertHCP4toHCP3(np.array([1.,-2.,1.,0]))])

        return OR_alpha, OR_beta, b0
    
    def PitschSchrader(self):

        #
        # Pitsch - Shrader OR
        #
        if 'BCC' in self.Matrix.CrystalStructure :
            a_BCC = self.a_Matrix
            a_Hex = self.a1
            c_Hex = self.a3 
            bcc_is = 'matrix'
            hcp_is = 'precip'
        elif 'HCP' in self.Matrix.CrystalStructure :
            a_BCC = self.a1
            a_Hex = self.a_Matrix
            c_Hex = self.c_Matrix
            bcc_is = 'precip'
            hcp_is = 'matrix'
        else:
            print('Incorrect crystal stucture for this OR. Must be "BCC" or "HCP"')
        
        #
        # Parallel Orientations
        #
       
        BCC = np.array([[1., -1., 0.],
                        [1., -1., 1.],
                        [1., 1., 0.]])
                        
        HCP = np.array([ConvertHCP4toHCP3(np.array([1., -1., 0., 0.])),
                        ConvertHCP4toHCP3(np.array([1., -2., 1., 0.])),
                        ConvertHCP4toHCP3(np.array([0., 0., 0., 1.]))])
                        
        #
        # 
        #
        if bcc_is == 'matrix':
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                                np.matmul(self.Matrix.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 

            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,HCP[0]),
                                np.matmul(self.Precipitate.CBtoRB,HCP[1]),
                                self.PlaneSpacing(HCP[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(HCP[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip']  
            
            OR_alpha[1] /= 2.

            b0 = a_BCC * np.array([[1.,-1.,-1.],[1.,-1.,1.],[0,0.,1.]]) / np.array([[2.,2.,1.]])
            
        else:
            OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,HCP[0]),
                                np.matmul(self.Matrix.CBtoRB,HCP[1]),
                                self.PlaneSpacing(HCP[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(HCP[2],self.Matrix.g)))
                                ]) * self.SubUnitDict['matrix'] 
            OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,BCC[0]),
                                np.matmul(self.Precipitate.CBtoRB,BCC[1]),
                                self.PlaneSpacing(BCC[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(BCC[2],self.Precipitate.g)))
                                ]) * self.SubUnitDict['precip']   
            
            OR_beta[1] /= 2.

            b0 = a_Hex * np.array([ConvertHCP4toHCP3(np.array([1.,1.,-2.,0])),
                                   ConvertHCP4toHCP3(np.array([-2.,1.,1.,0])),
                                   ConvertHCP4toHCP3(np.array([1.,-2.,1.,0]))])   
    
    
        return OR_alpha, OR_beta, b0
    
    



#%% Interface Energy Class
class NonSingularIsotropicBimaterialInterfaceEnergy():
    """
    Calculates elastic interfacial energy assuming (i) isotopic elastic constants (shear modulus, Poissons ratio) and (ii) employing the non-singular dislocation formulation. Within this formulation the Burgers vector of each infinitesimal segment of dislocation is distributed throughout a finite volume surrounding the dislocation core. This is in contrast to discrete dislocations, within which Burgers vectors are represented by heaviside functions. The conventional core cut-off length scale r0 is replaced by a characteristic distribution length scale aNS. In this work, we employ the minimum Burgers vector length in the reference state as the core distribution length scale, i.e. aNS = Afactor * min(b1_reference, b2_reference) where Afactor is 1. Far field isotropic bimaterial linear elastic stress calcuations are incorporated following Lubarda et al.

    Cai, W. et al. "A non-singular continuum theory of dislocations," Journal of the Mechanics and Physics of Solids. Volume 54, pp 561 - 587. 2006.

    Lubarda, V. A. "Energy Analysis of Dislocation Arrays Near Bimaterial Interfaces," Int. J. Solids and Structures. Volume 34, pp 1053 - 1073. 1997.

    Lubarda, V. A., Kouris D. A. "Stress fields due to dislocation arrays at interfaces," Mechanics of Materials. Volume 23, pp 191 - 203. 1996.

    Attributes (initialized)
    ----------
    muMatrix         : shear modulus of matrix
    nuMatrix         : Poisson's ratio of matrix 
    muPrecipitate    : shear modulus of precipitate
    nuPrecipitate    : Poisson's ratio of precipitate
    FBToA            : Transformation matrix from the precipitate to the matrix
    b1, b2           : Burgers vectors in natural matrix lattice.
    Afactor          : prefactor for characteristic distribution lengthscale, a_NS = Afactor * b. Default is 1.
    
    Attributes (calculated)
    ----------
    delta            : Partition factor between deformation of the matrix and deformation of the precipitate into the shared reference state (0 < \delta < 1). Under mappings (1.-delta)*I + delta*FBToA and delta*I + (1.-delta)*Inv(FBToA), respectively, the matrix and precipitate  are both mapped into their natural relaxed states. 
    InterfaceEnergy  : Computed interface energy (Equation 3)
    AnalyticalEnergy : Asymptotic discrete interface energy (Equation 19)
    d1, d2           : Dislocation spacing for dislocation arrays 1 and 2.
    
    """

    def __init__(self, MatrixPhase, PrecipPhase, Interface, ORinfo, Afactor=1.):
        """
        Initialize elastic interfacial energy calculation.

        Attributes (initialized)
        ----------
        muMatrix         : shear modulus of matrix (in Pa)
        muPrecipitate    : shear modulus of precipitate (in Pa)
        nuMatrix         : Poisson's ratio of matrix
        nuPrecipitate    : Poisson's ratio of precipitate
        FBToA            : Transformation matrix from the precipitate to the matrix

        Attributes (calculated)
        ----------
        delta            : partition between matrix and precipitate deformation (see above).
        xi1, xi2         : Line directions for dislocation arrays 1 and 2.
        b1, b2           : Burgers vectors for dislocation arrays 1 and 2 in the reference state.
        d1, d2           : Dislocation spacing for dislocation arrays 1 and 2.
        InterfaceEnergy  : Computed interface energy (initialized to False).

        Parameters
        ----------
        MatrixPhase : Instance of CrystalProperties class describing matrix.
        PrecipPhase : Instance of CrystalProperties class describing precipitate.
        Interface   : Name of Orientation Relationship.
        ORinfo      : Instance of CrystalInterface class describing the orientation relationship.
        Afactor     : prefactor for characteristic distribution lengthscale, a_NS = Afactor * b. Default is 1.

        """
        
        #
        # initialize data
        #
        
        #
        # Matrix elastic constants
        #
        self.muMatrix = MatrixPhase.ShearModulus * GigaPascal
        self.nuMatrix = MatrixPhase.Poisson

        #
        # Precipitate elastic constants
        #
        self.muPrecipitate = PrecipPhase.ShearModulus * GigaPascal
        self.nuPrecipitate = PrecipPhase.Poisson

        #
        # Interface properties
        #
        self.FBToA = ORinfo.FBToA[Interface]

        #
        # try each combination of Burgers vectors
        #
        #SurfaceDislocationDensity = 1.
        for (i,j) in it.combinations(range(len(ORinfo.b[Interface])),2):
            
            self.b1 = ORinfo.b[Interface][i]
            self.b2 = ORinfo.b[Interface][j]

            #
            # compute partition between matrix and precipitate deformation
            #
            self.delta = self.EnergyMinimizer().x

            #
            # set trial xi, b, d (Equations 1 and 2)
            #
            self.Afactor = Afactor
            self.SetLineDirectionsAndSpacings(self.delta)
            self.SetBurgersComponents()

            #
            # Store minimum Surface Dislocation Density index
            #
            if 'SurfaceDislocationDensity' in locals():
                if self.SurfaceDislocationDensity*1.e-10 < SurfaceDislocationDensity:
                    SurfaceDislocationDensity = self.SurfaceDislocationDensity*1.e-10
                    imin = i
                    jmin = j
            else:
                SurfaceDislocationDensity  = self.SurfaceDislocationDensity*1.e-10
                imin = i
                jmin = j
 

        #
        # set xi, b, d
        #
        self.b1 = ORinfo.b[Interface][imin]
        self.b2 = ORinfo.b[Interface][jmin]
        self.delta = self.EnergyMinimizer().x
        self.SetLineDirectionsAndSpacings(self.delta)
        self.SetBurgersComponents()

        #
        # initialize Interface Energy (Equation 3)
        #
        self._InterfaceEnergy = False

        return

    @property
    def InterfaceEnergy(self):
        """
        Returns interface energy
        """
        if not bool(self._InterfaceEnergy):
            #
            # Naive search / convergence algorithm 
            #
            eps = 99.
            self._InterfaceEnergy = self.ComputeEnergy(dIP = 1)
            l = 10
            while eps > 0.001:
                gammaOld = self._InterfaceEnergy
                self._InterfaceEnergy = self.ComputeEnergy(dIP = l)
                
                #
                # compute relative error
                #
                eps = abs(self._InterfaceEnergy - gammaOld)/abs(gammaOld)
                l = int(l*2.)

        #
        # Equation 3
        #
        return self._InterfaceEnergy
    @property
    def SurfaceDislocationDensity(self):
        """
        Returns surface dislocation density
        """
        return (self._d1+self._d2)/(self._d1 * self._d2) 
    def GetDelta(self):
        """
        Returns delta parameter (deformation partition between matrix and precipitate).
        """
        return self.delta
    @property
    def GetDs(self):
        """
        Returns dislocation spacing for each of two parallel dislocation arrays.
        """
        return [self._d1,self._d2]
    @property
    def FrankBilby(self):
        """
        Returns Frank-Bilby Equation.
        """
        return self._FB
    @property
    def AsymptoticAnalyticalInterfaceEnergy(self):
        """
        Returns Asymptotic Analytical Interface Energy.
        """
        r0 = np.min([np.linalg.norm(self.b1ref),np.linalg.norm(self.b2ref)])

        Norm = np.array([0.,0.,1.])
        SinTheta = np.linalg.norm(np.cross(self.xi1,self.xi2))
        CotTheta = np.dot(self.xi1,self.xi2)/SinTheta
        xi = np.array([self.xi1, self.xi2])
        d = np.array([self._d1, self._d2])
        bref = np.array([self.b1ref, self.b2ref])

        bscrew = np.array([self.bscrew1, self.bscrew2])
        bedge = np.array([self.bedge1, self.bedge2])

        k4 = (self.muPrecipitate - self.muMatrix)/(self.muPrecipitate + self.muMatrix)
        epsilon = self.muPrecipitate*(1.-k4)

        Gamma = self.muMatrix/self.muPrecipitate
        k1 = 3.-4.*self.nuPrecipitate
        k2 = 3.-4.*self.nuMatrix
        alpha = (Gamma*(k1+1.) - k2 - 1.)/(Gamma*(k1+1.)+k2 + 1.)
        beta = (Gamma * (k1-1.) - k2 + 1.)/(Gamma*(k1+1.) + k2 + 1.)
        mu_precipitate_star = self.muPrecipitate*(1.+alpha)/(1.-beta*beta)
        k0 = mu_precipitate_star/2./np.pi/(1.-self.nuPrecipitate)

        self._AsymptoticAnalyticalInterfaceEnergy = 0.0

        for idx in range(2):
            for jdx in range(2):

                m1 = xi[jdx]
                m2 = np.cross(Norm,xi[jdx])
                m3 = Norm

                e1 = np.array([1.,0.,0.])
                e2 = np.array([0.,1.,0.])
                e3 = np.array([0.,0.,1.])

                Q = np.array([[np.dot(m1,e1),np.dot(m2,e1),np.dot(m3,e1)],[np.dot(m1,e2),np.dot(m2,e2),np.dot(m3,e2)], [np.dot(m1,e3),np.dot(m2,e3),np.dot(m3,e3)]])

                sigmaLocal = np.array([[CotTheta*-epsilon*bscrew[jdx], CotTheta*mu_precipitate_star*bedge[jdx]/(1.0-self.nuPrecipitate), 0.],[-epsilon*bscrew[jdx], mu_precipitate_star*bedge[jdx]/(1.0-self.nuPrecipitate), 0.],[0., 0., 0.]])*np.log(d[jdx]/2./np.pi/r0/SinTheta)

                sigmaGlobal = np.matmul(np.matmul(Q,sigmaLocal),Q.T)

                self._AsymptoticAnalyticalInterfaceEnergy += 0.5* np.tensordot(np.outer(np.cross(Norm,xi[idx]),bref[idx])/2./np.pi/d[idx],(sigmaGlobal),axes=2)

        return self._AsymptoticAnalyticalInterfaceEnergy

    def PlotDisplacement(self, Field='ux'):
        """
        Plot Displacement

        returns a contour plot of the displacement field along the interface. The displacement field includes the superposition of the homogeneous linear displacment and the heaviside dislocation induced displacement fields. The two relevant in-plane displacement components are Z[0,x,y] = u_{x} and Z[1,x,y] = u_{y}. An example call to plot is shown as CS = ax.contourf(X/Angs, Y/Angs, Z[1],levels = 55).
        """

        if Field not in ['ux', 'uy']:
            print('usage: Object.PlotDisplacement(Field)\n')
            print('Field options are "ux" or "uy"')
            return

        #
        # Get Line Directions And Spacings
        #
        self.SetLineDirectionsAndSpacings(self.delta)

        #
        # Decompose Burgers vectors into edge and screw
        #
        self.SetBurgersComponents()

        #
        # Compute Affine Displacement
        #
        U_affine = self.GetReferenceState(self.F_Precipitate, self.F_Matrix)

        #
        # Plot Settings
        #
        WindowLen = 6.*max(self._d1,self._d2)
        Angs = 1.e-10
        N = 25
        MinLineLength = min(self._d1,self._d2)/N

        x = np.arange(-WindowLen/2., WindowLen/2., MinLineLength)
        y = np.arange(-WindowLen/2., WindowLen/2., MinLineLength)

        Xrange = len(x)
        Yrange = len(y)

        X, Y = np.meshgrid(x, y)
        Z = np.array([X*0. for dim in range(3)])

        for x in range(Xrange):
            for y in range(Yrange):

                #
                # Linear displacement due to homogeneous (constant) strain
                #
                Z[0, x,y] = -(np.dot(U_affine,np.array([X[x,y],Y[x,y],0.]))[0]  - (self.b1ref + self.b2ref)[0]/2.)
                Z[1, x,y] = -(np.dot(U_affine,np.array([X[x,y],Y[x,y],0.]))[1]  - (self.b1ref + self.b2ref)[1]/2.)
                Z[2, x,y] = -(np.dot(U_affine,np.array([X[x,y],Y[x,y],0.]))[2]  - (self.b1ref + self.b2ref)[2]/2.)

                for Disl in range(2):

                    #
                    # Displacement Field At Point
                    #
                    u = self.ComputeDisplacementFieldAtPoint(X[x,y], Y[x,y], [self.b1ref, self.b2ref][Disl], [self.xi1, self.xi2][Disl], [self._d1, self._d2][Disl])                

                    #
                    # Equation 9
                    #
                    Z[0, x,y] -= u[0]
                    Z[1, x,y] -= u[1]
                    Z[2, x,y] -= u[2] # should be zero

        #
        # Plot Displacement
        #
        fig, ax = plt.subplots()
        u = Z[0] if Field == 'ux' else Z[1]
        CS = ax.contourf(X/Angs, Y/Angs, u, levels = 55)
        fig.colorbar(CS, label = "Meters")
        plt.show()
        plt.clf()

        return

    def PlotStress(self,Field='Sxz'):
        """
        Plot Stress

        returns a contour plot of the stress field along the interface. The stress field includes the superposition homogeneous coherency stress and the heterogenoues dislocation stress fields. The two relevant shear components are Z[3,x,y] = sigma_{yz} and Z[4,x,y]= sigma_{xz}. An example call to plot is shown as CS = ax.contourf(X/Angs, Y/Angs, Z[4],levels = 55). 
        """

        if Field not in ['Sxz', 'Syz']:
            print('usage: Object.PlotStress(Field)\n')
            print('Field options are "Sxz" or "Syz"')
            return

        #
        # Get Line Directions And Spacings
        #
        self.SetLineDirectionsAndSpacings(self.delta)

        #
        # Decompose Burgers vectors into edge and screw
        #
        self.SetBurgersComponents()

        #
        # Get Far Field Dislocation Stress Matrix
        #
        sigmaD_Matrix = self.GetDislocationStressMatrix(PrecipitateBool=False)
        sigmaD_Precipitate = self.GetDislocationStressMatrix(PrecipitateBool=True)

        #
        # Plot Settings
        #
        WindowLen = 6.*max(self._d1,self._d2)
        Angs = 1.e-10
        N = 25
        MinLineLength = min(self._d1,self._d2)/N

        x = np.arange(-WindowLen/2., WindowLen/2., MinLineLength)
        y = np.arange(-WindowLen/2., WindowLen/2., MinLineLength)

        Xrange = len(x)
        Yrange = len(y)
        X, Y = np.meshgrid(x, y)
        Z = np.array([X*0. for dim in range(6)])

        for x in range(Xrange):
            for y in range(Yrange):

                #
                # Homogeneous Coherency Stress is Equal and Opposite to Far - Field Dislocation Stress (eq. 2)
                #
                Z[0, x,y] = - sigmaD_Precipitate[0,0]
                Z[1, x,y] = - sigmaD_Precipitate[1,1]
                Z[2, x,y] = - sigmaD_Precipitate[2,2]
                Z[3, x,y] = - sigmaD_Precipitate[1,2]
                Z[4, x,y] = - sigmaD_Precipitate[0,2]
                Z[5, x,y] = - sigmaD_Precipitate[0,1]

                for Disl in range(2):

                    #
                    # Stress Field At Point
                    #
                    aNS = self.Afactor * np.min([np.linalg.norm(self.b1ref),np.linalg.norm(self.b2ref)])
                    sigma = self.ComputeStressFieldAtPoint(aNS, X[x,y], Y[x,y], [self.bedge1,self.bedge2][Disl], [self.bscrew1,self.bscrew2][Disl], [self._d1,self._d2][Disl], [self.xi1,self.xi2][Disl])
                    Z[0, x,y] += sigma[0,0] # irrelevant
                    Z[1, x,y] += sigma[1,1] # irrelevant
                    Z[2, x,y] += sigma[2,2] # irrelevant
                    Z[3, x,y] += sigma[1,2] # important (shear)
                    Z[4, x,y] += sigma[0,2] # important (shear)
                    Z[5, x,y] += sigma[0,1] # shear not on surface

        #
        # Plot Stress
        #
        fig, ax = plt.subplots()
        u = Z[4] if Field == 'Sxz' else Z[3]
        CS = ax.contourf(X/Angs, Y/Angs, u, levels = 55)
        fig.colorbar(CS, label = "Pa")
        plt.show()
        plt.clf()

        return

    def PlotEnergyDensity(self):
        """
        Plot Energy Density

        returns a contour plot of the energy density field along the interface. The energy density field includes the superposition of the homogeneous elastic strain energy and the heterogenoues dislocation induced strain energy. An example call to plot is shown as CS = ax.contourf(X/Angs, Y/Angs, Z, levels = 55). 
        """
        #
        # Get InterfaceEnergyDensity
        #
        EnergyDensity = self.GetEnergyDensityField()
        X, Y, Z = EnergyDensity['X'], EnergyDensity['Y'], EnergyDensity['Z']

        #
        # Plot Settings
        #
        Angs = 1.e-10

        #
        # Plot Energy Density
        #
        fig, ax = plt.subplots()
        CS = ax.contourf(X/Angs, Y/Angs, Z, levels = 55)
        fig.colorbar(CS, label = r'Misfit Energy Density [$J/m^2$]')
        plt.show()
        plt.clf()

        return
    
    def ComputeEnergy(self,dIP = 100):
        
        #
        # Get Energy Density
        #
        EnergyDensity = self.GetEnergyDensityField(dIP = dIP)
        Z = EnergyDensity['Z']

        #
        # Initialize integral to 0.
        #
        U = 0.
        NumbPoints = int(dIP**2)

        #
        # Integrate NumbPoints over Z to compute the interface energy
        #
        for x in range(dIP):
            for y in range(dIP):

                U += Z[x,y]/2.

        #
        # equation (3)
        #
        return U/NumbPoints

    def GetEnergyDensityField(self, dIP = 100):

        #
        # instantiate Energy Field Dictionary
        #
        EnergyField = {}

        #
        # Get Line Directions and Spacings
        #
        self.SetLineDirectionsAndSpacings(self.delta)

        #
        # Decompose Burgers vectors into edge and screw
        # 
        self.SetBurgersComponents()

        #
        # Get Far Field Dislocation Stress Matrix
        #
        sigmaD_Matrix = self.GetDislocationStressMatrix(PrecipitateBool=False)
        sigmaD_Precipitate = self.GetDislocationStressMatrix(PrecipitateBool=True)

        #
        # Compute Affine Displacement
        #
        U_affine = self.GetReferenceState(self.F_Precipitate, self.F_Matrix)

        #
        # Numerical Settings
        #
        X, Y, x_orig, y_orig, phi_range, eta_range = self.GetNumericalSettings(dIP = dIP)

        V = np.array([X*0. for dim in range(3)])
        W = np.array([X*0. for dim in range(6)])
        Z = X*0.

        for x in range(len(phi_range)):
            for y in range(len(eta_range)):

                #    
                # Linear displacement due to homogeneous (constant) strain
                # 
                V[0, x,y] = -(np.dot(U_affine,np.array([X[x,y],Y[x,y],0.]))[0] - (self.b1ref + self.b2ref)[0]/2.)
                V[1, x,y] = -(np.dot(U_affine,np.array([X[x,y],Y[x,y],0.]))[1]  - (self.b1ref + self.b2ref)[1]/2.)
                V[2, x,y] = -(np.dot(U_affine,np.array([X[x,y],Y[x,y],0.]))[2]  - (self.b1ref + self.b2ref)[2]/2.)

                #
                # Homogeneous Coherency Stress is Equal and Opposite to Far - Field Dislocation Stress (eq. 2)
                #
                W[0, x,y] = - sigmaD_Precipitate[0,0]
                W[1, x,y] = - sigmaD_Precipitate[1,1]
                W[2, x,y] = - sigmaD_Precipitate[2,2]
                W[3, x,y] = - sigmaD_Precipitate[1,2]
                W[4, x,y] = - sigmaD_Precipitate[0,2]
                W[5, x,y] = - sigmaD_Precipitate[0,1]

                for Disl in range(2):

                    #
                    # Displacement Field At Point
                    #
                    u = self.ComputeDisplacementFieldAtPoint(X[x,y], Y[x,y], [self.b1ref, self.b2ref][Disl], [self.xi1, self.xi2][Disl], [self._d1, self._d2][Disl])
                    
                    #
                    # Equation 9
                    #
                    V[0, x,y] -= u[0]
                    V[1, x,y] -= u[1]
                    V[2, x,y] -= u[2]

                    #
                    # Stress Field At Point
                    #
                    aNS = self.Afactor * np.min([np.linalg.norm(self.b1ref),np.linalg.norm(self.b2ref)])
                    sigma = self.ComputeStressFieldAtPoint(aNS, X[x,y], Y[x,y], [self.bedge1, self.bedge2][Disl], [self.bscrew1, self.bscrew2][Disl], [self._d1,self._d2][Disl], [self.xi1,self.xi2][Disl])
                    W[0, x,y] += sigma[0,0] # irrelevant
                    W[1, x,y] += sigma[1,1] # irrelevant
                    W[2, x,y] += sigma[2,2] # irrelevant
                    W[3, x,y] += sigma[1,2] # important
                    W[4, x,y] += sigma[0,2] # important
                    W[5, x,y] += sigma[0,1] # irrelevant

                Z[x,y] = (V[0,x,y]*W[4,x,y] + V[1,x,y]*W[3,x,y] + V[2,x,y]*W[2,x,y])

        EnergyField['X'], EnergyField['Y'], EnergyField['Z'] = X, Y, Z

        return EnergyField

    def ComputeDisplacementFieldAtPoint(self,x,y,bref,xi,d):

        #
        # Rotate point into dislocation coordinate system (from m to e)
        #
        X, Y, Z = np.dot(self.RotationMatrix[tuple(xi)],np.array([x,y,0.]))

        #
        # Calculate number of discrete displacements
        #
        N = math.ceil(Y/d)

        #
        # return displacement Vector
        #
        
        #
        # Equation 10
        #
        return N*bref

    def ComputeStressFieldAtPoint(self,aNS,x,y,bedge,bscrew,d,xi):

        #
        # Rotate point into dislocation coordinate system (from m to e)
        #
        X, Y, Z = np.dot(self.RotationMatrix[tuple(xi)],np.array([x,y,0.]))

        #
        # Get Stress
        #
        sigma = self.LubardaStressFields(aNS,X,Y,d,bedge,bscrew)

        #
        # Rotate point into global coordinate system
        #
        return self.RotateStress(sigma,xi)

    def GetNumericalSettings(self, dIP = 100):

        phi0 = np.array([0.,0.,0.])
        eta0 = np.array([0.,0.,0.])
        SinTheta = abs(np.dot(np.array([0.,0.,1.]),np.cross(self.xi1,self.xi2)))
        phif = np.array(self.xi1)*self._d2/SinTheta
        etaf = np.array(self.xi2)*self._d1/SinTheta

        DiffPhi = (np.linalg.norm(phif - phi0))/(dIP*2)
        DiffEta = (np.linalg.norm(etaf - eta0))/(dIP*2)
        phi_range = np.linspace(np.linalg.norm(phi0)+DiffPhi, np.linalg.norm(phif)-DiffPhi, dIP)
        eta_range = np.linspace(np.linalg.norm(eta0)+DiffEta, np.linalg.norm(etaf)-DiffEta, dIP)

        X = np.array([[np.dot(phi*np.array(self.xi1)+eta*np.array(self.xi2), np.array([1.,0.,0.])) for phi in phi_range] for eta in eta_range])
        Y = np.array([[np.dot(phi*np.array(self.xi1)+eta*np.array(self.xi2), np.array([0.,1.,0.])) for phi in phi_range] for eta in eta_range])
        x_orig = X[int(X.shape[0]/2.),int(X.shape[1]/2.)]
        y_orig = Y[int(Y.shape[0]/2.),int(Y.shape[1]/2.)]

        return X, Y, x_orig, y_orig, phi_range, eta_range

    def EnergyMinimizer(self):
        return minimize_scalar(self.ComputeFarFieldStressError, bounds=(0,1), method = 'bounded')

    def ComputeFarFieldStressError(self,x):

        #
        # Get Line Directions and Spacings and trial deformation gradients
        #
        self.SetLineDirectionsAndSpacings(x)

        #
        # Decompose Burgers Vectors into edge and screw
        #
        self.SetBurgersComponents()

        #
        # Get Coherency Stresses (Deprecated)
        #
        ### sigmaC_Precipitate = GetCoherencyStressMatrix(F_Precipitate,mu_Precipitate,nu_Precipitate)
        ### sigmaC_Matrix = GetCoherencyStressMatrix(F_Matrix,mu_Matrix,nu_Matrix)

        #
        # Get Far Field Dislocation Stress Fields
        #
        sigmaD_Precipitate = self.GetDislocationStressMatrix(PrecipitateBool=True)
        sigmaD_Matrix = self.GetDislocationStressMatrix(PrecipitateBool=False)

        #
        # Get linearized symmetric deformation gradients for coherency strain and dislocation - induced strain
        #

        #
        # Equation 6
        #
        EC_Precipitate = 1./2.*(np.linalg.inv(self.F_Precipitate) + np.linalg.inv(self.F_Precipitate).transpose() - 2.*np.eye(3))
        EC_Matrix = 1./2.*(np.linalg.inv(self.F_Matrix) + np.linalg.inv(self.F_Matrix).transpose() - 2.*np.eye(3))
        ED_Precipitate = self.GetDislocationStrainMatrix(sigmaD_Precipitate,self.muPrecipitate,self.nuPrecipitate)
        ED_Matrix = self.GetDislocationStrainMatrix(sigmaD_Matrix,self.muMatrix,self.nuMatrix)

        #
        # Objective is to minimize E_{Dislocation} + E_{Coherency} (strains) for both matrix and precipitate with respect to delta (partition between matrix and precipitate deformation).
        #

        #
        # Notes:
        #       - Employ MSE
        #       - Matrix is bottom; Precipitate is top (consistent with figure).
        #       - axial and shear deformations in interface plane are counted.
        #
        MSE = 0.
        CoherencyStrains = np.array([EC_Precipitate[0,0], EC_Precipitate[1,1], EC_Precipitate[0,1], EC_Precipitate[1,0], EC_Matrix[0,0], EC_Matrix[1,1], EC_Matrix[0,1], EC_Matrix[1,0]])
        DislocationStrains = np.array([ED_Precipitate[0,0], ED_Precipitate[1,1], ED_Precipitate[0,1], ED_Precipitate[1,0], ED_Matrix[0,0], ED_Matrix[1,1], ED_Matrix[0,1], ED_Matrix[1,0]])
        
        #
        # Right hand side of Equation 8
        #
        for field in CoherencyStrains + DislocationStrains:

            MSE += field**2.

        #
        # return Equation 8. Solves Equations 2 and 7
        #
        return MSE

    def SetLineDirectionsAndSpacings(self,x):

        #
        # compute trial deformation gradients at delta
        #
        
        #
        # Equation 4a
        #
        self.F_Matrix = (1.-x)*np.eye(3) + x*self.FBToA

        #
        # Equation 4b
        #
        self.F_Precipitate = x*np.eye(3) + (1.-x)*np.linalg.inv(self.FBToA)

        #
        # Reference State
        #
        T = self.GetReferenceState(self.F_Precipitate, self.F_Matrix)

        #
        # Get Burgers vectors in Reference State
        #
        self.b1ref = np.dot(np.linalg.inv(self.F_Matrix),self.b1)
        self.b2ref = np.dot(np.linalg.inv(self.F_Matrix),self.b2)

        #
        # Identify line directions and dislocation line spacings
        #

        #
        # Equation 5a
        #
        self.xi1 = np.append(self.NormalizeVector(np.dot(np.linalg.inv(T[:2,:2]), self.NormalizeVector(self.b2ref[:2]))),0.)

        #
        # Equation 5b
        #
        self.xi2 = np.append(self.NormalizeVector(np.dot(np.linalg.inv(T[:2,:2]), self.NormalizeVector(self.b1ref[:2]))),0.)

        #
        # Get Dislocation Spacing(s)
        #
        self._d1 = np.dot(self.GetDirection2(self.xi1,self.xi2),[0.,0.,1.])*np.dot(self.b1ref,self.b1ref)/np.dot(np.dot(T,self.xi2),self.b1ref)
        self._d2 = np.dot(self.GetDirection2(self.xi2,self.xi1),[0.,0.,1.])*np.dot(self.b2ref,self.b2ref)/np.dot(np.dot(T,self.xi1),self.b2ref)

        if self._d1 < 0.:
            self.xi1 = -self.xi1
            self._d1 = np.dot(self.GetDirection2(self.xi1,self.xi2),[0.,0.,1.])*np.dot(self.b1ref,self.b1ref)/np.dot(np.dot(T,self.xi2),self.b1ref)

        if self._d2 < 0.:
            self.xi2 = -self.xi2
            self._d2 = np.dot(self.GetDirection2(self.xi2,self.xi1),[0.,0.,1.])*np.dot(self.b2ref,self.b2ref)/np.dot(np.dot(T,self.xi1),self.b2ref)

        #
        # Set Dislocation Rotation Matrix
        #
        self.SetDislocationRotationMatrix()
        
        #
        # Check F-B equation
        #
        p = np.array([np.sqrt(2.)/2.,np.sqrt(2.)/2.,0.])
        n = np.array([0.,0.,1.])
        
        #
        # Equation 1
        #
        self._FB = np.dot(np.cross(n,self.xi1),p)*self.b1ref/self._d1 + np.dot(np.cross(n,self.xi2),p)*self.b2ref/self._d2 - np.dot(T,p)

        return

    def GetReferenceState(self,F1,F2):
        return np.linalg.inv(F1) - np.linalg.inv(F2)

    def NormalizeVector(self,Vector):
        return np.array([v/np.linalg.norm(Vector) for v in Vector])

    def GetDirection2(self,Vector1, Vector2):
        return np.cross(Vector1,Vector2)

    def SetBurgersComponents(self):
        self.bscrew1 = np.dot(self.b1ref,self.xi1)
        self.bscrew2 = np.dot(self.b2ref,self.xi2)
        self.bedge1 = np.dot(np.cross(self.xi1,self.b1ref),np.array([0.,0.,1.]))
        self.bedge2 = np.dot(np.cross(self.xi2,self.b2ref),np.array([0.,0.,1.]))
        self.b1refNormal = self.NormalizeVector(self.b1ref)
        self.b2refNormal = self.NormalizeVector(self.b2ref)
        return

    def GetDislocationStressMatrix(self,PrecipitateBool):

        #
        # Compute Dislocation Stress Matrix
        #
        StressMatrix = 0.
        StressMatrix += self.RotateStress(self.LubardaFarFieldStress(self._d1,self.bedge1,self.bscrew1,Precipitate = PrecipitateBool),self.xi1)
        StressMatrix += self.RotateStress(self.LubardaFarFieldStress(self._d2,self.bedge2,self.bscrew2,Precipitate = PrecipitateBool),self.xi2)

        #
        # return StressMatrix
        #
        return StressMatrix

    def GetDislocationStrainMatrix(self,sigma,mu,nu):

        #
        # See Allan Bower: Applied Mechanics of Solids.
        #
        YoungsModulus = 2.*mu*(1.+nu)

        #
        # Compliance Tensor
        #
        Compliance = 1./YoungsModulus*np.array([[1.,-nu,-nu,0,0,0],[-nu,1.,-nu,0,0,0],[-nu,-nu,1.,0,0,0],[0,0,0,2.*(1.+nu),0.,0.],[0.,0.,0.,0.,2.*(1.+nu),0.],[0.,0.,0.,0.,0.,2.*(1.+nu)]])

        #
        # Stress Vector
        #
        Stress = np.array([sigma[0][0], sigma[1][1],sigma[2][2], sigma[1][2], sigma[0][2], sigma[0][1]])

        #
        # Calculate Strains
        #
        Strain = np.dot(Compliance,Stress)

        #
        # return Strain Matrix
        #
        return np.array([[Strain[0], Strain[5], Strain[4]],[Strain[5],Strain[1], Strain[3]],[Strain[4], Strain[3], Strain[2]]])

    def RotateStress(self,sigma,xi):
        #
        # from e to m (Dislocation to Global)
        #
        return np.matmul(np.matmul(self.RotationMatrix[tuple(xi)].transpose(),sigma),self.RotationMatrix[tuple(xi)])

    def SetDislocationRotationMatrix(self):

        #
        # from m to e (Global to Dislocation)
        #
        # sigma_e = Q * sigma_m *Q^T
        # v_e = Q * v_m
        #

        M = np.eye(3)

        e1 = np.array([0., 0., 1.])

        E = lambda X: np.array([e1,
                                np.cross(X,e1),
                                X])

        self.RotationMatrix = {}
        self.RotationMatrix[tuple(self.xi1)] = np.matmul(E(self.xi1),M)
        self.RotationMatrix[tuple(self.xi2)] = np.matmul(E(self.xi2),M)

        return

    def LubardaStressFields(self,aNS,x,y,D,bedge,bscrew):
        """
        See Appendix:

        Lubarda, V. A. "Energy Analysis of Dislocation Arrays Near Bimaterial Interfaces," Int. J. Solids and Structures. Volume 34, pp 1053 - 1073. 1997.

        Lubarda, V. A., Kouris D. A. "Stress fields due to dislocation arrays at interfaces," Mechanics of Materials. Volume 23, pp 191 - 203. 1996.

        Dislocation Coordinate System (e):

                      y
                      ^
                      |.|-
                      |.|-
        [2] (Matrix)  |.|-    [1] (Precipitate)
                      |.|-
                      ------------> x
                     / .|-
                    /  .|-
                   /   .|-
                  /    .|-
                Z<

        (Z is out of the page and parallel to all dislocation lines)
        """

        #
        # Normalize Variables
        #
        X = x/D
        Y = y/D
        A = aNS/D

        #
        # Partition between weighted convolution functions (Cai et al. 2006)
        #
        # a1 = 0.9038 * A
        a1 = A
        a2 = 0.5451 * A
        # m = 0.6575
        m = 0.

        #
        # edge parameter
        #
        Gamma = self.muMatrix/self.muPrecipitate
        k1 = 3.-4.*self.nuPrecipitate
        k2 = 3.-4.*self.nuMatrix
        #
        # Equation A.2b
        #
        alpha = (Gamma*(k1+1.) - k2 - 1.)/(Gamma*(k1+1.)+k2 + 1.)
        #
        # Equation A.2c
        #
        beta = (Gamma * (k1-1.) - k2 + 1.)/(Gamma*(k1+1.) + k2 + 1.)
        #
        # Equation A.2a
        #
        mu_precipitate_star = self.muPrecipitate*(1.+alpha)/(1.-beta*beta)
        k0 = mu_precipitate_star/2./np.pi/(1.-self.nuPrecipitate)
        #
        # Equation A.1a
        #
        omega = 1.-beta if x >= 0. else 1.+beta

        #
        # screw parameter
        #
        k4 = (self.muPrecipitate - self.muMatrix)/(self.muPrecipitate + self.muMatrix)
        #
        # Equation A.1b
        #
        epsilon = self.muPrecipitate*(1.-k4) if x >= 0. else self.muMatrix*(1.+k4)

        #
        # screw dislocation array
        #
        sigma_xz =  -epsilon*bscrew/2./np.pi/D * ( (1.-m) * ( self.S2(Y,np.sqrt(X**2.+a1**2.)) +  a1**2.*self.S5(Y,np.sqrt(X**2.+a1**2.))) + m * (self.S2(Y,np.sqrt(X**2.+a2**2.)) + a2**2.*self.S5(Y,np.sqrt(X**2.+a2**2.))))
        sigma_yz = epsilon*bscrew*X/2./np.pi/D * ( (1.-m) * (self.S3(Y,np.sqrt(X**2.+a1**2.)) -  a1**2./(2.*np.sqrt(X**2.+a1**2.)) * self.S6(Y,np.sqrt(X**2.+a1**2.))) + m * (self.S3(Y,np.sqrt(X**2.+a2**2.)) - a2**2./(2.*np.sqrt(X**2.+a2**2.)) * self.S6(Y,np.sqrt(X**2.+a2**2.))))

        #
        # edge dislocation array
        #
        sigma_xx = k0*bedge*X/D *  ( (1.-m) * (self.S3(Y,np.sqrt(X**2.+a1**2.)) + omega * np.sqrt(X**2. + a1**2.) * self.S6(Y,np.sqrt(X**2.+a1**2.))) + m * (self.S3(Y,np.sqrt(X**2.+a2**2.)) + omega * np.sqrt(X**2. + a2**2.) * self.S6(Y,np.sqrt(X**2.+a2**2.))))
        sigma_yy = k0*bedge*X/D *  ( (1.-m) * (self.S3(Y,np.sqrt(X**2.+a1**2.)) - 2.*omega*self.S4(Y,np.sqrt(X**2.+a1**2.)) - omega*(X**2.+2.*a1**2.)/np.sqrt(X**2.+a1**2.) * self.S6(Y,np.sqrt(X**2.+a1**2.)) ) + m * (self.S3(Y,np.sqrt(X**2.+a2**2.)) - 2.*omega * self.S4(Y,np.sqrt(X**2.+a2**2.)) - omega * (X**2.+2.*a2**2.)/np.sqrt(X**2.+a2**2.) * self.S6(Y,np.sqrt(X**2.+a2**2.))))
        sigma_xy = k0*bedge/D * ( (1.-m) * (self.S2(Y,np.sqrt(X**2.+a1**2.)) - 2.*omega * X**2. * self.S5(Y,np.sqrt(X**2.+a1**2.))) + m * (self.S2(Y,np.sqrt(X**2.+a2**2.)) - 2.*omega * X**2. * self.S5(Y,np.sqrt(X**2.+a2**2.))))

        #
        # plane strain. Equation A.1c
        #
        sigma_zz = self.nuPrecipitate * (sigma_yy + sigma_xx) if x >= 0. else self.nuMatrix * (sigma_yy + sigma_xx)

        #
        # Equations 15a - 15f
        #
        return np.array([[sigma_xx, sigma_xy, sigma_xz],[sigma_xy, sigma_yy, sigma_yz], [sigma_xz, sigma_yz, sigma_zz]])

    def LubardaFarFieldStress(self,D, bedge, bscrew, Precipitate = True):

        #
        # precipitate is on top
        #

        #
        # parameters
        #
        k1 = 3.-4.*self.nuPrecipitate
        k2 = 3.-4.*self.nuMatrix
        k4 = (self.muPrecipitate - self.muMatrix)/(self.muPrecipitate + self.muMatrix)
        Gamma = self.muMatrix/self.muPrecipitate
        #
        # Equation A.2b
        #
        alpha = (Gamma*(k1+1.) - k2 - 1.)/(Gamma*(k1+1.)+k2 + 1.)
        #
        # Equation A.2c
        #
        beta = (Gamma * (k1-1.) - k2 + 1.)/(Gamma*(k1+1.) + k2 + 1.)
        #
        # Equation A.2a
        #
        mu_precipitate_star = self.muPrecipitate*(1.+alpha)/(1.-beta*beta)
        k0 = mu_precipitate_star/2./np.pi/(1.-self.nuPrecipitate)
        k = k0*bedge/D
        #
        # Equation A.1a
        #
        omega = 1.-beta if Precipitate  else 1.+beta
        #
        # Equation A.1b
        #
        epsilon = self.muPrecipitate*(1.-k4) if Precipitate else self.muMatrix*(1.+k4)

        #
        # stress fields
        #
        sigma_xz = 0.
        sigma_xy = 0.
        sigma_xx = beta*k*np.pi
        sigma_yy = (beta**2.-(1.-omega)*2.)*k*np.pi/beta
        sigma_yz = epsilon*(1.-omega)*bscrew/2./D/beta
        #
        # Equation A.1c
        #
        sigma_zz = self.nuPrecipitate * (sigma_yy + sigma_xx) if Precipitate else self.nuMatrix * (sigma_yy + sigma_xx)

        #
        # Equations 17a - 17f
        #
        return np.array([[sigma_xx, sigma_xy, sigma_xz],[sigma_xy, sigma_yy, sigma_yz], [sigma_xz, sigma_yz, sigma_zz]])

    #
    # Equation B.2a
    #
    def S2(self,p,q):
        return np.pi * np.sin(2.*np.pi*p) / (np.cosh(2.*np.pi*q)-np.cos(2.*np.pi*p))
    #
    # Equation B.2b
    #
    def S3(self,p,q):
        return (np.pi/q) * np.sinh(2.*np.pi*p) / (np.cosh(2.*np.pi*q)-np.cos(2.*np.pi*p))
    #
    # Equation B.2c
    #
    def S4(self,p,q):
        return (2.*np.pi**2.) * (np.cosh(2.*np.pi*q) * np.cos(2.*np.pi) - 1.) / (np.cosh(2.*np.pi*q)-np.cos(2.*np.pi*p))**2.
    #
    # Equation B.2d
    #
    def S5(self,p,q):
       return (np.pi**2./q) * np.sinh(2.*np.pi*q) * np.sin(2.*np.pi*p) / (np.cosh(2.*np.pi*q)-np.cos(2.*np.pi*p))**2.
    #
    # Equation B.2e
    #
    def S6(self,p,q):
        return (-np.pi/q**2.)*((np.cosh(2.*np.pi*q) - np.cos(2.*np.pi*p))*np.sinh(2.*np.pi*q) + 2.*np.pi*q * (np.cos(2.*np.pi*p)*np.cosh(2.*np.pi*q) - 1.)) / (np.cosh(2.*np.pi*q)-np.cos(2.*np.pi*p))**2.
