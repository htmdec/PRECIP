################################################################################
#title          : Steel_Info.py
#description    : Necessary materials information for PIG_E code to calculate elastic interfacial energy 
#authors        : H. Murdoch, B. Szajewski, E. Hernandez-Rivera, M. Guziewski, D. Field, D. Magagnosc (ARL)
#revised        : 28JUL2025
#version        : 1.0
#usage          : PIG_E.py
#notes          : see ARL report AD1315229: Assessment of Materials Data for High-Throughput Interfacial Energy Calculations for Prevalent Carbides
################################################################################

#%% Necessary Modules
import numpy as np   

#%% Summary information about the material system

#
# Information about the material system 
#  

def GetMatrixPhase():
    """
    Returns
    -------
    the name of the matrix phase

    """
    return ['BCC_A2'] # Matrix phase alpha-ferrite / alpha'-martensite
 
def GetPrecipitateList():
    """
    Returns
    -------
    list of precipitate phases currently considered in the system in Thermo-Calc notation
    """
    return ['FCC_A1#2', 'M23C6_D84', 'M6C_E93', 'FE4N_LP1', 'KSI_CARBIDE', 'M5C2', 'HCP_A3#2', 'MC_SHP', 'MC_ETA', 'M7C3_D101', 'CEMENTITE_D011', 'M3C2_D510']

def GetPrecipitateORs():
    """
    Returns
    -------
    dictionary of possible Orientation Relationships for each named carbide phase
    """
    return {'FCC_A1#2': ['Bain', 'KurdjumovSachs', 'NishiyamaWasserman', 'Pitsch'], 
                             'M23C6_D84': ['Bain', 'KurdjumovSachs', 'NishiyamaWasserman', 'Pitsch'], 
                             'M6C_E93': ['Bain', 'KurdjumovSachs', 'NishiyamaWasserman', 'Pitsch'], 
                             'FE4N_LP1': ['Bain', 'KurdjumovSachs', 'NishiyamaWasserman', 'Pitsch'], 
                             'KSI_CARBIDE': ['KSI_CARBIDE'], 
                             'M5C2': ['Fe5C2'], 
                             'HCP_A3#2': ['Burgers', 'PitschSchrader'], 
                             'MC_SHP': ['Burgers', 'PitschSchrader'], 
                             'MC_ETA': ['MC_ETA_PitschSchrader'], 
                             'M7C3_D101': ['Bagaryatski', 'Isaichev'], 
                             'CEMENTITE_D011': ['Bagaryatski', 'Isaichev'], 
                             'M3C2_D510': ['Bagaryatski', 'Isaichev']}

def GetPhaseElements():
    """
    Returns
    -------
    list of elements considered in the current system
    """
    return ['C', 'Mn', 'Si', 'Cr', 'Ni', 'Mo', 'V', 'Co', 'Al', 'W', 'Cu', 'Ti', 'B', 'N', 'S', 'Fe']

#%% Lattice Paramter


def LatticeParameterInfo(Phase):
    """
    Dictionaries describing the crystallography of the phase. Phase names are the dictionary keys. 

    Parameters
    ----------
    Phase : Name of the phase (in Thermo-Calc notation)

    Returns
    -------
    Unit : number of atoms in the unit cell for the phase.
    Ratio : Ratio of lattice parameter b to a and c to a forthe phase.
    Cell Angles : Angles alpha/beta/gamma of the phase lattice.

    """
    #
    # Dictionaries describing the unit cells of each phase
    #
    
    # Number of atoms in a full unit cell [ref: Materials Project]
    Unit = {# cubic
            'FCC_A1#2' : 8, 
            'BCC_A2': 2, 
            'M23C6_D84':116, 
            'M6C_E93':112, 
            'FE4N_LP1':4,
            
            # hexagonal
            'HCP_A3#2':3, 
            'MC_SHP':2, 
            'MC_ETA':33,
            
            # monoclinic
            'KSI_CARBIDE':44, 
            'M5C2':28, 
            
            # orthorhombic
            'M7C3_D101':40, 
            'CEMENTITE_D011':16, 
            'M3C2_D510':20}
    
    # Ratios of lattice parameters
    # Determined via an average of literature values for the ratios [see documentation]
    Ratio={#cubic
           'FCC_A1#2' : {'b:a' : 1, 'c:a' : 1}, 
           'BCC_A2': {'b:a' : 1, 'c:a' : 1}, 
           'M23C6_D84': {'b:a' : 1, 'c:a' : 1}, 
           'M6C_E93': {'b:a' : 1, 'c:a' : 1}, 
           'FE4N_LP1': {'b:a' : 1, 'c:a' : 1},
           
           # hexagonal
           'HCP_A3#2': {'b:a' : 1, 
                        'c:a' : 1.58}, 
           'MC_SHP': {'b:a' : 1, 
                      'c:a' : 0.971}, 
           'MC_ETA': {'b:a' : 1, 
                      'c:a' : 2.81},
           
           # monoclinic
           'KSI_CARBIDE' : {'b:a' : 0.71, 
                            'c:a' : 0.6}, 
           'M5C2' : {'b:a' : 0.3966, 
                     'c:a' : 0.44}, 
           
           #orthorhombic 
           'CEMENTITE_D011' : {'b:a' : 1.33,
                               'c:a' : 0.880},
           'M7C3_D101' : {'b:a' : 1.53,
                          'c:a' : 2.65},
           'M3C2_D510' : {'b:a' : 0.51,
                          'c:a' : 2.08}
           }

    # Angles of cell [alpha , beta , gamma] [ref: Materials Project]
    CellAngles = {'FCC_A1#2' : [90,90,90], 
          'BCC_A2': [90,90,90], 
          'M23C6_D84':[90,90,90], 
          'M6C_E93':[90,90,90], 
          'FE4N_LP1':[90,90,90],
          
          'KSI_CARBIDE':[90,120.7,90],
          'M5C2':[90,97.5,90],
          
          'HCP_A3#2':[90,90,120], 
          'MC_SHP':[90,90,120], 
          'MC_ETA':[90,90,120], 
           
          'M7C3_D101':[90,90,90], 
          'CEMENTITE_D011':[90,90,90], 
          'M3C2_D510':[90,90,90]}
        
    return Unit[Phase], Ratio[Phase], CellAngles[Phase]

#%% Elastic Constants

def ElasticConstantsMATRIX(CrystalCLASS, Martensite=True, Curie=False):
    """
    Calculates elastc constants of ferrite or martensite matrix as a function of composition and temperature.
    Equations from Ghosh & Olson 2002 [https://doi.org/10.1016/S1359-6454(02)00096-4]

    Parameters
    ----------
    CrystalCLASS : The instance of the class to which the properties will be added.
    Martensite : Whether to use the equation for martensite. The default is True. If False, ferrite equation is used.
    Curie : Whether to use a compositionally dependent Curie temperature or that of pure Fe. The default is False. 
            If using a Curie temperature, the units are in KELVIN

    Returns
    -------
    ShearModulus : shear modulus of the matrix in GPa.
    Poisson : Poisson's ratio of the matrix.

    """

    if Curie:
        TCurie = Curie
    else:
        TCurie = 1043
    
    T_TC = CrystalCLASS.Temperature / TCurie
    
    # compositional effect dictionary ATOMIC FRACTION
    dx = {'Al': -10.085, 
          'C': -14.194, 
          'Co': 2.389, 
          'Cr': 3.406,
          'Cu': -0.145,
          'Mn' : -2.263,
          'Mo' : -1.671,
          'N' : -15.439,
          'Nb' : -8.707,
          'Ni' : -9.065,
          'Si' : -10.914,
          'Ti' : -7.459,
          'V' : 1.014,
          'W' : 7.267
          }
          
    dmu_dx = sum([CrystalCLASS.PhaseComposition[x] * dx[x] for x in CrystalCLASS.PhaseComposition.keys() if x in dx.keys()])
               
    # units of Pa
    if T_TC < 1:
        if Martensite:
            # Shear Modulus of alpha'-MARTENSITE Eq (15) from Ghosh & Olson 2002
            Shear = (8.068 + dmu_dx) * (1 - 0.48797*(T_TC)**2 + 0.12651*(T_TC)**3.) * 10.**10.
        else:
            # Shear Modulus of alpha-FERRITE Eq (7) from Ghosh & Olson 2002
            Shear = (8.407 + dmu_dx) * (1 - 0.48797*(T_TC)**2 + 0.12651*(T_TC)**3) * 10**10
        
        # Bulk Modulus of alpha-FERRITE (is no equation for Martensite) Eq (5a)
        Bulk = 17.187 * (1. - 0.28029*(T_TC)**2. + 0.07221 * (T_TC)**3.) * 10.**10.
    else:
        # Equations for Ferrite, Eq (5b) & Eq(6b)
        Shear = 10.296 * (1 - 0.48153*(T_TC))*10**10
        Bulk = 17.764 * (1. - 0.23234*T_TC ) * 10.**10.
    
    ShearModulus = Shear / 1e9
    BulkModulus = Bulk / 1e9
    Poisson = (3*BulkModulus - 2*ShearModulus) / (2*(3*BulkModulus + ShearModulus))
    
    return ShearModulus, Poisson

def ElasticConstantsPRECIP(CrystalCLASS):
    """
    Returns elastc constant of a carbide phase (as a function of composition).
    Contains dictionaries by phase. Subdictionary keys include specific elements if constants vary by composition,
    and 'Alloy' for an average / not varied by composition. 
    For details see [ARL report AD1315229: Assessment of Materials Data for High-Throughput Interfacial Energy Calculations for Prevalent Carbides]

    Parameters
    ----------
    CrystalCLASS : The instance of the class to which the properties will be added.

    Returns
    -------
    ShearModulus : shear modulus of the precipitate in GPa.
    Poisson : Poisson's ratio of the precipitate.

    """
    
    Shear = {}
    v = {}

    
    # Phases for which we are using an average shear modulus and poisson ratio, regardless of composition
    Shear['M23C6_D84'] = {'Alloy' : 137.5}
    v['M23C6_D84'] = {'Alloy' : 0.329}
    
    Shear['FE4N_LP1'] = {'Alloy' : 79.64}
    v['FE4N_LP1'] = {'Alloy' : 0.36}
        
    Shear['HCP_A3#2'] = {'Alloy' : 149.71}
    v['HCP_A3#2'] = {'Alloy' : 0.27}
        
    Shear['M3C2_D510'] = {'Alloy' : 182}
    v['M3C2_D510'] = {'Alloy' : 0.26}
    
    Shear['KSI_CARBIDE'] = {'Alloy' : 71.51}	
    v['KSI_CARBIDE'] = {'Alloy' : 0.367}
           
    
    # Phases for which the shear modulus and poisson are a function of lattice parameter in ANGSTROMS
    if (CrystalCLASS.a > 10.8) & (CrystalCLASS.a < 11.5):
        Shear['M6C_E93'] = {'Alloy' : -126.9 * CrystalCLASS.a + 1552.9}
    else: 
        # this is an average of all just to catch any issue with the fit line
        Shear['M6C_E93'] = {'Alloy' : 140.79}
    v['M6C_E93'] = {'Alloy' : 0.317}
    
    
    # Phases for which the shear modulus and poisson are a function of composition              
    # Shear modulus and Poissons ratio are determined by the primary element
    Shear['FCC_A1#2'] = {'Ti': 181.51, 
                         'V': 221.21,  
                         'Cr': 167.36, 
                         'Mo': 166.96,
                         'Alloy' : 184.26
                         }
    v['FCC_A1#2'] = {'Ti': 0.217, 
                     'V': 0.228,  
                     'Cr': 0.3, 
                     'Mo': 0.3,
                     'Alloy' : 0.26125
                     }

    Shear['MC_SHP'] = {'Mo': 232.99, 
                       'W':287.01}
    v['MC_SHP'] = {'Mo': 0.2225, 
                   'W':0.2033}
    
    Shear['MC_ETA'] = {'Mo' : 160.81,
                       'V' : 191.44,
                       'Alloy' : 191.44}
    v['MC_ETA'] = {'Mo' : 0.29,
                   'V' : 0.2,
                   'Alloy' : 0.2}
    
    Shear['CEMENTITE_D011'] = {'Cr': 140.09, 
                               'Fe':80.41, 
                               'Mn': 112.01}
    v['CEMENTITE_D011'] = {'Cr': 0.29, 
                           'Fe':0.30, 
                           'Mn':0.335}
    
    Shear['M7C3_D101'] = {'Mn' : 103.6, 
                          'Fe' : 107.6,
                          'Cr' : 119.6,
                          'Alloy': 138.4}
    v['M7C3_D101'] = {'Mn' : 0.36, 
                      'Fe' : 0.33, 
                      'Cr' : 0.33,
                      'Alloy': 0.32}

    Shear['M5C2'] = {'Mn': 127.7, 
                     'Fe':107.9}
    v['M5C2'] = {'Mn': 0.317, 
                 'Fe':0.337}

        
    # removing C from the composition in order to find max non-C element (important for close to stoichiometic carbides)            
    M_comp = CrystalCLASS.PhaseComposition.copy()
    del M_comp['C']
        
    MaxElem = max(M_comp, key=M_comp.get)
    
    # is MaxElem in the list of options for composition specific 
    if MaxElem in Shear[CrystalCLASS.CrystalStructure].keys():
        ShearModulus = Shear[CrystalCLASS.CrystalStructure][MaxElem]
        Poisson = v[CrystalCLASS.CrystalStructure][MaxElem]
                     
    else:
        ShearModulus = Shear[CrystalCLASS.CrystalStructure]['Alloy']
        Poisson = v[CrystalCLASS.CrystalStructure]['Alloy']
    
    return ShearModulus, Poisson

#%% Orientation Relationships

unity = lambda size : np.ones((size,1))
normalize = lambda x: x/np.linalg.norm(x)

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


def GetSubUnits():
    """
    Dictionary of phases for unit cells which contain more than one precipitate structural unit relative to a matrix structural unit on the interface plane.
    Keys are phase name, subdictionary keys are OR name and then:
        matrix : subunits of matrix unit cell
        precip : subunits of precipitate unit cell
    
    Returns
    -------
    Dictionary of sub units by precipitate and OR

    """
    return {'M23C6_D84' : {'Bain' : {'matrix' : np.array([[3.,3.,3.]]).T,
                                    'precip' : unity(3),
                                    },
                              'KurdjumovSachs' : {'matrix' : np.array([[3.,3.,3.]]).T,
                                                  'precip' : unity(3),
                                                  },
                              'NishiyamaWasserman' :{'matrix' : np.array([[3.,3.,3.]]).T,
                                                     'precip' : unity(3),
                                                     },
                              'Pitsch' :{'matrix' : np.array([[3.,3.,3.]]).T,
                                         'precip' : unity(3),
                                         },
                              },
               'M6C_E93' : {'Bain' : {'matrix' : np.array([[3.,3.,3.]]).T,
                                      'precip' : unity(3),
                                      },
                            'KurdjumovSachs' : {'matrix' : np.array([[3.,3.,3.]]).T,
                                                'precip' : unity(3),
                                                },
                            'NishiyamaWasserman' :{'matrix' : np.array([[3.,3.,3.]]).T,
                                                   'precip' : unity(3),
                                                   },
                            'Pitsch' :{'matrix' : np.array([[3.,3.,3.]]).T,
                                       'precip' : unity(3),
                                       },
                            },
               'CEMENTITE_D011' : {'Bagaryatski' : {'matrix' : unity(3),
                                                    'precip' : unity(3),
                                                    },
                                   'Isaichev' : {'matrix' : np.array([[1, 2, 1]]).T,
                                                 'precip' : unity(3),
                                                 }
                                   },
               'M7C3_D101' : {'Bagaryatski' : {'matrix' : np.array([[3.,1.,1]]).T,
                                               'precip' : np.array([[2.,1.,1.]]).T,
                                               },
                              'Isaichev' : {'matrix' : np.array([[3./2., 3., 1.]]).T,
                                            'precip' : unity(3),
                                            }
                              },
               'M3C2_D510' : {'Bagaryatski' : {'matrix' : np.array([[1., 2., 1]]).T,
                                               'precip' : np.array([[1., 3., 1.]]).T,
                                               },
                              'Isaichev' : {'matrix' : np.array([[1., 3., 1.]]).T,
                                            'precip' : unity(3),
                                            }
                              },
               'KSI_CARBIDE' : {'KSI_CARBIDE' : {'matrix' : np.array([[3.,3.,1.]]).T,
                                                'precip' : unity(3),
                                                },
                                },
               }   



class NewOrientationRelationships(object):
    """
    Adds new orientation relationships to those already included in the generalized CrystalInterface class
    
    functions are for the new ORs
    
    Summary dictionaries to update those in CrystalInterface class:
    ---     
    bCount : number of burgers vectors in the interface plane for the OR
        
    """
    #
    # Summary of new OR Burgers vector information
    #
            
    bCount = {'KSI_CARBIDE': 2,
              'Fe5C2' : 2,
              'MC_ETA_PitschSchrader' : 3
             }
    
    #
    # New OR functions
    #
    
    def KSI_CARBIDE(self):
        """
        Orientation relationship between Ksi-carbide (M17C5) and ferrite (BCC) matrix.

        """
    
        a_BCC = self.a_Matrix
        

        #
        # Parallel Orientations
        #        
        BCC = np.array([[1., 0., 0.], 
                        [0., 1., 1.], 
                        [0., -1., 1.]]) 
            
        Mono = np.array([[0., 1., 0],
                         [1., 0., 0.],
                         [0., 0., -1.]])
        
        OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                            np.matmul(self.Matrix.CBtoRB,BCC[1]),
                            self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,
                                                                                        self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                            ]) * self.SubUnitDict['matrix']
                            
                
        OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,Mono[0]),
                            np.matmul(self.Precipitate.CBtoRB,Mono[1]),
                            self.PlaneSpacing(Mono[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,
                                                                                              self.PlaneNormalVector(Mono[2],self.Precipitate.g)))
                            ]) * self.SubUnitDict['precip']

        #
        # Burgers vectors in BCC Reference Frame
        #
        b0 = a_BCC/2*np.array([[1.,1.,1.],
                               [-1.,1.,1.]]) 
        
        
        return OR_alpha, OR_beta, b0
    

    def Fe5C2(self):
        """
        Orientation relationship between Hagg carbide (M5C2) and ferrite (BCC) matrix.

        """
    
        a_BCC = self.a_Matrix
        
        
        #
        # Hagg Carbide (Fe5C2) OR
        #
        
        #
        # Parallel Orientations
        #
        BCC = np.array([[-1., 1., 0.],
                        [1., 1., 1.],
                        [1., 1., -2.]]) 
        
        
        Mono = np.array([[0., 1., 0],
                         [0., 0., 1],
                         [1., 0., 0.]])
        
        OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                            np.matmul(self.Matrix.CBtoRB,BCC[1]),
                            self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,
                                                                                        self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                            ]) * self.SubUnitDict['matrix']
                
        OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,Mono[0]),
                            np.matmul(self.Precipitate.CBtoRB,Mono[1]),
                            self.PlaneSpacing(Mono[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,
                                                                                              self.PlaneNormalVector(Mono[2],self.Precipitate.g)))
                            ]) * self.SubUnitDict['precip']  
        

        #
        # Burgers vectors in BCC Reference Frame
        #
        b0 = a_BCC * np.array([[-1., 1., 0.],[1., 1., 1.]]) / np.array([[1],[2] ])
        
        
        return OR_alpha, OR_beta, b0

    def MC_ETA_PitschSchrader(self):


        #
        # Pitsch - Shrader OR
        #
        a_BCC = self.a_Matrix
        a_Hex = self.a1
        c_Hex = self.a3 
        
        #
        # Parallel Orientations
        #
       
        BCC = np.array([[1., -1., 0.],
                        [1., -1., 1.],
                        [1., 1., 0.]])


        HCP = np.array([ConvertHCP4toHCP3(np.array([1., -1., 0., 0.])),
                        ConvertHCP4toHCP3(np.array([1., -2., 1., 0.])),
                        ConvertHCP4toHCP3(np.array([0., 0., 0., 1.]))])
        
        # MC eta phase has a lattice rotated 30 degrees relative to a simple HCP unit cell.
        # Rotating the Pitsch-Schrader OR by 30 degrees produces the appropriate OR
        theta = np.deg2rad(30)
        rot = np.array([[np.cos(theta), -1*np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])
        
        HCP = np.matmul(HCP,rot)
        
        #
        # Scale parallel orientations and convert to common basis
        #
        OR_alpha = np.array([np.matmul(self.Matrix.CBtoRB,BCC[0]),
                    np.matmul(self.Matrix.CBtoRB,BCC[1]),
                    self.PlaneSpacing(BCC[2],self.Matrix.g)*normalize(np.matmul(self.Matrix.CBtoRB,self.PlaneNormalVector(BCC[2],self.Matrix.g)))
                    ]) * self.SubUnitDict['matrix'] 

        OR_beta = np.array([np.matmul(self.Precipitate.CBtoRB,HCP[0]),
                            np.matmul(self.Precipitate.CBtoRB,HCP[1]),
                            self.PlaneSpacing(HCP[2],self.Precipitate.g)*normalize(np.matmul(self.Precipitate.CBtoRB,self.PlaneNormalVector(HCP[2],self.Precipitate.g)))
                            ]) * self.SubUnitDict['precip']  
                            
        OR_alpha[1] /= 2.

        #   
        # Burgers vectors in BCC Reference Frame
        #  
        b0 = a_BCC * np.array([[1.,-1.,-1.],[1.,-1.,1.],[0,0.,1.]]) / np.array([[2.,2.,1.]]).T
    
        return OR_alpha, OR_beta, b0
        
