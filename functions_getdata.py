# =============================================================================
#
# This scripts contains functions used by getdata_*.py for both corr and SO
#
# Copyright (C) 2021-2024 Julius Kleine Büning
#
# =============================================================================


import os
import numpy as np
from ase import neighborlist
from ase.io import read
from ase.data import covalent_radii


########## GLOBAL DECLARATIONS ##########

# use slightly modified covalent radii from ase for neighbor recognition
custom_radii = covalent_radii.copy()
custom_radii[3] -= 0.15   # reduce radius of Li
custom_radii[6] -= 0.05   # reduce radius of C

# Covalent radii (taken from Pyykko and Atsumi, Chem. Eur. J. 15, 2009, 188-197).
# Values for metals decreased by 10 %.
# This was copied from gituhb project dftd3/tad-dftd3/src/tad_dftd3/data.py
covalent_rad_2009 = np.array([ 
    0.00,                                                # None
    0.32,0.46,                                           # H,He
    1.20,0.94,0.77,0.75,0.71,0.63,0.64,0.67,             # Li-Ne
    1.40,1.25,1.13,1.04,1.10,1.02,0.99,0.96,             # Na-Ar
    1.76,1.54,                                           # K,Ca
    1.33,1.22,1.21,1.10,1.07,1.04,1.00,0.99,1.01,1.09,   # Sc-Zn
    1.12,1.09,1.15,1.10,1.14,1.17,                       # Ga-Kr
    1.89,1.67,                                           # Rb,Sr
    1.47,1.39,1.32,1.24,1.15,1.13,1.13,1.08,1.15,1.23,   # Y-Cd
    1.28,1.26,1.26,1.23,1.32,1.31,                       # In-Xe
    2.09,1.76,1.62,                                      # Cs-La
    1.47,1.58,1.57,1.56,1.55,1.51,1.52,                  # Ce-Gd
    1.51,1.50,1.49,1.49,1.48,1.53,1.46,                  # Tb-Lu
    1.37,1.31,1.23,1.18,1.16,1.11,1.12,1.13,1.32,        # Hf-Hg
    1.30,1.30,1.36,1.31,1.38,1.42,                       # Tl-Rn
    2.01,1.81,1.67,                                      # Fr-Ac
    1.58,1.52,1.53,1.54,1.55,1.49,1.49,                  # Th-Cm
    1.51,1.51,1.48,1.50,1.56,1.58,1.45,                  # Bk-Lr
    1.41,1.34,1.29,1.27,1.21,1.16,1.15,1.09,1.22,        # Rf-Cn
    1.36,1.43,1.46,1.58,1.48,1.57                        # Nh-Og
])

# D3 covalent radii used to construct the coordianation number
covalent_rad_d3 = 4.0 / 3.0 * covalent_rad_2009

########## END GLOBAL DECLARATIONS ##########


def all_equal(lst):
    """Checks if all elements of the input list are equal."""

    first = lst[0]
    equal = True
    for el in lst:
        if el != first:
            equal = False
            break

    return equal


def read_number(entry):
    """For an arbitrary string entry containing one (int or float) number surrounded by chars, extract the number."""

    start, end = None, None

    for i, c in enumerate(entry):
        if c.isdigit():
            start = i
            break
    if start is None:
        print("ERROR in read_number(): entry '{}' does not contain any digit!".format(entry))
        exit()
    
    for i, c in enumerate(reversed(entry)):
        if c.isdigit():
            end = len(entry) - i
            break

    return entry[start:end]


def get_end(data, start):
    """For a given read input file, get the line range until next blank line.
    
    data: complete list of lines that has been read from a file via .readlines()
    start: line number of the first line of the text block
    """

    n_lines = 0
    while data[start+n_lines] != '\n':
        n_lines += 1

    return start + n_lines


def get_number_nmr(outlist):
    """Get number of calculated NMR nuclei (not necessarily only 1H and 13C).

    outlist: list of lines read from a calculation output file
    """
    
    n_nmr = None
    for line in outlist[outlist.index('                            *     ORCA property calculations      *\n'):]:
        if 'Number of nuclei for epr/nmr' in line: n_nmr = int(line.split()[-1])
    if n_nmr is None:
        print("ERROR: Number of calculated NMR nuclei was not found in ORCA output!")
        exit()

    return n_nmr


def read_orca(outpath):
    """Read the DFT output from ORCA calculations."""

    with open(outpath, 'r') as inp:
        data = inp.readlines()

    # get number of calculated NMR nuclei
    n_nmr = get_number_nmr(data)

    # get the range in which the shieldings are listed
    start = data.index('CHEMICAL SHIELDING SUMMARY (ppm)\n') + 6
    end = start + n_nmr

    shieldings = []
    for line in data[start:end]:
        tmp = line.split()
        shieldings.append({
            'nuc': int(tmp[0]) + 1,   # ORCA starts at 0 counting the nuc numbers
            'elem': tmp[1].upper(),
            'val': float(tmp[2])
        })

    # make sure the shieldings are ordered according to the atom numbering (store as tuple)
    return tuple(sorted(shieldings, key=lambda s: s['nuc']))


def getref_orca(path):
    """Get the reference shieldings from the DFT output (ORCA).
    
    ATTENTION: all Hs and all Cs are averaged, works e.g. for TMS and CH4.
    """

    with open(os.path.join(path, "orca.out"), 'r') as inp:
        data = inp.readlines()

    # get number of calculated NMR nuclei
    n_nmr = get_number_nmr(data)

    # get the range in which the shieldings are listed
    start = data.index('CHEMICAL SHIELDING SUMMARY (ppm)\n') + 6
    end = start + n_nmr
    
    val_h = 0.0
    val_c = 0.0
    cnt_h = 0
    cnt_c = 0

    for line in data[start:end]:
        tmp = line.split()
        if tmp[1].upper() == 'H':
            val_h += float(tmp[2])
            cnt_h += 1
        if tmp[1].upper() == 'C':
            val_c += float(tmp[2])
            cnt_c += 1

    val_h = val_h / cnt_h
    val_c = val_c / cnt_c

    return {'H': val_h, 'C': val_c}


def read_mol(structpath):
    """Read the molecule and return mol (ase.Atoms object) and dict neighbors."""

    # read the .xyz coordinates from the molecular structures
    mol = read(os.path.join(structpath), format='xyz')

    # use covalent radii as thresholds for neighbor determination (what about vdW radii?)
    cutoffs = [custom_radii[atom.number] for atom in mol]

    # build neighbor list and write list of neighboring atoms to the dict neighbors
    nl = neighborlist.build_neighbor_list(mol, cutoffs, self_interaction=False, bothways=True)
    neighbors = {}

    for i in range(len(mol)):
        indices = nl.get_neighbors(i)[0]         # nl.get_neighbors(i) returns [0]: indices and [1]: offsets
        neighbors[i+1] = indices+1               # add 1 to key and to value to start counting of atoms at 1

        # exit if an H atom has not exactly 1 neighbor
        if mol.get_atomic_numbers()[i] == 1 and len(neighbors[i+1]) != 1:
            print("ERROR: H atom {} has not exactly one neighbor! File in: {}".format(i+1, structpath))
            exit()

    return mol, neighbors


def get_cn(neighbors):
    """Get the coordination number (CN) as number of neighbors (input is neighbors from read_mol)

    read_mol() ensures that every H atom has exactly one neighbor.
    """

    cn = {}
    for key, value in neighbors.items():
        cn[key] = len(value)

    return cn


def get_cn_d3(mol):
    """Get the coordination number (CN) as it is done in the D3 dispersion correction model.

    CN(A) = sum_B!=A^Nat ( 1 / (1 + exp(-k1 (k2*R_AB^cov/R_AB - 1))) )
    with A, B: atom indices, Nat: number of atoms in molecule,
    R_AB: distance of A and B according to input structure
    k2*R_AB^cov = k2*(R_A^cov + R_B^cov): element-specific covalent atom radii from Pyykkö and Atsumi scaled with k2 = 4/3
    k1 = 16: scaling factor; both k1 and k2 are set in the D3 model (J. Chem. Phys. 132, 154104); covalent_rad_d3 already include k2 (see above)
    """

    an = mol.get_atomic_numbers()             # get the atomic numbers as list
    distances = mol.get_all_distances()       # get all the atom distances as 2D list
    cn = {i+1: 0.0 for i in range(len(an))}   # fill a dict with 0.0 for all atom indices

    # loop over all atom pairs
    for a in range(len(an)):
        for b in range(a+1, len(an)):
            r_ab = distances[a][b]
            rcov_ab = covalent_rad_d3[an[a]] + covalent_rad_d3[an[b]]
            term = 1.0 / (1.0 + np.exp(-16.0 * (rcov_ab/r_ab - 1.0)))
            # add the term of the sum to the CN entry of both a and b (in dict: atom indices, no 0)
            cn[a+1] += term
            cn[b+1] += term

    return cn


def get_atomic_charges(mol, outputpath, mode):
    """Get atomic charges from ORCA output.

    Everything is stored in dicts {index: charge}.
    mode: can be 'mulliken' or 'loewdin'
    """

    nat = len(mol)
    charges = {}

    if mode == 'mulliken':
        pattern = 'MULLIKEN ATOMIC CHARGES\n'
        offset = 2
    elif mode == 'loewdin':
        pattern = 'LOEWDIN ATOMIC CHARGES\n'
        offset = 2
    else:
        print("ERROR: unknown mode in get_atomic_charges!")
        exit()

    with open(outputpath, 'r') as inp:
        data = inp.readlines()
    
    # get the range in which the Mulliken charges are listed
    start = data.index(pattern) + offset
    end = start + nat

    # first entry in line is index starting at 0, last is the value
    for line in data[start:end]:
        tmp = line.split()
        charges[int(tmp[0])+1] = float(tmp[-1])

    return charges


def get_Mayer_pop(mol, outputpath):
    """Get the Mayer population analysis.
    
    VA: total valence
    BVA: bonded valence
    FA: free valence
    """

    nat = len(mol)
    va = {}
    bva = {}
    fa = {}

    with open(outputpath, 'r') as inp:
        data = inp.readlines()
    
    # get the range in which the Mayer valence values are listed
    start = data.index('                      * MAYER POPULATION ANALYSIS *\n') + 11
    end = start + nat

    # first column is atom index, 6-8 in line is index of the valence quantities
    for line in data[start:end]:
        tmp = line.split()
        va[int(tmp[0])+1] = float(tmp[5])
        bva[int(tmp[0])+1] = float(tmp[6])
        fa[int(tmp[0])+1] = float(tmp[7])

    return va, bva, fa


def get_bond_orders(neighbors, outputpath, mode):
    """Get bond orders of Loewdin and Mayer type.
    
    Reads bond orders and calculates the sum and average for each atom.
    Returns: list of dicts
    mode: can be 'loewdin' or 'mayer'
    """

    bond_orders = []
    bond_orders_sum = {}
    bond_orders_av = {}
    if mode == 'loewdin':
        pattern = 'LOEWDIN BOND ORDERS (THRESH 0.050000)\n'
        offset = 2
        thresh_bo = 0.05
    elif mode == 'mayer':
        pattern = '  Mayer bond orders larger than 0.100000\n'
        offset = 1
        thresh_bo = 0.1
    else:
        print("ERROR: unknown mode in get_bond_orders!")
        exit()

    with open(outputpath, 'r') as inp:
        data = inp.readlines()
    
    # get the range in which the bond orders of the given pattern are listed
    start = data.index(pattern) + offset
    end = get_end(data, start)

    # read all bond orders and store in list of dicts (up to 3 entries per line)
    for line in data[start:end]:
        # remove all unnecessary characters from the list because they can unite with the element symbol if it has 2 letters
        # e.g. ['0-C', ','] but ['11-Li,'] (same for ')', same for 3-digit atoms numbers); without those, all entries should contain 3 elements
        tmp = line.replace(',', ' ').replace('B(', ' ').replace(')', ' ').replace(':', ' ').split()
        for i in range(int(len(tmp)/3)):
            bond_orders.append({
                'atom_A': int(read_number(tmp[i*3]))+1,
                'atom_B': int(read_number(tmp[i*3+1]))+1,
                'bond_order': float(tmp[i*3+2])
            })

    # use atoms (key) and neighbors (value) to look for all bonds for each atom and collect the respective bond orders
    for key, value in neighbors.items():
        these_bos = []
        for bo in bond_orders:
            if (bo['atom_A'] == key and bo['atom_B'] in value) or (bo['atom_B'] == key and bo['atom_A'] in value):
                these_bos.append(bo['bond_order'])
        
        # in case bond order of known neighbors are below the threshold, they are missing; add respective entries with value 1/2 * threshold
        bo_missing = len(value) - len(these_bos)
        if bo_missing != 0: these_bos.extend([thresh_bo/2 for i in range(bo_missing)])
        
        # calculate sum and average of the collected bond orders
        bond_orders_sum[key] = sum(these_bos)
        bond_orders_av[key] = sum(these_bos)/len(these_bos)   # this breaks if an atoms has no neighbors
        
    return bond_orders_sum, bond_orders_av

