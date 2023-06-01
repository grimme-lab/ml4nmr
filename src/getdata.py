#!/usr/bin/env python3

# This scripts processes data for an ML model
# Copyright (C) 2021-2023 Julius Kleine Büning


import os
import argparse
import copy
import numpy as np
import statistics
import random
from ase import neighborlist
from ase.io import read
from ase.data import covalent_radii

# for exponential fitting (CBS extrapolation)
from scipy.optimize import curve_fit


### GLOBAL DECLARATIONS ###

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

# number of digits of compounds (dc, e.g. 042) and structures (ds, e.g. 04)
dc = 3
ds = 2

# the following is used for ACSF and SOAP descriptors
elements_sym = ('H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl')
cutoff_sym = 5.0   # used for ACSF and SOAP, but may be varied

### END GLOBAL DECLARATIONS ###


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


def read_cfour(outpath):
    """Read the coupled cluster output from CFOUR calculations."""

    with open(outpath, 'r') as inp:
        data = inp.readlines()

    # get the range in which the shieldings are listed
    start = data.index('   CCSD(T) Nuclear Magnetic Resonance Shieldings and Anisotropies\n') + 5
    end = data.index('     HF-SCF Nuclear Magnetic Resonance Shieldings and Anisotropies\n') - 3
    
    shieldings = []
    for line in data[start:end]:
        tmp = line.split()
        shieldings.append({
            'nuc': int(tmp[0]),
            'elem': tmp[1].upper(),
            'val': float(tmp[2])
        })

    # make sure the shieldings are ordered according to the atom numbering (store as tuple)
    return tuple(sorted(shieldings, key=lambda s: s['nuc']))


# get number of calculated NMR nuclei
# outlist is a list of lines read from a calculation output file
def get_number_nmr(outlist):
    
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


# Get the reference shieldings from the coupled cluster output (CFOUR)
# ATTENTION: all Hs and all Cs are averaged, works e.g. for TMS and CH4
def getref_cfour(path, ref, basis):
    
    with open(os.path.join(path, ref, "ccsd_t", basis, "cfour.out"), 'r') as inp:
        data = inp.readlines()

    # get the range in which the shieldings are listed
    start = data.index('   CCSD(T) Nuclear Magnetic Resonance Shieldings and Anisotropies\n') + 5
    end = data.index('     HF-SCF Nuclear Magnetic Resonance Shieldings and Anisotropies\n') - 3
    
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


# Get the reference shieldings from the DFT output (ORCA)
# ATTENTION: all Hs and all Cs are averaged, works e.g. for TMS and CH4
def getref_orca(path):

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


def inv_cubic(x, a, b):
    return a / x**3 + b


def extrapolate(cc_3z, dft_3z, dft_4z, dft_5z):

    cardinal = np.array([4, 5, 6])
    shielding = np.array([dft_3z, dft_4z, dft_5z])

    xdata = cardinal
    ydata = shielding

    params_cub, _ = curve_fit(inv_cubic, xdata, ydata, p0=[10,100])   # params_cub = [a b]
    dft_cbs = params_cub[1]

    return cc_3z + (dft_cbs - dft_3z)


# read the molecule and return mol (ase.Atoms object) and dict neighbors
def read_mol(structpath):

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


# get the coordination number (CN) as number of neighbors (input is neighbors from read_mol)
# read_mol() ensures that every H atom has exactly one neighbor
def get_cn(neighbors):
    cn = {}
    for key, value in neighbors.items():
        cn[key] = len(value)

    return cn


# get the coordination number (CN) like it is done in the D3 dispersion correction model:
# CN(A) = sum_B!=A^Nat ( 1 / (1 + exp(-k1 (k2*R_AB^cov/R_AB - 1))) )
# with A, B: atom indices, Nat: number of atoms in molecule,
# R_AB: distance of A and B according to input structure
# k2*R_AB^cov = k2*(R_A^cov + R_B^cov): element-specific covalent atom radii from Pyykkö and Atsumi scaled with k2 = 4/3
# k1 = 16: scaling factor; both k1 and k2 are set in the D3 model (J. Chem. Phys. 132, 154104); covalent_rad_d3 already include k2 (see above)
def get_cn_d3(mol):
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


# in the eyes of a certain atom, get the number of bonded X atoms (X = H, C, N, O, S, Cl)
# retruns a dict with all atoms [no_H, no_C, no_N, no_O, no_S, no_Cl]
# if only_for is not empty, then get no_atoms only for chosen nuclei (e.g. [1, 6] for only H and C)
def get_no_bond_atoms(mol, neighbors, only_for=[]):
    no_atoms = {}
    elements = [1, 6, 7, 8, 16, 17]   # H, C, N, O, S, Cl

    for key, value in neighbors.items():
        # if only_for is not empty, make sure the atomic number of key fits entries in only_for
        if len(only_for) == 0 or mol.get_atomic_numbers()[key-1] in only_for:
            n = [0, 0, 0, 0, 0, 0]   # number of H, C, N, O, S, Cl
            for neigh in value:
                anx = mol.get_atomic_numbers()[neigh-1]
                for i, elem in enumerate(elements):
                    if anx == elem:
                        n[i] += 1
            no_atoms[key] = n
    
    return no_atoms


# in the eyes of a certain atom, get the number of secondarily bonded X atoms (X = H, C, N, O, S, Cl)
# retruns a dict with all atoms [no_H, no_C, no_N, no_O, no_S, no_Cl]
# if only_for is not empty, then get no_atoms only for chosen nuclei (e.g. [1, 6] for only H and C)
def get_no_sec_atoms(mol, neighbors, only_for=[]):
    no_atoms = {}
    elements = [1, 6, 7, 8, 16, 17]   # H, C, N, O, S, Cl
    cn = get_cn(neighbors)            # in this context, coordination number means number of directly bonded atoms
    prim_bonded = get_no_bond_atoms(mol, neighbors)

    for key, value in neighbors.items():
        # if only_for is not empty, make sure the atomic number of key fits entries in only_for
        if len(only_for) == 0 or mol.get_atomic_numbers()[key-1] in only_for:
            n = [0, 0, 0, 0, 0, 0]   # number of H, C, N, O, S, Cl

            # for all neighbors of key, add all entries of prim_bonded to n
            for neigh in value:
                for i in range(len(elements)):
                    n[i] += prim_bonded.get(neigh)[i]

            # now, atom key has falsely been counted cn(key) times (if it occurs in elements)
            this_an = mol.get_atomic_numbers()[key-1]   # the atomic number of atom key
            this_cn = cn.get(key)                       # the CN of atom key is 1 for H atoms
            # subtract respective entries in n
            if this_an in elements:
                n[elements.index(this_an)] -= this_cn

            no_atoms[key] = n
    
    return no_atoms


def get_atomic_charges(mol, outputpath, mode):
    """Get atomic charges from ORCA output.

    everything is stored in dicts {index: charge}
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


def get_orbital_charges(outputpath, mode):
    """Get orbital charges form ORCA output.

    for H: s-/p-orbital charges; for C also d-orbital charges
    also get the standard deviation of the 3 p orbital charges (px/py/pz) as mean of uniform charge distribution
    everything is stored in dicts {index: charge}
    mode: can be 'mulliken' or 'loewdin'
    """

    s_charges = {}
    p_charges = {}
    d_charges = {}
    px_charges = {}
    py_charges = {}
    pz_charges = {}
    p_stdev = {}

    if mode == 'mulliken':
        pattern = 'MULLIKEN REDUCED ORBITAL CHARGES\n'
        offset = 2
    elif mode == 'loewdin':
        pattern = 'LOEWDIN REDUCED ORBITAL CHARGES\n'
        offset = 2
    else:
        print("ERROR: unknown mode in get_orbital_charges!")
        exit()

    with open(outputpath, 'r') as inp:
        data = inp.readlines()
    
    # get the range in which the orbital charges of the given pattern are listed
    start = data.index(pattern) + offset
    end = get_end(data, start)

    # go through the lines and store the sums of s/p/d orbital entries (last column each) in the dicts
    for index, line in enumerate(data[start:end]):
        tmp = line.split()

        if tmp[0].isdigit():
            # atom number and next lines
            atnum = int(tmp[0]) + 1
            tmp_next1 = data[start+index+1].split()
            tmp_next2 = data[start+index+2].split()
            tmp_next3 = data[start+index+3].split()
            # s- and p-orbital charges
            s_charges[atnum] = float(tmp[-1])
            p_charges[atnum] = float(tmp_next1[5])
            pz_charges[atnum] = float(tmp_next1[2])
            px_charges[atnum] = float(tmp_next2[2])
            py_charges[atnum] = float(tmp_next3[2])

            # d-orbital charge (only for carbon)
            if tmp[1] == 'C':
                tmp_next4 = data[start+index+4].split()
                d_charges[atnum] = float(tmp_next4[5])

    for at in p_charges.keys():
        p_stdev[at] = statistics.stdev([px_charges[at], py_charges[at], pz_charges[at]])
        #p_stdev[at] = np.std([px_charges[at], py_charges[at], pz_charges[at]], ddof=1)

    return s_charges, p_charges, d_charges, p_stdev


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
    """get bond orders for of Loewdin and Mayer type.
    
    
    reads bond orders and calculates the sum and average for each atom
    returns list of dicts
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
        #tmp = list(filter(lambda c: c not in [',', 'B(', ')', ':'], line.split()))   # old way to deal with the problem
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


# in addition to the total isotropic shielding constant (read by read_orca), get the following additional NMR quantities:
# diamagnetic shielding constant, paramagnetic shielding constant, span, skew, asymmetry, anisotropy
def get_nmr_quantities(outputpath):

    with open(outputpath, 'r') as inp:
        data = inp.readlines()

    # get number of calculated NMR nuclei
    n_nmr = get_number_nmr(data)

    # get start of the chemical shift block
    start = data.index('CHEMICAL SHIFTS\n') + 6

    # data is arranged in blocks of 27 lines for every nucleus
    # line 21/22 contain dia-/paramagnetic shielding contants, 24 the total shielding with its components
    quantities = []
    for at in range(n_nmr):
        
        atnum = ""
        for c in data[start+at*27].split()[1]:
            if c.isdigit(): atnum += c
        atnum = int(atnum) + 1
        tmp = data[start+at*27+24].split()
        quantities.append({
            'at_index': atnum,
            'shielding_dia': data[start+at*27+21].split()[5],
            'shielding_para': data[start+at*27+22].split()[5],
            'sigma_components': [float(tmp[i+1]) for i in range(3)]
        })

    # calculate the quantities for all atoms
    for quant in quantities:
        
        # from shielding tensor sigma get min, mid and max value of the diagonal sigma_ii elements and sigma_iso
        sigma = quant['sigma_components'].copy()
        sigma_iso = sum(sigma)/len(sigma)
        sigma_min = min(sigma)
        sigma.remove(sigma_min)
        sigma_mid = min(sigma)
        sigma_max = max(sigma)

        # get span: Omega = sigma_max - sigma_min (>= 0)
        span = sigma_max - sigma_min

        # escape somehow if span == 0.0 (all other quantities are 0.0, too)
        if span < 0.0001:
            skew = 0.0
            asymmetry = 0.0
            anisotropy = 0.0
        else:
            # get skew: kappa = 3*(sigma_iso - sigma_mid)/Omega (-1 <= kappa <= 1)
            skew = 3*(sigma_iso - sigma_mid)/span

            # get asymmetry: eta = (sigma_mid - sigma_min)/(sigma_max - sigma_iso)
            asymmetry = (sigma_mid - sigma_min)/(sigma_max - sigma_iso)

            # get anisotropy: Delta = sigma_max - (sigma_min + sigma_mid)/2
            anisotropy = sigma_max - (sigma_min + sigma_mid)/2

        quant['span'] = span
        quant['skew'] = skew
        quant['asymmetry'] = asymmetry
        quant['anisotropy'] = anisotropy

    return tuple(sorted(quantities, key=lambda q: q['at_index']))


def get_reference(path_data, ref, func, basis, high=None):

    has_highlevel = False
    if high is not None: has_highlevel = True

    # get the reference shielding values (e.g. from TMS molecule)
    refshieldings = getref_orca(os.path.join(path_data, ref, func, basis))
    print("using low-level DFT 1H reference shielding ({}): {} ppm".format(ref, refshieldings['H']))
    print("using low-level DFT 13C reference shielding ({}): {} ppm\n".format(ref, refshieldings['C']))

    refshieldings_high = {'H': None, 'C': None}
    if has_highlevel:
        # get high-level shieldings of reference compound if provided
        refshieldings_high_cc = getref_cfour(path_data, ref, high['basis_3z'])
        refshieldings_high_dft_cc = getref_orca(os.path.join(path_data, ref, high['functional'], high['basis_3z']))
        refshieldings_high_dft_x = getref_orca(os.path.join(path_data, ref, high['functional'], high['basis_4z']))
        refshieldings_high_dft_y = getref_orca(os.path.join(path_data, ref, high['functional'], high['basis_5z']))
        refshieldings_high_cc_cbs = {
            'H': extrapolate(refshieldings_high_cc['H'], refshieldings_high_dft_cc['H'], refshieldings_high_dft_x['H'], refshieldings_high_dft_y['H']),
            'C': extrapolate(refshieldings_high_cc['C'], refshieldings_high_dft_cc['C'], refshieldings_high_dft_x['C'], refshieldings_high_dft_y['C'])
        }

        print("using high-level CCSD(T)/TZ+ 1H reference shielding ({}): {} ppm".format(ref, refshieldings_high_cc_cbs['H']))
        print("using high-level CCSD(T)/TZ+ 13C reference shielding ({}): {} ppm\n".format(ref, refshieldings_high_cc_cbs['C']))

        refshieldings_high = {'H': refshieldings_high_cc_cbs['H'], 'C': refshieldings_high_cc_cbs['C']}

    return refshieldings, refshieldings_high


# Read all data from a molecule and process it to get the shieldings and shifts of all atoms
# Save all information in two lists of dicts (data_h, data_c) and return further data to extend the ML input with
def get_data(path_xyz, path_out, name, ref_shield, include, high=None, ref_shield_high={'H': None, 'C': None}, path_data=None, print_names=False):

    has_highlevel = False
    if high is not None: has_highlevel = True

    data_h = []
    data_c = []

    # read the molecular data from the .xyz file and get the number of atoms and list of element symbols
    mol, neighbors = read_mol(path_xyz)
    nat = len(mol)

    # get shieldings of the sample compound from ORCA output
    shieldings = read_orca(path_out)
    nmr_nuc = [s['nuc'] for s in shieldings]

    if has_highlevel:
        # get high-level shieldings if provided
        shieldings_high_cc_full = read_cfour(os.path.join(path_data, 'ccsd_t', high['basis_3z'], 'cfour.out'))
        shieldings_high_dft_cc = read_orca(os.path.join(path_data, high['functional'], high['basis_3z'], 'orca.out'))
        shieldings_high_dft_x = read_orca(os.path.join(path_data, high['functional'], high['basis_4z'], 'orca.out'))
        shieldings_high_dft_y = read_orca(os.path.join(path_data, high['functional'], high['basis_5z'], 'orca.out'))
        # reduce the list to only the relevant NMR nuclei (only necessary for shielding from read_cfour)
        shieldings_high_cc = tuple([s for s in shieldings_high_cc_full if s['nuc'] in nmr_nuc])
    
    # get some data needed as descriptors
    bond_atoms = get_no_bond_atoms(mol, neighbors)
    sec_atoms = get_no_sec_atoms(mol, neighbors)
    cn_d3 = get_cn_d3(mol)
    distances = mol.get_all_distances()

    # get a buch of ML descriptors from the orca output (mainly based on density matrix and NMR properties)
    at_chrg_mulliken = get_atomic_charges(mol, path_out, 'mulliken')
    at_chrg_loewdin = get_atomic_charges(mol, path_out, 'loewdin')
    orb_chrg_mulliken_s, orb_chrg_mulliken_p, orb_chrg_mulliken_d, orb_stdev_mulliken_p = get_orbital_charges(path_out, 'mulliken')
    orb_chrg_loewdin_s, orb_chrg_loewdin_p, orb_chrg_loewdin_d, orb_stdev_loewdin_p = get_orbital_charges(path_out, 'loewdin')
    mayer_VA, _, _ = get_Mayer_pop(mol, path_out)
    bond_orders_loewdin_sum, bond_orders_loewdin_av = get_bond_orders(neighbors, path_out, 'loewdin')
    bond_orders_mayer_sum, bond_orders_mayer_av = get_bond_orders(neighbors, path_out, 'mayer')
    nmr_quantities = get_nmr_quantities(path_out)

    # get the symmetric fingerprint descriptors
    if 'acsf' in include:
        acsf_mol = descriptors.acsf.create(mol)
    if 'soap' in include:
        soap_mol = descriptors.soap.create(mol)
        soap_mol = np.delete(soap_mol, descriptors.soap_zero_indexlist, 1)

    # loop over all atoms in the compound
    for iat in range(len(nmr_nuc)):

        high_cc_3z = high_dft_3z = high_dft_4z = high_dft_5z = None

        if has_highlevel:
            # ensure 'nuc' and 'elem' is the same for all the data (shieldings_high_cc, _dft_cc, _dft_x, _dft_y and the low level sample)
            nucs = [shieldings_high_cc[iat]['nuc'], shieldings_high_dft_cc[iat]['nuc'], shieldings_high_dft_x[iat]['nuc'], shieldings_high_dft_y[iat]['nuc'], shieldings[iat]['nuc']]
            elems = [shieldings_high_cc[iat]['elem'], shieldings_high_dft_cc[iat]['elem'], shieldings_high_dft_x[iat]['elem'], shieldings_high_dft_y[iat]['elem'], shieldings[iat]['elem']]
            if not all_equal(nucs):
                print("ERROR with sample compound {}: nuclei order in CFOUR and ORCA outputs is not the same!".format(name))
                exit()
            if not all_equal(elems):
                print("ERROR with sample compound {}: element order in CFOUR and ORCA outputs is not the same!".format(name))
                exit()
            
            high_cc_3z = shieldings_high_cc[iat]['val']
            high_dft_3z = shieldings_high_dft_cc[iat]['val']
            high_dft_4z = shieldings_high_dft_x[iat]['val']
            high_dft_5z = shieldings_high_dft_y[iat]['val']

        # if the actual atom is a H atom, get the distance to the neighboring C atom and its computed chemical shift
        if shieldings[iat]['elem'] == 'H':
            # get the atom number of the neighbor (at starts from 0, but neighbors from 1; -1 to adjust neigh to at scale)
            neigh = neighbors[nmr_nuc[iat]][0]   # there is only one neighbor because this is a H atom
            # skip the whole data point if the H atom is bound to something else than C (atomic number 6) (this can be extended later)
            if not mol.get_atomic_numbers()[neigh-1] == 6: continue
            # get the distance between the actual atom and its neigbor (distances list starts from 0)
            dist = distances[nmr_nuc[iat]-1][neigh-1]
            # get the shift of the neighoring C atom
            neighshift = ref_shield['C'] - next((s for s in shieldings if s['nuc'] == neigh), None)['val']

        # add all ACSF and SOAP descriptors
        # acsf_mol and soap_mol need the atom index, but start with 0 so nmr_nuc-1
        if 'acsf' in include:
            acsf_data = {acsf_name: acsf_mol[nmr_nuc[iat]-1][i] for i, acsf_name in enumerate(descriptors.acsf_labels)}
        if 'soap' in include:
            soap_data = {soap_name: soap_mol[nmr_nuc[iat]-1][i] for i, soap_name in enumerate(descriptors.soap_labels_reduced)}

        if shieldings[iat]['elem'] == 'H':

            datapoint = DataPointH(name, nmr_nuc[iat], print_names)
            datapoint.set_attr_general(
                shieldings[iat]['val'],
                at_chrg_mulliken[nmr_nuc[iat]],
                at_chrg_loewdin[nmr_nuc[iat]],
                orb_chrg_mulliken_s[nmr_nuc[iat]],
                orb_chrg_loewdin_s[nmr_nuc[iat]],
                orb_chrg_mulliken_p[nmr_nuc[iat]],
                orb_chrg_loewdin_p[nmr_nuc[iat]],
                mayer_VA[nmr_nuc[iat]],
                nmr_quantities[iat]['shielding_dia'],
                nmr_quantities[iat]['shielding_para'],
                nmr_quantities[iat]['span'],
                nmr_quantities[iat]['skew'],
                nmr_quantities[iat]['asymmetry'],
                nmr_quantities[iat]['anisotropy'],
                high_cc_3z,
                high_dft_3z,
                high_dft_4z,
                high_dft_5z
            )
            if 'acsf' in include: datapoint.set_acsf(acsf_data)
            if 'soap' in include: datapoint.set_soap(soap_data)
            datapoint.set_attr_special(
                ref_shield['H'],
                neighshift,
                cn_d3[neigh],                                     # the D3 CN of the neighboring C atom
                bond_atoms[neigh][0],                             # number of H bonded to neighboring C
                [sec_atoms[nmr_nuc[iat]][i] for i in range(4)],   # number of secondarily bonded [H, C, N, O] (first 4 elements of sec_atoms)
                dist,
                bond_orders_loewdin_sum[nmr_nuc[iat]],            # starts counting with 1
                bond_orders_mayer_sum[nmr_nuc[iat]],              # starts counting with 1
                ref_shield_high['H']
            )

            data_h.append(datapoint)


        if shieldings[iat]['elem'] == 'C':

            datapoint = DataPointC(name, nmr_nuc[iat], print_names)
            datapoint.set_attr_general(
                shieldings[iat]['val'],
                at_chrg_mulliken[nmr_nuc[iat]],
                at_chrg_loewdin[nmr_nuc[iat]],
                orb_chrg_mulliken_s[nmr_nuc[iat]],
                orb_chrg_loewdin_s[nmr_nuc[iat]],
                orb_chrg_mulliken_p[nmr_nuc[iat]],
                orb_chrg_loewdin_p[nmr_nuc[iat]],
                mayer_VA[nmr_nuc[iat]],
                nmr_quantities[iat]['shielding_dia'],
                nmr_quantities[iat]['shielding_para'],
                nmr_quantities[iat]['span'],
                nmr_quantities[iat]['skew'],
                nmr_quantities[iat]['asymmetry'],
                nmr_quantities[iat]['anisotropy'],
                high_cc_3z,
                high_dft_3z,
                high_dft_4z,
                high_dft_5z
            )
            if 'acsf' in include: datapoint.set_acsf(acsf_data)
            if 'soap' in include: datapoint.set_soap(soap_data)
            datapoint.set_attr_special(
                ref_shield['C'],
                cn_d3[nmr_nuc[iat]],                               # the D3 CN of the C atom
                [bond_atoms[nmr_nuc[iat]][i] for i in range(4)],   # number of bonded [H, C, N, O] (first 4 elements of bond_atoms)
                [sec_atoms[nmr_nuc[iat]][i] for i in range(4)],    # number of secondarily bonded [H, C, N, O] (first 4 elements of sec_atoms)
                orb_chrg_mulliken_d[nmr_nuc[iat]],                 # starts counting with 1
                orb_chrg_loewdin_d[nmr_nuc[iat]],                  # starts counting with 1
                orb_stdev_mulliken_p[nmr_nuc[iat]],                # starts counting with 1
                orb_stdev_loewdin_p[nmr_nuc[iat]],                 # starts counting with 1
                bond_orders_loewdin_sum[nmr_nuc[iat]],             # starts counting with 1
                bond_orders_mayer_sum[nmr_nuc[iat]],               # starts counting with 1
                bond_orders_loewdin_av[nmr_nuc[iat]],              # starts counting with 1
                bond_orders_mayer_av[nmr_nuc[iat]],                # starts counting with 1
                ref_shield_high['C']
            )

            data_c.append(datapoint)


    # define metadata to be added to the ML input
    extension = [
        "# nat: {}".format(nat),
        "# n_nmr_nuc: {}".format(len(nmr_nuc)),
        "# ref_h: {}".format(ref_shield['H']),
        "# ref_c: {}".format(ref_shield['C'])
    ]

    return data_h, data_c, "\n".join(extension)


# shuffles a datalist in optional groups of atoms (=no groups), structures or compounds
# datalist is a list of DataPointH and DataPointC objects
def shuffle_data(datalist, compounds, structures, mode='structures'):

    if mode == 'atoms':
        new_datalist = copy.deepcopy(datalist)
        random.Random(random_seed).shuffle(new_datalist)

    if mode == 'structures':
        metalist = []
        for comp in compounds:
            for struct in structures[comp]:
                tmp = []
                for data in datalist:
                    if int(data.name[:dc]) == comp and int(data.name[-ds:]) == struct: tmp.append(data)
                metalist.append(tmp)
        random.Random(random_seed).shuffle(metalist)
        # the following combines lists in a list to one list (e.g. [[1,2],[3,4]] -> [1,2,3,4])
        new_datalist = [i for j in metalist for i in j]

    if mode == 'compounds':
        metalist = []
        for comp in compounds:
            tmp = []
            for data in datalist:
                if int(data.name[:dc]) == comp: tmp.append(data)
            metalist.append(tmp)
        random.Random(random_seed).shuffle(metalist)
        # the following combines lists in a list to one list (e.g. [[1,2],[3,4]] -> [1,2,3,4])
        new_datalist = [i for j in metalist for i in j]

    return new_datalist


# Write full 1H/13C data from datalist into a file used as input for 1H/13C ML correction
# mode can be 'H' or 'C' for 1H or 13C data; is_dataset is True for ML dataset or False for a sample molecule
# additionally add an extension at the end of the file
# (this is useful for a sample compound, where the dataset is a list of only one compound, to provide information for the ML script)
# ATTENTION: This only gives reasonable output if datalist is purely H or purely C!
def write_ml_input(datalist, outpath, outname, extension="", is_sample=False):

    atnums = "# atom_numbers:"
    printout = datalist[0].get_header()
    for data in datalist:
        printout.append(data.get_printout())
        if is_sample: atnums += " " + str(data.atom)
    if is_sample: printout.append(atnums)

    with open(os.path.join(outpath, outname), 'w') as out:
        out.write("\n".join(printout) + "\n")
        out.write(extension)


class DataPoint:

    def __init__(self, name, atom, print_names=False):
        self.name = name
        self.atom = atom
        self.print_names = print_names
        self.include_acsf = False
        self.include_soap = False

    def set_attr_general(
        self, shielding_low,
        atomic_charge_mulliken, atomic_charge_loewdin,
        orbital_charge_mulliken_s, orbital_charge_loewdin_s,
        orbital_charge_mulliken_p, orbital_charge_loewdin_p,
        mayer_valence_total,
        shielding_diamagnetic, shielding_paramagnetic,
        span, skew, asymmetry, anisotropy,
        shielding_cc_3z=None, shielding_dft_3z=None, shielding_dft_4z=None, shielding_dft_5z=None
    ):
        self.shielding_low = shielding_low
        self.atomic_charge_mulliken = atomic_charge_mulliken
        self.atomic_charge_loewdin = atomic_charge_loewdin
        self.orbital_charge_mulliken_s = orbital_charge_mulliken_s
        self.orbital_charge_loewdin_s = orbital_charge_loewdin_s
        self.orbital_charge_mulliken_p = orbital_charge_mulliken_p
        self.orbital_charge_loewdin_p = orbital_charge_loewdin_p
        self.mayer_valence_total = mayer_valence_total
        self.shielding_diamagnetic = shielding_diamagnetic
        self.shielding_paramagnetic = shielding_paramagnetic
        self.span = span
        self.skew = skew
        self.asymmetry = asymmetry
        self.anisotropy = anisotropy
        self.shielding_cc_3z = shielding_cc_3z
        self.shielding_dft_3z = shielding_dft_3z
        self.shielding_dft_4z = shielding_dft_4z
        self.shielding_dft_5z = shielding_dft_5z
        if None in [self.shielding_cc_3z, self.shielding_dft_3z, self.shielding_dft_4z, self.shielding_dft_5z]:
            self.shielding_cc_ext = None
        else:
            self.shielding_cc_ext = extrapolate(self.shielding_cc_3z, self.shielding_dft_3z, self.shielding_dft_4z, self.shielding_dft_5z)

    def set_acsf(self, acsf_data):
        self.include_acsf = True
        self.acsf_data = acsf_data

    def set_soap(self, soap_data):
        self.include_soap = True
        self.soap_data = soap_data


class DataPointH(DataPoint):

    def set_attr_special(
        self, refshielding_low, shift_neighbor,
        cn_d3, number_hch, number_hyx,
        distance_hc, bond_order_loewdin, bond_order_mayer,
        refshielding_cc_ext=None
    ):
        self.element = 'H'
        self.shift_low = refshielding_low - self.shielding_low
        self.shift_neighbor = shift_neighbor
        self.cn_d3 = cn_d3
        self.number_hch = number_hch
        self.number_hyx = number_hyx
        self.distance_hc = distance_hc
        self.bond_order_loewdin = bond_order_loewdin
        self.bond_order_mayer = bond_order_mayer
        if refshielding_cc_ext is None:
            self.shift_cc_ext = self.deviation = None
        else:
            self.shift_cc_ext = refshielding_cc_ext - self.shielding_cc_ext
            self.deviation = self.shift_cc_ext - self.shift_low


    def get_header(self):

        nvar = 25
        if self.print_names: nvar += 3
        if self.include_acsf: nvar += len(descriptors.acsf_labels)
        if self.include_soap: nvar += len(descriptors.soap_labels_reduced)

        header_1 = "# " + " ".join([str(i) for i in range(nvar)])
        header_2 = "# shift_high-low shift_low CN(X) no_HCH no_HYH no_HYC no_HYN no_HYO dist_HC shift_low_neighbor_C shielding_dia shielding_para span skew asymmetry anisotropy at_charge_mull at_charge_loew orb_charge_mull_s orb_charge_mull_p orb_charge_loew_s orb_charge_loew_p BO_loew BO_mayer mayer_VA"
        if self.print_names: header_2 = "# compound structure atom" + header_2[1:]
        if self.include_acsf: header_2 += " " + " ".join(descriptors.acsf_labels)
        if self.include_soap: header_2 += " " + " ".join(descriptors.soap_labels_reduced)

        return [header_1, header_2]


    def get_printout(self):

        if self.print_names:
            beginning = [self.name[:3], self.name[4:], self.atom]
        else:
            beginning = []

        line = (
            beginning
            + [self.deviation, self.shift_low, self.cn_d3, self.number_hch]
            + self.number_hyx
            + [
                self.distance_hc, self.shift_neighbor,
                self.shielding_diamagnetic, self.shielding_paramagnetic,
                self.span, self.skew, self.asymmetry, self.anisotropy,
                self.atomic_charge_mulliken, self.atomic_charge_loewdin,
                self.orbital_charge_mulliken_s, self.orbital_charge_mulliken_p,
                self.orbital_charge_loewdin_s, self.orbital_charge_loewdin_p,
                self.bond_order_loewdin, self.bond_order_mayer, self.mayer_valence_total
            ]
        )

        if self.include_acsf: line += [self.acsf_data[acsf_name] for acsf_name in descriptors.acsf_labels]
        if self.include_soap: line += [self.soap_data[soap_name] for soap_name in descriptors.soap_labels_reduced]

        return " ".join([str(val) for val in line])


class DataPointC(DataPoint):

    def set_attr_special(
        self, refshielding_low, cn_d3, number_cx, number_cyx,
        orbital_charge_mulliken_d, orbital_charge_loewdin_d, orbital_stdev_mulliken_p, orbital_stdev_loewdin_p,
        bond_orders_loewdin_sum, bond_orders_mayer_sum, bond_orders_loewdin_av, bond_orders_mayer_av,
        refshielding_cc_ext=None
    ):
        self.element = 'C'
        self.shift_low = refshielding_low - self.shielding_low
        self.cn_d3 = cn_d3
        self.number_cx = number_cx
        self.number_cyx = number_cyx
        self.orbital_charge_mulliken_d = orbital_charge_mulliken_d
        self.orbital_charge_loewdin_d = orbital_charge_loewdin_d
        self.orbital_stdev_mulliken_p = orbital_stdev_mulliken_p
        self.orbital_stdev_loewdin_p = orbital_stdev_loewdin_p
        self.bond_orders_loewdin_sum = bond_orders_loewdin_sum
        self.bond_orders_mayer_sum = bond_orders_mayer_sum
        self.bond_orders_loewdin_av = bond_orders_loewdin_av
        self.bond_orders_mayer_av = bond_orders_mayer_av
        if refshielding_cc_ext is None:
            self.shift_cc_ext = self.deviation = None
        else:
            self.shift_cc_ext = refshielding_cc_ext - self.shielding_cc_ext
            self.deviation = self.shift_cc_ext - self.shift_low


    def get_header(self):

        nvar = 32
        if self.print_names: nvar += 3
        if self.include_acsf: nvar += len(descriptors.acsf_labels)
        if self.include_soap: nvar += len(descriptors.soap_labels_reduced)

        header_1 = "# " + " ".join([str(i) for i in range(nvar)])
        header_2 = "# shift_high-low shift_low CN(X) no_CH no_CC no_CN no_CO no_CYH no_CYC no_CYN no_CYO shielding_dia shielding_para span skew asymmetry anisotropy at_charge_mull at_charge_loew orb_charge_mull_s orb_charge_mull_p orb_charge_mull_d orb_stdev_mull_p orb_charge_loew_s orb_charge_loew_p orb_charge_loew_d orb_stdev_loew_p BO_loew_sum BO_loew_av BO_mayer_sum BO_mayer_av mayer_VA"
        if self.print_names: header_2 = "# compound structure atom" + header_2[1:]
        if self.include_acsf: header_2 += " " + " ".join(descriptors.acsf_labels)
        if self.include_soap: header_2 += " " + " ".join(descriptors.soap_labels_reduced)

        return [header_1, header_2]


    def get_printout(self):

        if self.print_names:
            beginning = [self.name[:3], self.name[4:], self.atom]
        else:
            beginning = []

        line = (
            beginning
            + [self.deviation, self.shift_low, self.cn_d3]
            + self.number_cx + self.number_cyx
            + [
                self.shielding_diamagnetic, self.shielding_paramagnetic,
                self.span, self.skew, self.asymmetry, self.anisotropy,
                self.atomic_charge_mulliken, self.atomic_charge_loewdin,
                self.orbital_charge_mulliken_s, self.orbital_charge_mulliken_p, self.orbital_charge_mulliken_d, self.orbital_stdev_mulliken_p,
                self.orbital_charge_loewdin_s, self.orbital_charge_loewdin_p, self.orbital_charge_loewdin_d, self.orbital_stdev_loewdin_p,
                self.bond_orders_loewdin_sum, self.bond_orders_loewdin_av, self.bond_orders_mayer_sum, self.bond_orders_mayer_av,
                self.mayer_valence_total,
            ]
        )

        if self.include_acsf: line += [self.acsf_data[acsf_name] for acsf_name in descriptors.acsf_labels]
        if self.include_soap: line += [self.soap_data[soap_name] for soap_name in descriptors.soap_labels_reduced]

        return " ".join([str(val) for val in line])


if __name__ == "__main__":

    workdir = os.getcwd()

    # initialize some proper parser for the command line input
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--set', type=str, help='choose to extract data from the test set compounds; provide the path to the data set directory')
    parser.add_argument('-a', '--sample', nargs=3, help='choose to extract data from a sample compound; provide structure in .xyz format, an orca output file, and a data directory (path to the data of the desired reference compound)', metavar=('xyz', 'out', 'data'))
    parser.add_argument('-ar', '--sample_ref', nargs=3, help='choose to extract data from a sample compound with high-level reference shifts; provide structure in .xyz format, an orca output file, and a data directory (path to the data of the desired reference compound and the high-level reference data)', metavar=('xyz', 'out', 'data'))
    parser.add_argument('-fl', '--functional_low', default='pbe0', help='functional for low-level DFT NMR shift to be corrected, default: pbe0')
    parser.add_argument('-bl', '--basis_low', default='pcSseg-2', help='basis set for low-level DFT NMR shift to be corrected, default: pcSseg-2')
    parser.add_argument('-r', '--reference', default='tms', help='reference compound for NMR shift, default: tms')
    parser.add_argument('-s', '--shuffle', default='structures', choices=['none', 'atoms', 'structures', 'compounds'], help='shuffle mode for the data set: atoms, structures or compounds, default: structures')
    parser.add_argument('-rs', '--randomseed', type=int, default=0, help='random seed for data shuffling, default: 0')
    parser.add_argument('-pn', '--print_names', action='store_true', help='additionally print names (i.e. compound, structures, and atom numbers) in the data set file (only relevant for --set option)')
    parser.add_argument('-i', '--include', nargs='+', type=str.lower, choices=['acsf', 'soap'], help='optionally include ACSF and/or SOAP features')

    args = parser.parse_args()

    if args.set is not None:
        input_mode = 'set'
        datapath = args.set
    elif args.sample is not None:
        input_mode = 'sample'
        samplepath_xyz, samplepath_out, datapath = args.sample
    elif args.sample_ref is not None:
        input_mode = 'sample_ref'
        samplepath_xyz, samplepath_out, datapath = args.sample_ref
    else:
        print("ERROR: no input mode selected! Please select --set, --sample <str> or --sample_ref <str>")
        exit()

    reference = args.reference
    # The settings for the reference and extrapolation methods are currently fixed
    functional_high = 'bhlyp'
    basis_3z = 'pcSseg-2'
    basis_4z = 'pcSseg-3'
    basis_5z = 'pcSseg-4'

    functional_low = args.functional_low
    basis_low = args.basis_low
    shuffle_mode = args.shuffle
    random_seed = args.randomseed
    print_names = args.print_names

    if args.include is None: include = []
    else: include = include = args.include

    # import the ACSF and SOAP routines within dscribe only when they are needed
    if any(i in include for i in ['acsf', 'soap']): import additional_descriptors as descriptors

    if input_mode == 'set':

        compounds = [c + 1 for c in range(100)]   # compound numbers from 1 to 100
        structures = {}
        for comp in compounds: structures[comp] = [int(f) for f in sorted(os.listdir(os.path.join(datapath, str(comp).zfill(3))))]

        print("Extracting ML descriptors from the data set\n")
        print("Settings:")
        print("Reference coupled-cluser calculation level: CCSD(T)/{}".format(basis_3z))
        print("Functional for DFT CBS extrapolation      : {}".format(functional_high))
        print("Basis sets for DFT CBS extrapolation      : {}, {}, {}".format(basis_3z, basis_4z, basis_5z))
        print("Functional for low-level DFT calculation  : {}".format(functional_low))
        print("Basis set for low-level DFT calculation   : {}".format(basis_low))
        print("Reference compound                        : {}".format(reference))
        print("Shuffle mode                              : {}".format(shuffle_mode))
        print("Random seed for shuffling                 : {}\n".format(random_seed))

        high_level = {
            'functional': functional_high,
            'basis_3z': basis_3z,
            'basis_4z': basis_4z,
            'basis_5z': basis_5z
        }

        ref, ref_high = get_reference(datapath, reference, functional_low, basis_low, high_level)

        data_h = []
        data_c = []
        for comp in compounds:
            print("collecting data from compound {} ({} structures)".format(comp, len(structures[comp])))
            for struct in structures[comp]:
                name = str(comp).zfill(dc) + '_' + str(struct).zfill(ds)
                structpath = os.path.join(datapath, str(comp).zfill(dc), str(struct).zfill(ds))
                path_xyz = os.path.join(structpath, functional_low, basis_low, name + '.xyz')
                path_out = os.path.join(structpath, functional_low, basis_low, 'orca.out')

                dath, datc, _ = get_data(path_xyz, path_out, name, ref, include, high_level, ref_high, structpath, print_names)
                data_h.extend(dath)
                data_c.extend(datc)
        
        n_structures = sum([len(v) for v in structures.values()])
        print("\nAnalyzed a total number of:")
        print("{} compounds".format(len(compounds)))
        print("{} structures".format(n_structures))
        print("{} 1H NMR shifts".format(len(data_h)))
        print("{} 13C NMR shifts".format(len(data_c)))

        # optionally shuffle the data (mode = 'structures' (default), 'compounds', 'atoms', 'none')
        if shuffle_mode == 'none':
            data_h_shuffled = data_h
            data_c_shuffled = data_c
        else:
            data_h_shuffled = shuffle_data(data_h, compounds, structures, mode=shuffle_mode)
            data_c_shuffled = shuffle_data(data_c, compounds, structures, mode=shuffle_mode)

        # add some method data to the end of the data set file
        extension_settings = [
            "# high-level CC level                 : CCSD(T)/{}".format(basis_3z),
            "# functional for DFT CBS extrapolation: {}".format(functional_high),
            "# basis sets for DFT CBS extrapolation: {}, {}, {}".format(basis_3z, basis_4z, basis_5z),
            "# low-level functional (DFT)          : {}".format(functional_low),
            "# low-level basis set (DFT)           : {}".format(basis_low),
            "# NMR reference compound              : {}".format(reference),
            "# shuffle mode                        : {}".format(shuffle_mode),
            "# random seed for shuffling           : {}\n".format(random_seed)
        ]

        write_ml_input(data_h_shuffled, workdir, "ml_" + functional_low + "_" + basis_low + "_h.dat", extension="\n".join(extension_settings))
        write_ml_input(data_c_shuffled, workdir, "ml_" + functional_low + "_" + basis_low + "_c.dat", extension="\n".join(extension_settings))

    elif input_mode in ['sample', 'sample_ref']:

        # define a compound name (use .xyz file and delete .xyz if present)
        sample_name = os.path.abspath(samplepath_xyz).split(os.sep)[-1].replace('.xyz', '')

        print("Extracting ML descriptors from sample compound: {}".format(sample_name))
        print("... using 3D structure provided in:   {}".format(os.path.abspath(samplepath_xyz)))
        print("... using ORCA calculation output in: {}".format(os.path.abspath(samplepath_out)))
        print("... using supplementary data in:      {}\n".format(os.path.abspath(datapath)))
        print("Settings:")
        if input_mode == 'sample_ref':
            print("Reference coupled-cluser calculation level: CCSD(T)/{}".format(basis_3z))
            print("Functional for DFT CBS extrapolation      : {}".format(functional_high))
            print("Basis sets for DFT CBS extrapolation      : {}, {}, {}".format(basis_3z, basis_4z, basis_5z))
            print("Functional for low-level DFT calculation  : {}".format(functional_low))
            print("Basis set for low-level DFT calculation   : {}".format(basis_low))
            print("Reference compound                        : {}\n".format(reference))
        elif input_mode == 'sample':
            print("Functional for low-level DFT calculation: {}".format(functional_low))
            print("Basis set for low-level DFT calculation : {}".format(basis_low))
            print("Reference compound                      : {}\n".format(reference))

        # get all the data
        if input_mode == 'sample_ref':
            high_level = {
                'functional': functional_high,
                'basis_3z': basis_3z,
                'basis_4z': basis_4z,
                'basis_5z': basis_5z
            }
            ref, ref_high = get_reference(datapath, reference, functional_low, basis_low, high_level)
            data_h, data_c, extension = get_data(samplepath_xyz, samplepath_out, sample_name, ref, include, high_level, ref_high, datapath)
        elif input_mode == 'sample':
            ref, ref_high = get_reference(datapath, reference, functional_low, basis_low)
            data_h, data_c, extension = get_data(samplepath_xyz, samplepath_out, sample_name, ref, include)

        print("\nAnalyzed a total number of:")
        print("{} 1H NMR shifts".format(len(data_h)))
        print("{} 13C NMR shifts".format(len(data_c)))

        # add some method data to the beginning of the extension
        extension_settings = [
            "# low-level functional (DFT): {}".format(functional_low),
            "# low-level basis set (DFT): {}".format(basis_low),
            "# NMR reference compound: {}".format(reference),
        ]
        if input_mode == 'sample_ref':
            extension_refsettings = [
                "# high-level CC level: CCSD(T)/{}".format(basis_3z),
                "# functional for DFT CBS extrapolation: {}".format(functional_high),
                "# basis sets for DFT CBS extrapolation: {}, {}, {}".format(basis_3z, basis_4z, basis_5z)
            ]
            extension_settings = extension_refsettings + extension_settings
        extension = "\n".join(extension_settings) + "\n" + extension + "\n"

        if input_mode == 'sample_ref':
            write_ml_input(data_h, workdir, "ml_" + sample_name + "_ref_h.dat", extension=extension, is_sample=True)
            write_ml_input(data_c, workdir, "ml_" + sample_name + "_ref_c.dat", extension=extension, is_sample=True)
        else:
            write_ml_input(data_h, workdir, "ml_" + sample_name + "_h.dat", extension=extension, is_sample=True)
            write_ml_input(data_c, workdir, "ml_" + sample_name + "_c.dat", extension=extension, is_sample=True)

