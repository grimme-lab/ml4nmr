#!/usr/bin/env python3

# =============================================================================
#
# This scripts read, processes, and prints data for the Delta_SO_ML model
# https://doi.org/10.1039/D3CP05556F
#
# Run:
# getdata_so.py [options]
#
# Copyright (C) 2023-2024 Julius Kleine Büning
#
# =============================================================================


import os
import argparse
import copy
import json
import numpy as np
import statistics
import random

from functions_getdata import *
from classes_so import soDataPointH, soDataPointC
import additional_so as addso


########## GLOBAL DECLARATIONS ##########

# number of digits of compounds (dc, e.g. 0142) and structures (ds, e.g. 03)
dc = 4
ds = 2

# list of known basis set choices that include ECPs:
# ATTENTION: if other strings are chosen, the settings for SR-ZORA are used
basis_sets_ecp = ('def2-TZVP')

# list of all heavy elements handeled by the model (Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi)
heavy_atoms = (17, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 53, 80, 81, 82, 83)
heavy_atom_symbols = ('Cl', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Hg', 'Tl', 'Pb', 'Bi')

def number_core_electrons(atomic_number: int, _is_ecp=False) -> int:
    """Get the number of core electrons in s-/p-/d-orbitals needed to determine the number of respective valence electrons.
    
    This is, e.g., used in get_data_rel.
    """

    if atomic_number < 11 or atomic_number > 86:
        print("ERROR: Function number_core_electrons can only handle elements with an atomic number from 11 (Na) to 86 (Rn)!")
        exit()
    
    no_el = [None, None, None]   # s, p, d
    if 11 <= atomic_number <= 18: no_el = [4, 6, 0]
    elif 19 <= atomic_number <= 36: no_el = [6, 12, 0]
    elif 37 <= atomic_number <= 54: no_el = [8, 18, 10]
    elif 55 <= atomic_number <= 57 or 72 <= atomic_number <= 86: no_el = [10, 24, 20]
    else:
        print("ERROR: Function number_core_electrons cannot handle f-block elements!")
        exit()
    
    # remove core electrons that are not considered in case an ECP is used (relevant beyond krypton (=36))
    # ATTENTION: This has been implemented for the def2-ECP and is only guaranteed to work with the def2-XVP basis sets
    # core electrons for def2-ECP:
    # atnum 37-54: 1s, 2s, 3s, 2p, 3p, 3d (28 electrons)
    # atnum 55-57: 1s, 2s, 3s, 4s, 2p, 3p, 4p, 3d, 4d (46 electrons)
    # atnum 72-86: 1s, 2s, 3s, 4s, 2p, 3p, 4p, 3d, 4d, 4f (60 electrons)
    # this yields no_el = [2, 6, 0] for all atomic numbers < 36 defined above
    if _is_ecp and atomic_number > 36: no_el = [2, 6, 0]
    
    return no_el

########## END GLOBAL DECLARATIONS ##########


########## RE-DEFINED FUNCTIONS THAT ALSO EXIST IN getdata_corr.py ##########

def get_no_bond_atoms(mol, neighbors, only_for=[]):
    """In the eyes of a certain atom, get the number of bonded X atoms (X = H, C, N, O, S, Cl, Se, Br, Te, I).

    Retruns a dict with all atoms [no_H, no_C, no_N, no_O, no_S, no_Cl, no_Se, no_Br, no_Te, no_I].
    If only_for is not empty, then get no_atoms only for chosen nuclei (e.g. [1, 6] for only H and C).
    """

    no_atoms = {}
    elements = [1, 6, 7, 8, 16, 17, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 53, 80, 81, 82, 83]   # H, C, N, O, S, Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi

    for key, value in neighbors.items():
        # if only_for is not empty, make sure the atomic number of key fits entries in only_for
        if len(only_for) == 0 or mol.get_atomic_numbers()[key-1] in only_for:
            n = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # number of H, C, N, O, S, Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi
            for neigh in value:
                anx = mol.get_atomic_numbers()[neigh-1]
                for i, elem in enumerate(elements):
                    if anx == elem:
                        n[i] += 1
            no_atoms[key] = n
    
    return no_atoms


def get_no_sec_atoms(mol, neighbors, only_for=[]):
    """In the eyes of a certain atom, get the number of secondarily bonded X atoms (X = H, C, N, O, S, Cl, Se, Br, Te, I).

    Retruns a dict with all atoms [no_H, no_C, no_N, no_O, no_S, no_Cl, no_Se, no_Br, no_Te, no_I].
    If only_for is not empty, then get no_atoms only for chosen nuclei (e.g. [1, 6] for only H and C).
    """

    no_atoms = {}
    elements = [1, 6, 7, 8, 16, 17, 30, 31, 32, 33, 34, 35, 48, 49, 50, 51, 52, 53, 80, 81, 82, 83]   # H, C, N, O, S, Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi
    cn = get_cn(neighbors)            # in this context, coordination number means number of directly bonded atoms
    prim_bonded = get_no_bond_atoms(mol, neighbors)

    for key, value in neighbors.items():
        # if only_for is not empty, make sure the atomic number of key fits entries in only_for
        if not only_for or mol.get_atomic_numbers()[key-1] in only_for:
            n = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # number of H, C, N, O, S, Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi

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


def get_orbital_charges(outputpath, mode):
    """Get orbital charges form ORCA output.

    For H: s-/p-orbital charges; for C and heavy atoms also d-orbital charges.
    Also get the standard deviation of the 3 p orbital charges (px/py/pz) as mean of uniform charge distribution.
    Everything is stored in dicts {index: charge}.
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

            # d-orbital charge (only for carbon and heavy atoms)
            if tmp[1][:2] in ['C'] + list(heavy_atom_symbols):
                tmp_next4 = data[start+index+4].split()
                d_charges[atnum] = float(tmp_next4[5])

    for at in p_charges.keys():
        p_stdev[at] = statistics.stdev([px_charges[at], py_charges[at], pz_charges[at]])

    return s_charges, p_charges, d_charges, p_stdev


def get_nmr_quantities(outputpath, _is_ecp=False):
    """In addition to the total isotropic shielding constant (read by read_orca), get additional NMR quantities.

    The following quantities are returned:
    diamagnetic shielding constant, paramagnetic shielding constant, span, skew, asymmetry, anisotropy
    """

    with open(outputpath, 'r') as inp:
        data = inp.readlines()

    # get number of calculated NMR nuclei
    n_nmr = get_number_nmr(data)

    # get start of the chemical shift block
    # offset must be 17 for SR-ZORA output and 6 for non-rel output
    offset = 6 if _is_ecp else 17
    start = data.index('CHEMICAL SHIFTS\n') + offset

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


def shuffle_data(datalist, compounds, structures, seed, mode='structures'):
    """Shuffles a datalist in optional groups of atoms (=no groups), structures or compounds.

    datalist: list of DataPointH and DataPointC objects
    """

    if mode == 'atoms':
        new_datalist = copy.deepcopy(datalist)
        random.Random(seed).shuffle(new_datalist)

    if mode == 'structures':
        metalist = []
        for comp in compounds:
            for struct in structures[comp]:
                tmp = []
                for data in datalist:
                    if int(data.name[:dc]) == comp and int(data.name[-ds:]) == struct: tmp.append(data)
                metalist.append(tmp)
        random.Random(seed).shuffle(metalist)
        # the following combines lists in a list to one list (e.g. [[1,2],[3,4]] -> [1,2,3,4])
        new_datalist = [i for j in metalist for i in j]

    if mode == 'compounds':
        metalist = []
        for comp in compounds:
            tmp = []
            for data in datalist:
                if int(data.name[:dc]) == comp: tmp.append(data)
            metalist.append(tmp)
        random.Random(seed).shuffle(metalist)
        # the following combines lists in a list to one list (e.g. [[1,2],[3,4]] -> [1,2,3,4])
        new_datalist = [i for j in metalist for i in j]

    return new_datalist

########## END RE-DEFINED FUNCTIONS ##########


########## NEW FUNCTIONS THAT DO NOT EXIST IN getdata_corr.py ##########

def read_adf(outpath):
    """Read the DFT output from ADF calculations (without PLAMS, just from the output file)."""

    with open(outpath, 'r') as inp:
        data = inp.readlines()

    start_nmr = data.index(' *                              |     N M R     |                              *\n')
    shieldings = []
    found_atom = found_total = False
    for line in data[start_nmr:]:
        if 'Atom input number in the ADF calculation:' in line and not found_atom:
            atom = line.split()[7].split('(')
            elem = atom[0].upper()
            nuc = int(atom[1][:-1])
            found_atom = True
        if 'TOTAL NMR SHIELDING' in line and found_atom and not found_total:
            found_total = True
        if 'isotropic' in line and found_atom and found_total:
            val = float(line.split()[2])
            shieldings.append({
                'nuc': nuc,
                'elem': elem,
                'val': val
            })
            found_atom = found_total = False

    # make sure the shieldings are ordered according to the atom numbering (store as tuple)
    return tuple(sorted(shieldings, key=lambda s: s['nuc']))


def get_rel_contribution(shieldings_nonrel, shieldings_rel):
    """Get the shieldings object containing just the relativistic contributions per atom (= target)."""

    if not [(s['nuc'], s['elem']) for s in shieldings_nonrel] == [(s['nuc'], s['elem']) for s in shieldings_rel]:
        print("ERROR: ADF non-relativistic and relativistic calculation do not yield values for the same nuclei! Check the adf.out files or the code.")
        exit()

    rel_contributions = []
    for at in range(len(shieldings_nonrel)):
        rel_contributions.append({
            'nuc': shieldings_nonrel[at]['nuc'],
            'elem': shieldings_nonrel[at]['elem'],
            'val': shieldings_rel[at]['val'] - shieldings_nonrel[at]['val']
        })
    
    return tuple(rel_contributions)


def getref_adf(path):
    """Get the reference shieldings from the ADF output.

    ATTENTION: all Hs and all Cs are averaged, works e.g. for TMS and CH4.
    """

    ref_shieldings = read_adf(path)

    vals_h = []
    vals_c = []
    for atom in ref_shieldings:
        if atom['elem'].lower() == 'h':
            vals_h.append(atom['val'])
        if atom['elem'].lower() == 'c':
            vals_c.append(atom['val'])

    return {'H': sum(vals_h)/len(vals_h), 'C': sum(vals_c)/len(vals_c)}


def get_sum_prim(mol, neighbors: dict, mode: str) -> dict:
    """For each atom, get the sum of a certain value of all atoms in the primary bonding sphere (= 1 covalent bond away).
    
    Args:
        mol: ase.atoms.Atoms object containing the molecular information (from function read_mol).
        neighbors: dict containing covalent neighbor lists for each atom (from function read_mol).
        mode: str object identifier, must be one out of: 'atomic_numbers', 'masses'.
    
    Returns:
        dict containing the desired sum for each atom.
    """
    
    if mode == 'atomic_numbers':
        values = mol.get_atomic_numbers()
    elif mode == 'masses':
        values = mol.get_masses()
    else:
        print("ERROR: mode {} in get_sum_prim not known!".format(mode))
        exit()

    sums_prim = {}
    for at, neighs in neighbors.items():
        sums_prim[at] = sum([values[neigh-1] for neigh in neighs])

    return sums_prim


def get_sum_sec(mol, neighbors: dict, mode: str) -> dict:
    """For each atom, get the sum of a certain value of all atoms in the secondary bonding sphere (= 2 covalent bonds away).
    
    Args:
        mol: ase.atoms.Atoms object containing the molecular information (from function read_mol).
        neighbors: dict containing covalent neighbor lists for each atom (from function read_mol).
        mode: str object identifier, must be one out of: 'atomic_numbers', 'masses'.
    
    Returns:
        dict containing the desired sum for each atom (via function get_sum_prim).
    """

    # get secondary neighborlist, then call get_sum_prim with that list
    neighbors_sec = {}
    for at, neighs in neighbors.items():
        forbidden = np.append(neighs, at)
        neighbors_sec[at] = []
        for neigh in neighs:
            for sec_neigh in neighbors[neigh]:
                if sec_neigh not in neighbors_sec[at] and sec_neigh not in forbidden:
                    neighbors_sec[at].append(sec_neigh)
    
    return get_sum_prim(mol, neighbors_sec, mode)


def get_next_sphere(atoms_in: dict, exclude: dict, neighbors: dict) -> dict:
    """For a list of atoms, get all bonded atoms without duplicates.
    
    Args:
        atoms in: dict containing one list for each atom that contains all atoms to get the neighbors for.
        exclude: dict containing one list for each atom that contains atoms to be excluded.
        neighbors: dict containing covalent neighbor lists for each atom (from function read_mol).
    
    Returns:
        dict containing the desired atom lists.
    """
    
    atoms_out = {}
    for origin, atoms in atoms_in.items():
        next_atoms = []
        for at in atoms:
            for neigh in neighbors[at]:
                if neigh not in next_atoms and neigh not in exclude[origin]:
                    next_atoms.append(neigh)
        atoms_out[origin] = next_atoms

    return atoms_out


def get_neighbors_far(neighbors: dict, sphere_max: int) -> dict:
    """For each atom, get the bonded atoms in a desired bonding sphere (= number of bonds away).
    
    Args:
        neighbors: dict containing covalent neighbor lists for each atom (from function read_mol).
        sphere_max: The maximum number of bonds to be considered.
    
    Returns:
        dict containing atoms lists for each sphere with keys 'sphere_XX'.
    """

    sphere = 0
    far_neighbors = {at: {} for at in neighbors.keys()}
    atoms_neighbors = {at: [at] for at in neighbors.keys()}
    atoms_exclude = {at: [] for at in neighbors.keys()}
    while sphere < sphere_max:
        for origin, atoms in atoms_neighbors.items():
            atoms_exclude[origin].extend(atoms)
        atoms_neighbors = get_next_sphere(atoms_neighbors, atoms_exclude, neighbors)
        sphere += 1
        for origin, atoms in atoms_neighbors.items():
            far_neighbors[origin]['sphere_' + str(sphere)] = atoms
    
    return far_neighbors


def get_from_far(mol, far_neighbors: dict, max_depth: int, mode: str) -> dict:
    """For each atom, get a certain property for all selected bonding depths.
    
    Args:
        mol: ase.atoms.Atoms object containing the molecular information (from function read_mol).
        far_neighbors: dict containing neighbor lists of several spheres (from function get_neighbors_far).
        max_depth: The maximum number of bonds to be considered.
        mode: str identifier for the desired property, must be 'no_heavy' or 'av_masses'.
    
    Returns:
        dict containing the desired property information.
    """

    if mode == 'no_heavy':
        atomic_numbers = mol.get_atomic_numbers()
    elif mode == 'av_masses':
        atomic_masses = mol.get_masses()
    else:
        print("ERROR: Unknown mode {} in get_from_far!".format(mode))
        exit()

    prop_dict = {at: {} for at in far_neighbors.keys()}
    for origin, far_neighs in far_neighbors.items():
        for sphere in range(max_depth):
            if mode == 'no_heavy':
                tmplst = [1 for at in far_neighs['sphere_' + str(sphere+1)] if atomic_numbers[at-1] in heavy_atoms]
                this_prop = sum(tmplst)
            if mode == 'av_masses':
                tmplst = [atomic_masses[at-1] for at in far_neighs['sphere_' + str(sphere+1)]]
                if not tmplst:
                    this_prop = 0.0
                else:
                    this_prop = sum(tmplst)/len(tmplst)
            prop_dict[origin]['sphere_' + str(sphere+1)] = this_prop

    return prop_dict


def get_reference_rel(path_data, ref, func, basis, high=None):
    """Get the shielding values for a reference compound such as CH4 or TMS.
    
    Returns two dicts (base value and high-level target) with keys 'H' and 'C'.
    """

    has_highlevel = False
    if high is not None: has_highlevel = True

    # get the reference shielding values (e.g. from TMS molecule)
    refshieldings = getref_orca(os.path.join(path_data, ref, func, basis))
    print("using low-level DFT 1H reference shielding ({}): {} ppm".format(ref, refshieldings['H']))
    print("using low-level DFT 13C reference shielding ({}): {} ppm\n".format(ref, refshieldings['C']))

    ref_rel_contribution = {'H': None, 'C': None}
    if has_highlevel:

        ref_nonrel = getref_adf(os.path.join(path_data, ref, high['functional_nonrel'], high['basis'], 'adf.out'))
        ref_rel = getref_adf(os.path.join(path_data, ref, high['functional_rel'], high['basis'], 'adf.out'))
        ref_rel_contribution = {'H': ref_rel['H'] - ref_nonrel['H'], 'C': ref_rel['C'] - ref_nonrel['C']}

        print("using relativistic contribution to 1H reference shielding ({}): {} ppm".format(ref, ref_rel_contribution['H']))
        print("using relativistic contribution to 13C reference shielding ({}): {} ppm\n".format(ref, ref_rel_contribution['C']))

    return refshieldings, ref_rel_contribution


def get_reference_rel_json(path_data, ref, func, basis, high=None):
    """Get the shielding values for a reference compound provided in a json file.
    
    Returns two dicts (base value and high-level target) with keys 'H' and 'C'.
    """

    has_highlevel = False
    if high is not None: has_highlevel = True

    # get the reference shielding values (e.g. from TMS molecule)
    refshieldings = getref_orca(os.path.join(path_data, ref, func, basis))
    print("using low-level DFT 1H reference shielding ({}): {} ppm".format(ref, refshieldings['H']))
    print("using low-level DFT 13C reference shielding ({}): {} ppm\n".format(ref, refshieldings['C']))

    ref_rel_contribution = {'H': None, 'C': None}
    if has_highlevel:
        
        contrib_h = [data['rel_contrib'] for data in high['contributions']['tms'].values() if data['element'] == 'H']
        contrib_c = [data['rel_contrib'] for data in high['contributions']['tms'].values() if data['element'] == 'C']
        ref_rel_contribution = {'H': sum(contrib_h)/len(contrib_h), 'C': sum(contrib_c)/len(contrib_c)}

        print("using relativistic contribution to 1H reference shielding ({}): {} ppm".format(ref, ref_rel_contribution['H']))
        print("using relativistic contribution to 13C reference shielding ({}): {} ppm\n".format(ref, ref_rel_contribution['C']))

    return refshieldings, ref_rel_contribution


def get_data_rel(path_xyz, path_out, name, ref_shield, include, high=None, ref_relativistic_sigma=None, print_names=False, is_ecp= False):
    """Read all data from a molecule and process it to get the shieldings and shifts of all atoms.

    Save all information in two lists of dicts (data_h, data_c) and return further data to extend the ML input with.
    """

    has_highlevel = False
    if high is not None: has_highlevel = True

    data_h = []
    data_c = []

    # read the molecular data from the .xyz file and get the number of atoms and list of element symbols
    mol, neighbors = read_mol(path_xyz)
    nat = len(mol)
    atomic_numbers = mol.get_atomic_numbers()

    # get shieldings of the sample compound from ORCA output
    shieldings = read_orca(path_out)
    nmr_nuc = [s['nuc'] for s in shieldings]

    if has_highlevel:

        rel_contribution_full = tuple([{
                'nuc': int(atom),
                'elem': data['element'].upper(),
                'val': data['rel_contrib']
            } for atom, data in high['contributions'][name].items()])

        # reduce the list to only the relevant NMR nuclei
        rel_contribution = tuple([c for c in rel_contribution_full if c['nuc'] in nmr_nuc])
    
    # get some data needed as descriptors
    bond_atoms = get_no_bond_atoms(mol, neighbors)
    sec_atoms = get_no_sec_atoms(mol, neighbors)
    cn_d3 = get_cn_d3(mol)
    distances = mol.get_all_distances()
    
    # get the sum of the atomic numbers / masses of all atoms in the primary / secondary bonding sphere
    sum_atomic_numbers_prim = get_sum_prim(mol, neighbors, 'atomic_numbers')
    sum_masses_prim = get_sum_prim(mol, neighbors, 'masses')
    sum_atomic_numbers_sec = get_sum_sec(mol, neighbors, 'atomic_numbers')
    sum_masses_sec = get_sum_sec(mol, neighbors, 'masses')

    # get some descriptors from atoms further away
    far_neighbor_depth = 5
    far_neighbors = get_neighbors_far(neighbors, far_neighbor_depth)
    num_heavy_atoms = get_from_far(mol, far_neighbors, far_neighbor_depth, 'no_heavy')
    av_mass_atoms = get_from_far(mol, far_neighbors, far_neighbor_depth, 'av_masses')

    # get a bunch of ML descriptors from the orca output (mainly based on density matrix and NMR properties)
    at_chrg_mulliken = get_atomic_charges(mol, path_out, 'mulliken')
    at_chrg_loewdin = get_atomic_charges(mol, path_out, 'loewdin')
    orb_chrg_mulliken_s, orb_chrg_mulliken_p, orb_chrg_mulliken_d, orb_stdev_mulliken_p = get_orbital_charges(path_out, 'mulliken')
    orb_chrg_loewdin_s, orb_chrg_loewdin_p, orb_chrg_loewdin_d, orb_stdev_loewdin_p = get_orbital_charges(path_out, 'loewdin')
    mayer_VA, _, _ = get_Mayer_pop(mol, path_out)
    bond_orders_loewdin_sum, bond_orders_loewdin_av = get_bond_orders(neighbors, path_out, 'loewdin')
    bond_orders_mayer_sum, bond_orders_mayer_av = get_bond_orders(neighbors, path_out, 'mayer')
    nmr_quantities = get_nmr_quantities(path_out, is_ecp)

    # get the symmetric fingerprint descriptors
    acsf_mol = addso.acsf.create(mol)
    if 'soap' in include:
        soap_mol = addso.soap.create(mol)
        soap_mol = np.delete(soap_mol, addso.soap_zero_indexlist, 1)

    # loop over all atoms in the compound
    for iat in range(len(nmr_nuc)):

        # skip everything if the atom is not H or C
        this_atnum = atomic_numbers[nmr_nuc[iat]-1]
        if this_atnum not in [1, 6]: continue

        # this is the target value of the ML model
        relativistic_contribution = None

        if has_highlevel:
            # ensure 'nuc' and 'elem' is the same in rel_contribution and shieldings
            nucs = [rel_contribution[iat]['nuc'], shieldings[iat]['nuc']]
            elems = [rel_contribution[iat]['elem'], shieldings[iat]['elem']]

            if not all_equal(nucs):
                print("ERROR with compound {}: nuclei order in ADF and ORCA outputs is not the same!".format(name))
                exit()
            if not all_equal(elems):
                print("ERROR with compound {}: element order in ADF and ORCA outputs is not the same!".format(name))
                exit()

            if this_atnum == 1:
                relativistic_contribution = ref_relativistic_sigma['H'] - rel_contribution[iat]['val']
            elif this_atnum == 6:
                relativistic_contribution = ref_relativistic_sigma['C'] - rel_contribution[iat]['val']

        # if the actual atom is a H atom, get the distance to the neighboring atom
        if this_atnum == 1:
            # check if the H atom has exactly 1 neighbor; exit if this is not the case
            if len(neighbors[nmr_nuc[iat]]) != 1:
                print("More than 1 neighbor detected for H atom {} in {}. Please check the structure!".format(nmr_nuc[iat], path_xyz))
                exit()
            neigh_prim = neighbors[nmr_nuc[iat]][0]
            dist = distances[nmr_nuc[iat]-1][neigh_prim-1]   # distances list starts from 0

        # Get up to 4 (13C, 3 for 1H) descriptor sets for each neighbor (1H: second neighbor) that is a heavy atom
        heavy_props = []
        if this_atnum == 1:
            max_no_heavy = 3
            neigh_prim = neighbors[nmr_nuc[iat]][0]
            if atomic_numbers[neigh_prim-1] in heavy_atoms:
                relevant_neighbors = [neigh_prim]
            else:
                relevant_neighbors = [n for n in neighbors[neigh_prim] if n != nmr_nuc[iat]]
        elif this_atnum == 6:
            max_no_heavy = 4
            relevant_neighbors = list(neighbors[nmr_nuc[iat]])

        for neigh in relevant_neighbors:
            if atomic_numbers[neigh-1] in heavy_atoms:
                core_el = number_core_electrons(atomic_numbers[neigh-1], is_ecp)
                heavy_props.append({
                    'nuc': neigh,
                    'at_num': atomic_numbers[neigh-1],
                    'cn_d3': cn_d3[neigh],
                    'at_chrg_mulliken': at_chrg_mulliken[neigh],
                    's_occ_valance_mulliken': orb_chrg_mulliken_s[neigh] - core_el[0],
                    'p_occ_valance_mulliken': orb_chrg_mulliken_p[neigh] - core_el[1],
                    'd_occ_valance_mulliken': orb_chrg_mulliken_d[neigh] - core_el[2]
                })

        while len(heavy_props) < max_no_heavy:
            heavy_props.append({
                'nuc': None,
                'at_num': 0,
                'cn_d3': 0.0,
                'at_chrg_mulliken': 0.0,
                's_occ_valance_mulliken': 0.0,
                'p_occ_valance_mulliken': 0.0,
                'd_occ_valance_mulliken': 0.0
            })

        # escape in case there are more than 4 heavy neihbors detected or filling up to 4 entries did't work for whatever reason
        if len(heavy_props) != max_no_heavy:
            print("ERROR for atom {} in {}. Counted more than 4 heavy neighbor atoms or other technical problem.".format(nmr_nuc[iat], path_xyz))
            exit()
        
        # combine all enries in heavy_props to a unified list (len = 5 for H, len = 4*5 = 20 for C)
        heavy_props_list = []
        for p in heavy_props: heavy_props_list.extend([p['at_num'], p['cn_d3'], p['at_chrg_mulliken'], p['s_occ_valance_mulliken'], p['p_occ_valance_mulliken'], p['d_occ_valance_mulliken']])

        # add all ACSF and SOAP descriptors
        # acsf_mol and soap_mol need the atom index, but start with 0 so nmr_nuc-1
        acsf_data = {acsf_name: acsf_mol[nmr_nuc[iat]-1][i] for i, acsf_name in enumerate(addso.acsf_labels)}
        if 'soap' in include:
            soap_data = {soap_name: soap_mol[nmr_nuc[iat]-1][i] for i, soap_name in enumerate(addso.soap_labels_reduced)}

        # store descriptors for 1H
        if this_atnum == 1:

            datapoint = soDataPointH(name, nmr_nuc[iat], print_names)
            datapoint.set_attr_general(
                relativistic_contribution,
                shieldings[iat]['val'],
                sum_atomic_numbers_prim[nmr_nuc[iat]],
                sum_masses_prim[nmr_nuc[iat]],
                sum_atomic_numbers_sec[nmr_nuc[iat]],
                sum_masses_sec[nmr_nuc[iat]],
                [num_heavy_atoms[nmr_nuc[iat]]['sphere_' + str(s+1)] for s in range(5)],
                [av_mass_atoms[nmr_nuc[iat]]['sphere_' + str(s+1)] for s in range(5)],
                at_chrg_mulliken[nmr_nuc[iat]],
                at_chrg_loewdin[nmr_nuc[iat]],
                orb_chrg_mulliken_s[nmr_nuc[iat]],
                orb_chrg_loewdin_s[nmr_nuc[iat]],
                orb_chrg_mulliken_p[nmr_nuc[iat]],
                orb_chrg_loewdin_p[nmr_nuc[iat]],
                mayer_VA[nmr_nuc[iat]],
                heavy_props_list,
                nmr_quantities[iat]['shielding_dia'],
                nmr_quantities[iat]['shielding_para'],
                nmr_quantities[iat]['span'],
                nmr_quantities[iat]['skew'],
                nmr_quantities[iat]['asymmetry'],
                nmr_quantities[iat]['anisotropy']
            )
            datapoint.set_acsf(acsf_data)
            if 'soap' in include: datapoint.set_soap(soap_data)
            datapoint.set_attr_special_rel(
                ref_shield['H'],
                cn_d3[neigh_prim],                                 # the D3 CN of the neighboring C atom
                [sec_atoms[nmr_nuc[iat]][i] for i in range(22)],   # number of secondarily bonded [H, C, N, O, S, Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi] (all 22 elements of sec_atoms)
                dist,
                bond_orders_loewdin_sum[nmr_nuc[iat]],             # starts counting with 1
                bond_orders_mayer_sum[nmr_nuc[iat]]                # starts counting with 1
            )

            data_h.append(datapoint)

        # store descriptors for 13C
        if this_atnum == 6:

            datapoint = soDataPointC(name, nmr_nuc[iat], print_names)
            datapoint.set_attr_general(
                relativistic_contribution,
                shieldings[iat]['val'],
                sum_atomic_numbers_prim[nmr_nuc[iat]],
                sum_masses_prim[nmr_nuc[iat]],
                sum_atomic_numbers_sec[nmr_nuc[iat]],
                sum_masses_sec[nmr_nuc[iat]],
                [num_heavy_atoms[nmr_nuc[iat]]['sphere_' + str(s+1)] for s in range(4)],
                [av_mass_atoms[nmr_nuc[iat]]['sphere_' + str(s+1)] for s in range(4)],
                at_chrg_mulliken[nmr_nuc[iat]],
                at_chrg_loewdin[nmr_nuc[iat]],
                orb_chrg_mulliken_s[nmr_nuc[iat]],
                orb_chrg_loewdin_s[nmr_nuc[iat]],
                orb_chrg_mulliken_p[nmr_nuc[iat]],
                orb_chrg_loewdin_p[nmr_nuc[iat]],
                mayer_VA[nmr_nuc[iat]],
                heavy_props_list,
                nmr_quantities[iat]['shielding_dia'],
                nmr_quantities[iat]['shielding_para'],
                nmr_quantities[iat]['span'],
                nmr_quantities[iat]['skew'],
                nmr_quantities[iat]['asymmetry'],
                nmr_quantities[iat]['anisotropy']
            )
            datapoint.set_acsf(acsf_data)
            if 'soap' in include: datapoint.set_soap(soap_data)
            datapoint.set_attr_special_rel(
                ref_shield['C'],
                cn_d3[nmr_nuc[iat]],                                # the D3 CN of the C atom
                [bond_atoms[nmr_nuc[iat]][i] for i in range(22)],   # number of bonded [H, C, N, O, S, Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi] (all 22 elements of bond_atoms)
                [sec_atoms[nmr_nuc[iat]][i] for i in range(22)],    # number of secondarily bonded [H, C, N, O, S, Cl, Zn, Ga, Ge, As, Se, Br, Cd, In, Sn, Sb, Te, I, Hg, Tl, Pb, Bi] (all 22 elements of sec_atoms)
                orb_chrg_mulliken_d[nmr_nuc[iat]],                  # starts counting with 1
                orb_chrg_loewdin_d[nmr_nuc[iat]],                   # starts counting with 1
                orb_stdev_mulliken_p[nmr_nuc[iat]],                 # starts counting with 1
                orb_stdev_loewdin_p[nmr_nuc[iat]],                  # starts counting with 1
                bond_orders_loewdin_sum[nmr_nuc[iat]],              # starts counting with 1
                bond_orders_mayer_sum[nmr_nuc[iat]],                # starts counting with 1
                bond_orders_loewdin_av[nmr_nuc[iat]],               # starts counting with 1
                bond_orders_mayer_av[nmr_nuc[iat]]                  # starts counting with 1
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


def write_ml_input_rel(datalist, outpath, outname, extension="", is_sample=False):
    """Write full 1H/13C data from datalist into a file used as input for 1H/13C ML correction.
    
    Additionally add an extension at the end of the file.
    (this is useful for a sample compound, where the dataset is a list of only one compound, to provide information for the ML script).
    Essentially, this does the same as write_ml_input, but with target = relativistic contribution.
    """

    atnums = "# atom_numbers:"
    if not datalist: return
    
    printout = datalist[0].get_header_rel()
    for data in datalist:
        printout.append(data.get_printout_rel(dc=dc))
        if is_sample: atnums += " " + str(data.atom)
    if is_sample: printout.append(atnums)

    with open(os.path.join(outpath, outname), 'w') as out:
        out.write("\n".join(printout) + "\n")
        out.write(extension)

########## END NEW FUNCTIONS ##########


if __name__ == "__main__":

    workdir = os.getcwd()

    # initialize some proper parser for the command line input
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--set', type=str, help='choose to extract data from the test set compounds; provide the path to the data set directory')
    parser.add_argument('-a', '--sample', nargs=3, help='choose to extract data from a sample compound; provide structure in .xyz format, an orca output file, and a data directory (path to the data of the desired reference compound)', metavar=('xyz', 'out', 'data'))
    parser.add_argument('-ar', '--sample_ref', nargs=3, help='choose to extract data from a sample compound with high-level reference shifts; provide structure in .xyz format, an orca output file, and a data directory (path to the data of the desired reference compound and the high-level reference data including a rel_contributions.json file)', metavar=('xyz', 'out', 'data'))
    parser.add_argument('-fl', '--functional_low', default='pbe0', help='functional for low-level DFT NMR shift to be corrected, default: pbe0')
    parser.add_argument('-bl', '--basis_low', default='ZORA-def2-TZVP', help='basis set for low-level DFT NMR shift to be corrected, default: ZORA-def2-TZVP')
    parser.add_argument('-r', '--reference', default='tms', help='reference compound for NMR shift, default: tms')
    parser.add_argument('-s', '--shuffle', default='structures', choices=['none', 'atoms', 'structures', 'compounds'], help='shuffle mode for the data set: atoms, structures or compounds, default: structures')
    parser.add_argument('-rs', '--randomseed', type=int, default=0, help='random seed for data shuffling, default: 0')
    parser.add_argument('-pn', '--print_names', action='store_true', help='additionally print names (i.e. compound, structures, and atom numbers) in the data set file (only relevant for --set option)')
    parser.add_argument('-i', '--include', nargs='+', type=str.lower, choices=['soap'], help='optionally include SOAP features')

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
    # The settings for the relativistic reference calculations are currently fixed
    functional_nonrel = 'pbe0_scalar'
    functional_rel = 'pbe0_so-zora'
    basis_rel = 'TZ2P'

    functional_low = args.functional_low
    basis_low = args.basis_low
    is_ecp = True if basis_low in basis_sets_ecp else False
    shuffle_mode = args.shuffle
    random_seed = args.randomseed
    print_names = args.print_names

    include = ['acsf']
    if args.include is not None: include.extend(args.include)


    ### Read in (reference) data from the whole data base ###
    if input_mode == 'set':

        # set up lists of compound names and the corresponding structures
        compounds = [c + 1 for c in range(1597)]
        structures = {c: [0, 1, 2, 3] for c in compounds}

        print("Extracting ML descriptors from the data set\n")
        print("Settings:")
        print("Reference relativistic contribution via : ({} - {}) / {}".format(functional_rel, functional_nonrel, basis_rel))
        print("Functional for low-level DFT calculation: {}".format(functional_low))
        print("Basis set for low-level DFT calculation : {}".format(basis_low))
        print("Reference compound                      : {}".format(reference))
        print("Shuffle mode                            : {}".format(shuffle_mode))
        print("Random seed for shuffling               : {}\n".format(random_seed))

        reference_json_file = os.path.join(workdir, 'rel_contributions.json')
        with open(reference_json_file, 'r') as inp:
            rel_contributions = json.load(inp)

        high_level = {
            'functional_nonrel': functional_nonrel,
            'functional_rel': functional_rel,
            'basis': basis_rel,
            'contributions': rel_contributions
        }

        ref, ref_rel_contributions = get_reference_rel_json(datapath, reference, functional_low, basis_low, high_level)

        data_h = []
        data_c = []
        for comp in compounds:
            print("collecting data from compound {} ({} structures)".format(comp, len(structures[comp])))
            for struct in structures[comp]:
                name = str(comp).zfill(dc) + '_' + str(struct).zfill(ds)
                structpath = os.path.join(datapath, str(comp).zfill(dc), str(struct).zfill(ds))
                path_xyz = os.path.join(structpath, functional_low, basis_low, name + '.xyz')
                path_out = os.path.join(structpath, functional_low, basis_low, 'orca.out')

                dath, datc, _ = get_data_rel(path_xyz, path_out, name, ref, include, high_level, ref_rel_contributions, print_names, is_ecp)

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
            data_h_shuffled = shuffle_data(data_h, compounds, structures, random_seed, mode=shuffle_mode)
            data_c_shuffled = shuffle_data(data_c, compounds, structures, random_seed, mode=shuffle_mode)

        # add some method data to the end of the data set file
        extension_settings = [
            "# Relativistic contribution via     : ({} - {}) / {}".format(functional_rel, functional_nonrel, basis_rel),
            "# low-level functional (DFT)        : {}".format(functional_low),
            "# low-level basis set (DFT)         : {}".format(basis_low),
            "# NMR reference compound            : {}".format(reference),
            "# low-level 1H reference shielding  : {} ppm".format(ref['H']),
            "# low-level 13C reference shielding : {} ppm".format(ref['C']),
            "# reference SO contribution 1H      : {} ppm".format(ref_rel_contributions['H']),
            "# reference SO contribution 13C     : {} ppm".format(ref_rel_contributions['C']),
            "# shuffle mode                      : {}".format(shuffle_mode),
            "# random seed for shuffling         : {}\n".format(random_seed)
        ]

        write_ml_input_rel(data_h_shuffled, workdir, "ml-so_" + functional_low + "_" + basis_low + "_h.dat", extension="\n".join(extension_settings))
        write_ml_input_rel(data_c_shuffled, workdir, "ml-so_" + functional_low + "_" + basis_low + "_c.dat", extension="\n".join(extension_settings))


    ### Read in data (optionally with reference) from one sample calculation ###
    elif input_mode in ['sample', 'sample_ref']:

        # define a compound name (use .xyz file and delete '.xyz' if present)
        sample_name = os.path.abspath(samplepath_xyz).split(os.sep)[-1].replace('.xyz', '')

        print("Extracting ML descriptors from sample compound: {}".format(sample_name))
        print("... using 3D structure provided in:   {}".format(os.path.abspath(samplepath_xyz)))
        print("... using ORCA calculation output in: {}".format(os.path.abspath(samplepath_out)))
        print("... using supplementary data in:      {}\n".format(os.path.abspath(datapath)))
        print("Settings:")
        if input_mode == 'sample_ref':
            print("Reference relativistic contribution via : ({} - {}) / {}".format(functional_rel, functional_nonrel, basis_rel))
        print("Functional for low-level DFT calculation: {}".format(functional_low))
        print("Basis set for low-level DFT calculation : {}".format(basis_low))
        print("Reference compound                      : {}\n".format(reference))

        # get all the data
        if input_mode == 'sample_ref':
            reference_json_file = os.path.join(datapath, 'rel_contributions.json')
            with open(reference_json_file, 'r') as inp:
                rel_contributions = json.load(inp)

            high_level = {
                'functional_nonrel': functional_nonrel,
                'functional_rel': functional_rel,
                'basis': basis_rel,
                'contributions': rel_contributions
            }

            ref, ref_rel_contributions = get_reference_rel_json(datapath, reference, functional_low, basis_low, high_level)
            data_h, data_c, extension = get_data_rel(samplepath_xyz, samplepath_out, sample_name, ref, include, high_level, ref_rel_contributions, print_names, is_ecp)

        elif input_mode == 'sample':
            ref, ref_rel_contributions = get_reference_rel_json(datapath, reference, functional_low, basis_low)
            data_h, data_c, extension = get_data_rel(samplepath_xyz, samplepath_out, sample_name, ref, include)

        print("\nAnalyzed a total number of:")
        print("{} 1H NMR shifts".format(len(data_h)))
        print("{} 13C NMR shifts".format(len(data_c)))

        # add some method data to the beginning of the extension
        extension_settings = [
            "# low-level functional (DFT)       : {}".format(functional_low),
            "# low-level basis set (DFT)        : {}".format(basis_low),
            "# NMR reference compound           : {}".format(reference),
            "# low-level 1H reference shielding : {} ppm".format(ref['H']),
            "# low-level 13C reference shielding: {} ppm".format(ref['C']),
        ]
        if input_mode == 'sample_ref':
        # add some method data to the end of the data set file
            extension_refsettings = [
                "# Relativistic contribution via    : ({} - {}) / {}".format(functional_rel, functional_nonrel, basis_rel),
                "# reference SO contribution 1H     : {} ppm".format(ref_rel_contributions['H']),
                "# reference SO contribution 13C    : {} ppm".format(ref_rel_contributions['C'])
            ]
            extension_settings = extension_refsettings + extension_settings
        extension = "\n".join(extension_settings) + "\n" + extension + "\n"

        if input_mode == 'sample_ref':
            write_ml_input_rel(data_h, workdir, "ml-so_" + sample_name + "_ref_h.dat", extension=extension, is_sample=True)
            write_ml_input_rel(data_c, workdir, "ml-so_" + sample_name + "_ref_c.dat", extension=extension, is_sample=True)
        else:
            write_ml_input_rel(data_h, workdir, "ml-so_" + sample_name + "_h.dat", extension=extension, is_sample=True)
            write_ml_input_rel(data_c, workdir, "ml-so_" + sample_name + "_c.dat", extension=extension, is_sample=True)

