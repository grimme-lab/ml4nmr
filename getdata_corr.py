#!/usr/bin/env python3

# =============================================================================
#
# This scripts read, processes, and prints data for the Delta_corr_ML model
# https://doi.org/10.1021/acs.jctc.3c00165
#
# Run:
# getdata_corr.py [options]
#
# Copyright (C) 2021-2024 Julius Kleine Büning
#
# =============================================================================


import os
import argparse
import copy
import numpy as np
import statistics
import random

from functions_getdata import *
from classes_corr import corrDataPointH, corrDataPointC
import additional_corr as addcorr


########## GLOBAL DECLARATIONS ##########

# number of digits of compounds (dc, e.g. 042) and structures (ds, e.g. 04)
dc = 3
ds = 2

########## END GLOBAL DECLARATIONS ##########


########## OWN FUNCTIONS ##########

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


def getref_cfour(path, ref, basis):
    """Get the reference shieldings from the coupled cluster output (CFOUR).

    ATTENTION: all Hs and all Cs are averaged, works e.g. for TMS and CH4.
    """
    
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


def get_no_bond_atoms(mol, neighbors, only_for=[]):
    """In the eyes of a certain atom, get the number of bonded X atoms (X = H, C, N, O, S, Cl).
    
    Retruns a dict with all atoms [no_H, no_C, no_N, no_O, no_S, no_Cl].
    If only_for is not empty, then get no_atoms only for chosen nuclei (e.g. [1, 6] for only H and C).
    """

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


def get_no_sec_atoms(mol, neighbors, only_for=[]):
    """In the eyes of a certain atom, get the number of secondarily bonded X atoms (X = H, C, N, O, S, Cl).
    
    Retruns a dict with all atoms [no_H, no_C, no_N, no_O, no_S, no_Cl].
    If only_for is not empty, then get no_atoms only for chosen nuclei (e.g. [1, 6] for only H and C).
    """

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


def get_orbital_charges(outputpath, mode):
    """Get orbital charges form ORCA output.

    For H: s-/p-orbital charges; for C also d-orbital charges.
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

            # d-orbital charge (only for carbon)
            if tmp[1] == 'C':
                tmp_next4 = data[start+index+4].split()
                d_charges[atnum] = float(tmp_next4[5])

    for at in p_charges.keys():
        p_stdev[at] = statistics.stdev([px_charges[at], py_charges[at], pz_charges[at]])
        #p_stdev[at] = np.std([px_charges[at], py_charges[at], pz_charges[at]], ddof=1)

    return s_charges, p_charges, d_charges, p_stdev


def get_nmr_quantities(outputpath, zora=False):
    """In addition to the total isotropic shielding constant (read by read_orca), get additional NMR quantities.

    The following quantities are returned:
    diamagnetic shielding constant, paramagnetic shielding constant, span, skew, asymmetry, anisotropy
    """

    with open(outputpath, 'r') as inp:
        data = inp.readlines()

    # get number of calculated NMR nuclei
    n_nmr = get_number_nmr(data)

    # get start of the chemical shift block (shifted if SR-ZORA is applied)
    offset = 17 if zora else 6
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


def get_reference(path_data, ref, func, basis, high=None):
    """Get the shielding values for a reference compound such as CH4 or TMS.
    
    Returns two dicts (base value and high-level target) with keys 'H' and 'C'.
    """

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
            'H': addcorr.extrapolate(refshieldings_high_cc['H'], refshieldings_high_dft_cc['H'], refshieldings_high_dft_x['H'], refshieldings_high_dft_y['H']),
            'C': addcorr.extrapolate(refshieldings_high_cc['C'], refshieldings_high_dft_cc['C'], refshieldings_high_dft_x['C'], refshieldings_high_dft_y['C'])
        }

        print("using high-level CCSD(T)/TZ+ 1H reference shielding ({}): {} ppm".format(ref, refshieldings_high_cc_cbs['H']))
        print("using high-level CCSD(T)/TZ+ 13C reference shielding ({}): {} ppm\n".format(ref, refshieldings_high_cc_cbs['C']))

        refshieldings_high = {'H': refshieldings_high_cc_cbs['H'], 'C': refshieldings_high_cc_cbs['C']}

    return refshieldings, refshieldings_high


def get_data(path_xyz, path_out, name, ref_shield, include, zora, high=None, ref_shield_high={'H': None, 'C': None}, path_data=None, print_names=False):
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
    nmr_quantities = get_nmr_quantities(path_out, zora)

    # get the symmetric fingerprint descriptors
    if 'acsf' in include:
        acsf_mol = addcorr.acsf.create(mol)
    if 'soap' in include:
        soap_mol = addcorr.soap.create(mol)
        soap_mol = np.delete(soap_mol, addcorr.soap_zero_indexlist, 1)

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
            acsf_data = {acsf_name: acsf_mol[nmr_nuc[iat]-1][i] for i, acsf_name in enumerate(addcorr.acsf_labels)}
        if 'soap' in include:
            soap_data = {soap_name: soap_mol[nmr_nuc[iat]-1][i] for i, soap_name in enumerate(addcorr.soap_labels_reduced)}

        if shieldings[iat]['elem'] == 'H':

            datapoint = corrDataPointH(name, nmr_nuc[iat], print_names)
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

            datapoint = corrDataPointC(name, nmr_nuc[iat], print_names)
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


def shuffle_data(datalist, compounds, structures, seed, mode='structures'):
    """Shuffles a datalist in optional groups of atoms (=no groups), structures or compounds.
    
    datalist: list of corrDataPointH and corrDataPointC objects
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


def write_ml_input(datalist, outpath, outname, extension="", is_sample=False):
    """Write full 1H/13C data from datalist into a file used as input for 1H/13C ML correction.
    
    Additionally add an extension at the end of the file.
    (this is useful for a sample compound, where the dataset is a list of only one compound, to provide information for the ML script)
    
    ATTENTION: This function only gives reasonable output if datalist is purely H or purely C!
    """

    atnums = "# atom_numbers:"
    if not datalist: return
    
    printout = datalist[0].get_header()
    for data in datalist:
        printout.append(data.get_printout(dc=dc))
        if is_sample: atnums += " " + str(data.atom)
    if is_sample: printout.append(atnums)

    with open(os.path.join(outpath, outname), 'w') as out:
        out.write("\n".join(printout) + "\n")
        out.write(extension)

########## END OWN FUNCTIONS ##########


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
    # if low-level basis set starts with 'ZORA', assume that SR-ZORA approach in ORCA was used (this may be changed in the future)
    is_zora = basis_low[:4] == 'ZORA'
    shuffle_mode = args.shuffle
    random_seed = args.randomseed
    print_names = args.print_names

    if args.include is None: include = []
    else: include = include = args.include


    ### Read in (reference) data from the whole data base ###
    if input_mode == 'set':

        compounds = [c + 1 for c in range(100)]   # compound numbers from 1 to 100
        structures = {}
        for comp in compounds: structures[comp] = [int(f) for f in sorted(os.listdir(os.path.join(datapath, str(comp).zfill(3))))]

        print("Extracting ML descriptors from the data set\n")
        print("Settings:")
        print("Reference coupled cluster calculation level: CCSD(T)/{}".format(basis_3z))
        print("Functional for DFT CBS extrapolation       : {}".format(functional_high))
        print("Basis sets for DFT CBS extrapolation       : {}, {}, {}".format(basis_3z, basis_4z, basis_5z))
        print("Functional for low-level DFT calculation   : {}".format(functional_low))
        print("Basis set for low-level DFT calculation    : {}".format(basis_low))
        print("Reference compound                         : {}".format(reference))
        print("Shuffle mode                               : {}".format(shuffle_mode))
        print("Random seed for shuffling                  : {}\n".format(random_seed))

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

                dath, datc, _ = get_data(path_xyz, path_out, name, ref, include, is_zora, high_level, ref_high, structpath, print_names)
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
            structpath = os.path.join(datapath, sample_name)
            high_level = {
                'functional': functional_high,
                'basis_3z': basis_3z,
                'basis_4z': basis_4z,
                'basis_5z': basis_5z
            }
            ref, ref_high = get_reference(datapath, reference, functional_low, basis_low, high_level)
            data_h, data_c, extension = get_data(samplepath_xyz, samplepath_out, sample_name, ref, include, is_zora, high_level, ref_high, structpath)
        elif input_mode == 'sample':
            ref, ref_high = get_reference(datapath, reference, functional_low, basis_low)
            data_h, data_c, extension = get_data(samplepath_xyz, samplepath_out, sample_name, ref, include, is_zora)

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

