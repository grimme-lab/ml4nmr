#!/usr/bin/env python3

# This scripts manages additional descriptors
# (such as ACSF and SOAP) within ml4nmr
# Copyright (C) 2023 Julius Kleine Büning

from dscribe.descriptors import ACSF, SOAP

# the following is used for ACSF and SOAP descriptors
elements_sym = ('H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl')
cutoff_sym = 5.0   # used for ACSF and SOAP, but may be varied

# declare global ACSF constructor
acsf = ACSF(
    rcut=cutoff_sym,
    species=elements_sym,
    sparse=False
)

acsf_labels = []
for e in elements_sym:
    acsf_labels.append('ACSF_{}'.format(e))
acsf_labels = tuple(acsf_labels)

# declare global SOAP constructor
soap = SOAP(
    r_cut=cutoff_sym,
    n_max=1,
    l_max=0,
    sigma=1.0,
    rbf='gto',
    average='off',
    species=elements_sym,
    sparse=False
)

soap_labels = []
for e1 in elements_sym:
    for e2 in elements_sym[elements_sym.index(e1):]:
        soap_labels.append('SOAP_{}-{}'.format(e1, e2))
soap_labels = tuple(soap_labels)

# indices of all soap values that are 0.0 with the above soap settings because not all elements are contained in all molecules (Attention: indices start at 0)
soap_zero_indexlist = (15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 52, 53, 54, 55, 56, 57, 59, 62, 63, 64, 70, 71, 72, 73, 74, 76, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 91, 92, 93, 96, 97, 100, 101, 103)

soap_labels_reduced = list(soap_labels)
for red_index in sorted(soap_zero_indexlist, reverse=True): soap_labels_reduced.pop(red_index)
soap_labels_reduced = tuple(soap_labels_reduced)
