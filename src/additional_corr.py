# =============================================================================
#
# This scripts manages additional functions and descriptors (such as ACSF and
# SOAP) for the Delta_corr_ML model (for Delta_SO_ML, see additional_so.py)
#
# Copyright (C) 2021-2024 Julius Kleine Büning
#
# =============================================================================


import numpy as np
from scipy.optimize import curve_fit
from dscribe.descriptors import ACSF, SOAP


# the following is used for ACSF and SOAP descriptors
elements_sym = ('H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl')
cutoff_sym = 5.0   # used for ACSF and SOAP, but may be varied

# declare global ACSF constructor
acsf = ACSF(
    rcut=cutoff_sym,   # works with dscribe version 1.2
    species=elements_sym,
    sparse=False
)

# define the labels for ACSF
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

# define the labels for SOAP
soap_labels = []
for e1 in elements_sym:
    for e2 in elements_sym[elements_sym.index(e1):]:
        soap_labels.append('SOAP_{}-{}'.format(e1, e2))
soap_labels = tuple(soap_labels)

# indices of all soap values that are 0.0 for all systems in the Delta_corr_ML data set with the above soap settings,
# because not all elements are contained in all molecules (Attention: indices start at 0)
soap_zero_indexlist = (15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 36, 37, 38, 52, 53, 54, 55, 56, 57, 59, 62, 63, 64, 70, 71, 72, 73, 74, 76, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 91, 92, 93, 96, 97, 100, 101, 103)

# with this information, define a reduced list of SOAP labels
soap_labels_reduced = list(soap_labels)
for red_index in sorted(soap_zero_indexlist, reverse=True): soap_labels_reduced.pop(red_index)
soap_labels_reduced = tuple(soap_labels_reduced)


### Function declarations that are also needed for class definitions (classes_corr) ###

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

