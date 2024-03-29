# =============================================================================
#
# This scripts provides classes for the Delta_corr_ML method
#
# Copyright (C) 2021-2024 Julius Kleine Büning
#
# =============================================================================


import additional_corr as addcorr


class corrDataPoint:
    """Class that manages data acquisition and processing for the Delta_corr_ML model."""

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
        """Set general settings valid for every derived class."""

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
            self.shielding_cc_ext = addcorr.extrapolate(self.shielding_cc_3z, self.shielding_dft_3z, self.shielding_dft_4z, self.shielding_dft_5z)

    def set_acsf(self, acsf_data):
        """Store ACSF data if used."""

        self.include_acsf = True
        self.acsf_data = acsf_data

    def set_soap(self, soap_data):
        """Store SOAP data if used."""

        self.include_soap = True
        self.soap_data = soap_data


class corrDataPointH(corrDataPoint):
    """Class managing data for a 1H nucleus within the Delta_corr_ML model."""

    def set_attr_special(
        self, refshielding_low, shift_neighbor,
        cn_d3, number_hch, number_hyx,
        distance_hc, bond_order_loewdin, bond_order_mayer,
        refshielding_cc_ext=None
    ):
        """Set additional settings valid only for a 1H datapoint."""

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
        """Create the header for a 1H ML data file."""

        nvar = 25
        if self.print_names: nvar += 3
        if self.include_acsf: nvar += len(addcorr.acsf_labels)
        if self.include_soap: nvar += len(addcorr.soap_labels_reduced)

        header_1 = "# " + " ".join([str(i) for i in range(nvar)])
        header_2 = "# shift_high-low shift_low CN(X) no_HCH no_HYH no_HYC no_HYN no_HYO dist_HC shift_low_neighbor_C shielding_dia shielding_para span skew asymmetry anisotropy at_charge_mull at_charge_loew orb_charge_mull_s orb_charge_mull_p orb_charge_loew_s orb_charge_loew_p BO_loew BO_mayer mayer_VA"
        if self.print_names: header_2 = "# compound structure atom" + header_2[1:]
        if self.include_acsf: header_2 += " " + " ".join(addcorr.acsf_labels)
        if self.include_soap: header_2 += " " + " ".join(addcorr.soap_labels_reduced)

        return [header_1, header_2]


    def get_printout(self, dc=3):
        """Get the whole printout for a 1H ML data file."""

        if self.print_names:
            beginning = [self.name[:dc], self.name[dc+1:], self.atom]
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
                self.bond_order_loewdin, self.bond_order_mayoutputer, self.mayer_valence_total
            ]
        )

        if self.include_acsf: line += [self.acsf_data[acsf_name] for acsf_name in addcorr.acsf_labels]
        if self.include_soap: line += [self.soap_data[soap_name] for soap_name in addcorr.soap_labels_reduced]

        return " ".join([str(val) for val in line])


class corrDataPointC(corrDataPoint):
    """Class managing data for a 13C nucleus within the Delta_corr_ML model."""

    def set_attr_special(
        self, refshielding_low, cn_d3, number_cx, number_cyx,
        orbital_charge_mulliken_d, orbital_charge_loewdin_d, orbital_stdev_mulliken_p, orbital_stdev_loewdin_p,
        bond_orders_loewdin_sum, bond_orders_mayer_sum, bond_orders_loewdin_av, bond_orders_mayer_av,
        refshielding_cc_ext=None
    ):
        """Set additional settings valid only for a 13C datapoint."""

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
        """Create the header for a 13C ML data file."""

        nvar = 32
        if self.print_names: nvar += 3
        if self.include_acsf: nvar += len(addcorr.acsf_labels)
        if self.include_soap: nvar += len(addcorr.soap_labels_reduced)

        header_1 = "# " + " ".join([str(i) for i in range(nvar)])
        header_2 = "# shift_high-low shift_low CN(X) no_CH no_CC no_CN no_CO no_CYH no_CYC no_CYN no_CYO shielding_dia shielding_para span skew asymmetry anisotropy at_charge_mull at_charge_loew orb_charge_mull_s orb_charge_mull_p orb_charge_mull_d orb_stdev_mull_p orb_charge_loew_s orb_charge_loew_p orb_charge_loew_d orb_stdev_loew_p BO_loew_sum BO_loew_av BO_mayer_sum BO_mayer_av mayer_VA"
        if self.print_names: header_2 = "# compound structure atom" + header_2[1:]
        if self.include_acsf: header_2 += " " + " ".join(addcorr.acsf_labels)
        if self.include_soap: header_2 += " " + " ".join(addcorr.soap_labels_reduced)

        return [header_1, header_2]


    def get_printout(self, dc=3):
        """Get the whole printout for a 13C ML data file."""

        if self.print_names:
            beginning = [self.name[:dc], self.name[dc:], self.atom]
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

        if self.include_acsf: line += [self.acsf_data[acsf_name] for acsf_name in addcorr.acsf_labels]
        if self.include_soap: line += [self.soap_data[soap_name] for soap_name in addcorr.soap_labels_reduced]

        return " ".join([str(val) for val in line])

