# =============================================================================
#
# This scripts provides classes for the Delta_SO_ML method
#
# Copyright (C) 2023-2024 Julius Kleine Büning
#
# =============================================================================


import additional_so as addso


class soDataPoint:
    """Class that manages data acquisition and processing for the Delta_SO_ML model."""

    def __init__(self, name, atom, print_names=False):
        self.name = name
        self.atom = atom
        self.print_names = print_names
        self.include_acsf = False
        self.include_soap = False

    def set_attr_general(
        self, rel_contribution, shielding_low,
        sum_atomic_numbers_prim, sum_masses_prim,
        sum_atomic_numbers_sec, sum_masses_sec,
        num_heavy_atoms, av_mass_atoms,
        atomic_charge_mulliken, atomic_charge_loewdin,
        orbital_charge_mulliken_s, orbital_charge_loewdin_s,
        orbital_charge_mulliken_p, orbital_charge_loewdin_p,
        mayer_valence_total,
        heavy_props,
        shielding_diamagnetic, shielding_paramagnetic,
        span, skew, asymmetry, anisotropy
    ):
        """Set general settings valid for every derived class."""

        self.rel_contribution = rel_contribution
        self.shielding_low = shielding_low
        self.sum_atomic_numbers_prim = sum_atomic_numbers_prim
        self.sum_masses_prim = sum_masses_prim
        self.sum_atomic_numbers_sec = sum_atomic_numbers_sec
        self.sum_masses_sec = sum_masses_sec
        self.num_heavy_atoms = num_heavy_atoms
        self.av_mass_atoms = av_mass_atoms
        self.atomic_charge_mulliken = atomic_charge_mulliken
        self.atomic_charge_loewdin = atomic_charge_loewdin
        self.orbital_charge_mulliken_s = orbital_charge_mulliken_s
        self.orbital_charge_loewdin_s = orbital_charge_loewdin_s
        self.orbital_charge_mulliken_p = orbital_charge_mulliken_p
        self.orbital_charge_loewdin_p = orbital_charge_loewdin_p
        self.mayer_valence_total = mayer_valence_total
        self.heavy_props = heavy_props
        self.shielding_diamagnetic = shielding_diamagnetic
        self.shielding_paramagnetic = shielding_paramagnetic
        self.span = span
        self.skew = skew
        self.asymmetry = asymmetry
        self.anisotropy = anisotropy

    def set_acsf(self, acsf_data):
        """Store ACSF data if used."""

        self.include_acsf = True
        self.acsf_data = acsf_data

    def set_soap(self, soap_data):
        """Store SOAP data if used."""

        self.include_soap = True
        self.soap_data = soap_data


class soDataPointH(soDataPoint):
    """Class managing data for a 1H nucleus within the Delta_SO_ML model."""

    def set_attr_special_rel(
        self, refshielding_low,
        cn_d3, number_hyx,
        distance_hx, bond_order_loewdin, bond_order_mayer
    ):
        """Set additional settings valid only for a 1H datapoint."""

        self.element = 'H'
        self.shift_low = refshielding_low - self.shielding_low
        self.cn_d3 = cn_d3
        self.number_hyx = number_hyx
        self.distance_hx = distance_hx
        self.bond_order_loewdin = bond_order_loewdin
        self.bond_order_mayer = bond_order_mayer
    

    def get_header_rel(self):
        """Create the header for a 1H ML data file."""

        nvar = 73
        if self.print_names: nvar += 3
        if self.include_acsf: nvar += len(addso.acsf_labels)
        if self.include_soap: nvar += len(addso.soap_labels_reduced)

        header_1 = "# " + " ".join([str(i) for i in range(nvar)])
        header_2 = "# so_contribution shift_low CN(X) no_HYH no_HYC no_HYN no_HYO no_HYS no_HYCl no_HYZn no_HYGa no_HYGe no_HYAs no_HYSe no_HYBr no_HYCd no_HYIn no_HYSn no_HYSb no_HYTe no_HYI no_HYHg no_HYTl no_HYPb no_HYBi sum_atnum_prim sum_mass_prim sum_atnum_sec sum_mass_sec no_HA_1 no_HA_2 no_HA_3 no_HA_4 no_HA_5 av_mass_1 av_mass_2 av_mass_3 av_mass_4 av_mass_5 HA1_atnum HA1_CN HA1_atchrg HA1_orbocc_s HA1_orbocc_p HA1_orbocc_d HA2_atnum HA2_CN HA2_atchrg HA2_orbocc_s HA2_orbocc_p HA2_orbocc_d HA3_atnum HA3_CN HA3_atchrg HA3_orbocc_s HA3_orbocc_p HA3_orbocc_d dist_HX shielding_dia shielding_para span skew asymmetry anisotropy at_charge_mull at_charge_loew orb_charge_mull_s orb_charge_mull_p orb_charge_loew_s orb_charge_loew_p BO_loew BO_mayer mayer_VA"
        if self.print_names: header_2 = "# compound structure atom" + header_2[1:]
        if self.include_acsf: header_2 += " " + " ".join(addso.acsf_labels)
        if self.include_soap: header_2 += " " + " ".join(addso.soap_labels_reduced)

        return [header_1, header_2]
    

    def get_printout_rel(self, dc=4):
        """Get the whole printout for a 1H ML data file."""

        if self.print_names:
            beginning = [self.name[:dc], self.name[dc+1:], self.atom]
        else:
            beginning = []

        line = (
            beginning
            + [self.rel_contribution, self.shift_low, self.cn_d3]
            + self.number_hyx
            + [
                self.sum_atomic_numbers_prim, self.sum_masses_prim,
                self.sum_atomic_numbers_sec, self.sum_masses_sec
            ]
            + self.num_heavy_atoms + self.av_mass_atoms + self.heavy_props
            + [
                self.distance_hx,
                self.shielding_diamagnetic, self.shielding_paramagnetic,
                self.span, self.skew, self.asymmetry, self.anisotropy,
                self.atomic_charge_mulliken, self.atomic_charge_loewdin,
                self.orbital_charge_mulliken_s, self.orbital_charge_mulliken_p,
                self.orbital_charge_loewdin_s, self.orbital_charge_loewdin_p,
                self.bond_order_loewdin, self.bond_order_mayer, self.mayer_valence_total
            ]
        )

        if self.include_acsf: line += [self.acsf_data[acsf_name] for acsf_name in addso.acsf_labels]
        if self.include_soap: line += [self.soap_data[soap_name] for soap_name in addso.soap_labels_reduced]

        return " ".join([str(val) for val in line])


class soDataPointC(soDataPoint):
    """Class managing data for a 13C nucleus within the Delta_SO_ML model."""

    def set_attr_special_rel(
        self, refshielding_low, cn_d3, number_cx, number_cyx,
        orbital_charge_mulliken_d, orbital_charge_loewdin_d, orbital_stdev_mulliken_p, orbital_stdev_loewdin_p,
        bond_orders_loewdin_sum, bond_orders_mayer_sum, bond_orders_loewdin_av, bond_orders_mayer_av
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


    def get_header_rel(self):
        """Create the header for a 13C ML data file."""

        nvar = 104
        if self.print_names: nvar += 3
        if self.include_acsf: nvar += len(addso.acsf_labels)
        if self.include_soap: nvar += len(addso.soap_labels_reduced)

        header_1 = "# " + " ".join([str(i) for i in range(nvar)])
        header_2 = "# so_contribution shift_low CN(X) no_CH no_CC no_CN no_CO no_CS no_CCl no_CZn no_CGa no_CGe no_CAs no_CSe no_CBr no_CCd no_CIn no_CSn no_CSb no_CTe no_CI no_CHg no_CTl no_CPb no_CBi no_CYH no_CYC no_CYN no_CYO no_CYS no_CYCl no_CYZn no_CYGa no_CYGe no_CYAs no_CYSe no_CYBr no_CYCd no_CYIn no_CYSn no_CYSb no_CYTe no_CYI no_CYHg no_CYTl no_CYPb no_CYBi sum_atnum_prim sum_mass_prim sum_atnum_sec sum_mass_sec no_HA_1 no_HA_2 no_HA_3 no_HA_4 av_mass_1 av_mass_2 av_mass_3 av_mass_4 HA1_atnum HA1_CN HA1_atchrg HA1_orbocc_s HA1_orbocc_p HA1_orbocc_d HA2_atnum HA2_CN HA2_atchrg HA2_orbocc_s HA2_orbocc_p HA2_orbocc_d HA3_atnum HA3_CN HA3_atchrg HA3_orbocc_s HA3_orbocc_p HA3_orbocc_d HA4_atnum HA4_CN HA4_atchrg HA4_orbocc_s HA4_orbocc_p HA4_orbocc_d shielding_dia shielding_para span skew asymmetry anisotropy at_charge_mull at_charge_loew orb_charge_mull_s orb_charge_mull_p orb_charge_mull_d orb_stdev_mull_p orb_charge_loew_s orb_charge_loew_p orb_charge_loew_d orb_stdev_loew_p BO_loew_sum BO_loew_av BO_mayer_sum BO_mayer_av mayer_VA"
        if self.print_names: header_2 = "# compound structure atom" + header_2[1:]
        if self.include_acsf: header_2 += " " + " ".join(addso.acsf_labels)
        if self.include_soap: header_2 += " " + " ".join(addso.soap_labels_reduced)

        return [header_1, header_2]


    def get_printout_rel(self, dc=4):
        """Get the whole printout for a 13C ML data file."""

        if self.print_names:
            beginning = [self.name[:dc], self.name[dc+1:], self.atom]
        else:
            beginning = []

        line = (
            beginning
            + [self.rel_contribution, self.shift_low, self.cn_d3]
            + self.number_cx + self.number_cyx
            + [
                self.sum_atomic_numbers_prim, self.sum_masses_prim,
                self.sum_atomic_numbers_sec, self.sum_masses_sec,
            ]
            + self.num_heavy_atoms + self.av_mass_atoms + self.heavy_props
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

        if self.include_acsf: line += [self.acsf_data[acsf_name] for acsf_name in addso.acsf_labels]
        if self.include_soap: line += [self.soap_data[soap_name] for soap_name in addso.soap_labels_reduced]

        return " ".join([str(val) for val in line])

