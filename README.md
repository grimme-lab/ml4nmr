# ml4nmr
Machine learning-based correction for computed NMR chemical shifts

This repository contains the scripts used to correct NMR chemical shifts calculated with DFT toward CCSD(T) quality used in the original publication: https://doi.org/10.1021/acs.jctc.3c00165.

The given data set files are identical with those used in the publication and can be used to reproduce the results.

This repository also contains the data set files used in the publication of the second method, that predicts the spin-orbit-related relativistic contribution to the chemical shift caused by heavy atoms. The paper can be found here: https://doi.org/10.1039/d3cp05556f. The code used therein will be published soon.

Currently still under construction
----------------------------------

The scripts found in this repository are not yet usable in a user-friendly way, but are already provided for completeness. Commented code and documentation will soon be provided in a first release. However, the code already does its job when executed with Python 3.7 and TensorFlow 2.7.
