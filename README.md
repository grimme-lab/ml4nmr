# ML4NMR
Machine learning-based correction for computed NMR chemical shifts.

This program enables the correction of <sup>1</sup>H and <sup>13</sup>C NMR chemical shifts calculated with DFT toward CCSD(T) quality (&Delta;<sub>corr</sub>ML) and the prediction of spin-orbit relativistic contributions to NMR chemical shifts caused by heavy atoms (&Delta;<sub>SO</sub>ML).

## Introduction
This repository contains two major functionalities: the &Delta;<sub>corr</sub>ML and the &Delta;<sub>SO</sub>ML model.

First, the &Delta;<sub>corr</sub>ML method can be used to correct NMR chemical shifts calculated with DFT toward CCSD(T) quality (original publication: https://doi.org/10.1021/acs.jctc.3c00165). All files related to this method contain the suffix `corr`.

The second method, &Delta;<sub>SO</sub>ML, was published later (original publication: https://doi.org/10.1039/d3cp05556f) and predicts the relativistic contribution to a chemical shift based on a non- or scalar-relativistic DFT calculation. All files related to this method contain the suffix `so`.

For both methods, the procedure is done in two steps:

1. Data acquisition, program prefix `getdata`: Read and process the data from a set of calculations or a sample molecule.
2. Correction, program prefix `mlcorrect`: Train the ML model or predict the desired quantity for a sample molecule.

This repository also contains the data set files which are identical with those used in the publications and can be used to reproduce the results.

Note that for now, only DFT calculation output files from ORCA 5 can be processed. ORCA is free of charge for academic use. https://www.faccts.de/orca/


## Installation
The easiest way to install the package is to clone this directory, place it in a directoy included in your `$PATH` variable, and create symbolic links to the relevant scripts:

```bash
git clone https://github.com/grimme-lab/ml4nmr.git .
```

```bash
ln -s /path/to/your/cloned/ml4nmr/src/getdata_corr.py getdata_corr
ln -s /path/to/your/cloned/ml4nmr/src/getdata_so.py getdata_so
ln -s /path/to/your/cloned/ml4nmr/src/mlcorrect_corr.py mlcorrect_corr
ln -s /path/to/your/cloned/ml4nmr/src/mlcorrect_so.py mlcorrect_so
```

Since the projects have grown over a considerable amount of time, the `corr` variants work with Python 3.7 and TensorFlow 2.7 while the newer `so` use Python 3.11 and TensorFlow 2.12. The `conda` environments used the calculations in the original publications are also given in `.yml` files for a semiautomatic setup.

```bash
conda env create -f env_corr.yml
conda activate ml4nmr-corr
```

```bash
conda env create -f env_so.yml
conda activate ml4nmr-so
```


## Usage
### 1. Data Acquisition
The explanations in this paragraph hold for `getdata_corr` and `getdata_so` in the same way but the `corr` variant is presented exemplarily.

In order to read the data of a sample molecule, `getdata --sample` needs an XYZ file of the molecule (the same as the one used for the DFT calculation), the calculation output (ORCA 5 output file), and the path to the directory with the same data for the reference compound. This will automatically generate two data files for <sup>1</sup>H and <sup>13</sup>C data, respectively.

```bash
getdata_corr --sample mol.xyz orca.out /path/to/reference
```

In `/path/to/reference`, there must be a directory tree with the names of the reference compound, the chosen density functional and the basis set. For instance, if the reference compound is chosen to be named `tms` and the NMR shieldings were calculated with PBE0/pcSseg-2 (named `pbe0` and `pcSseg-2`), there should be a directory `/path/ro/reference/tms/pbe0/pcSseg-2` which contains at least the files `tms.xyz` and `orca.out`.

If other methods or reference compounds were used or a different data shuffle mode shall be selected, this should be reflected in the command-line arguments. For example:

```bash
getdata_corr --sample mol.xyz orca.out /path/to/reference --functional_low pbe --basis_low def2-TZVP --reference ch4 --shuffle compounds --print_names
```

For more information on the possible command-line options, consult `getdata_corr --help` and `getdata_so --help`, the source code or the publicatioins (Supplementary Information).


### 2. Correction
After a data file has been generated, say, `ml_mol_h.dat`, `mlcorrect` can be used to predict the target quantity. For that, also a pre-trained model will be needed (see below), which is a directory called `tf_model_h` in this example. Furthermore, the nucleus of interest (`h` or `c`) should be indicated. The results will be piped wo stdout and a file with a list of all atoms and their predicted values will be written.

```bash
mlcorrect_corr --predict ml_mol_h.dat tf_model_h --nucleus h
```

Again, the predictions of spin-orbit relativistic contributions for NMR chemical shifts works in the same way using `mlcorrect_so`.


## Advanced Usage
### Training an ML Model
The `mlcorrect` methods can also be used to train an ML model with the `--train` flag. For this, a dataset file generated by `getdata` from a large amount of calculations is needed. For all method combinations investigated in the original publications, these data files are provided in [`ml4nmr/data_sets`](data_sets). For instance, a training run with default settings for the ML model and data based on the PBE0/def2-TZVP method for the <sup>13</sup>C nucleus is started with:

```bash
mlcorrect_corr --train ml_pbe0_def2-TZVP_c.dat --nucleus c
```

When the training has finished, the model is saved in a directory called `tf_model_c` (or `_h` for <sup>1</sup>H) which can be used with `mlcorrect --predict` (see above).

With various command-line options, the hyperparameters (e.g., number of nodes in the hidden layers, dropout rate, optimizer, activation function, loss function) of the ML model can be modified. For mor information, please refer to `mlcorrect_corr --help` and `mlcorrect_so --help`, the source code or the publicatioins (Supplementary Information).


### Read a Dataset
This option is not straightforward and requires some studying of the source code.

If many calculations have been performed, they must be structured in a specific directory system in order to be read by `getdata`. Currently, this is hard-coded for the two data sets used in the original publications.

#### `getdata_corr`
For the &Delta;<sub>corr</sub>ML method, ORCA calculation files must be in the following subdirectories:

```
XXX/YY/FUNC/BAS/
```

where `XXX` is the 3-digit compound number (001-100, adjust the main function of `getdata_corr.py` if other compounds are desired), `YY` is the 2-digit structure number (00-09), and `FUNC` and `BAS` are the names of the density functional and basis set, respectively. Each directory must at least contain the files `XXX_YY.xyz` and `orca.out`. Additionally, the proper ORCA and CFOUR calculation output files must be in the following directories in order to get the target reference values.

```
XXX/YY/bhlyp/pcSseg-2/
XXX/YY/bhlyp/pcSseg-3/
XXX/YY/bhlyp/pcSseg-4/
XXX/YY/ccsd_t/pcSseg-2/
```

The data for the reference compound must be handled in the same way. In such a directory system, the dataset can be read via the following statement and further command-line options are available. The data files are generated automatically and some statistics are printed to stdout.

```bash
getdata_corr --set /path/to/the/directory
```

#### `getdata_so`
Again, the script is very similar for the &Delta;<sub>SO</sub>ML method but some important details differ. The compound number `XXXX` has to be 4-digit (0001-1597), there are only four structures per compound (`YY`, 00-03), and the target reference values have to be provided in a `rel_contributions.json` file. When all these conditions are met, the data can be read as before.

```bash
getdata_so --set /path/to/the/directory
```
