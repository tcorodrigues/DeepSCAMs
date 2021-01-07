DeepSCAMs is a deep learning tool to help identifying small colloidally aggregating molecules (SCAMs) at a typical primary screen concentration.

The software is written in the Python 2.7 programming language and uses the following dependencies, which should be installed:

- NumPy (>= 1.11.3)
- Pandas (>= 0.19.2)
- Scikit-learn (>= 0.18.1)
- RDKit (>= 2016.03.3)





========================================================================================

DeepSCAMs.py

This script creates one text file with predicted class (0 or 1) for each molecule and their probabilities.

Class 0 - non-SCAM
Class 1 - SCAM


The script does not require editing.

For predictions, a "test.txt" file should be provided containing molecule ID and SMILES. A sample file is provided.
 strings.

To run the script, open a terminal in your destiny folder and type:

>>> python DeepSCAMs.py

========================================================================================




