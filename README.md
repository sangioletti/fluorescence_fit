# fluorescence_fit
Copyright Stefano Angioletti-Uberti 2024
email: sangiole@imperial.ac.uk
This code is distributed under a very permissive licence, see details in the LICENCE file.

Repository with the code to fit fluorescence data for Ig-mediated adsorption.

Needs the following libraries to be installed (they usually come with a standard anaconda installation):

- numpy 
- scipy
- python (version 3.8 or higher works, lower versions not tested)

After modifying the file fitting_data.py with the necessary entries, the code can be simply executed using

python fitting_data.py

The code generates a graph with the experimental data, the best fitting curve and the geometric construction used to determine it. The code also generates an OUTPUT.txt file with the results of the fitting.
