# Fluorescence Fit

### Copyright Stefano Angioletti-Uberti 2024
### email: sangiole@imperial.ac.uk
### Please use the email above and/or use github to report any bug!

This code is distributed under a very permissive licence, see details in the LICENCE file.
It should be read together with the paper "NAME_OF_PAPER" from Young et al.





Repository with the code to fit fluorescence data for Ig-mediated adsorption.

Needs the following libraries to be installed (they usually come with a standard anaconda installation):

- numpy 
- scipy
- python (version 3.8 or higher works, lower versions not tested)

After modifying the file no_gui_fitting.py with the necessary entries, the code can be simply typing the command
(on a terminal or inside a jupyter notebook):

## python no_gui_fitting.py

Alternatively, a GUI can be used (better if you know little about Python / want to be quick) by typing: 

## python Graphic_user_interface.py

In this case, a GUI will appear where one can easily set the range of the different parameters. Note that the Monte Carlo
fitting procedure will not look for optimal parameters outside of those ranges. For this reason, if the final fitting is not
particularly good, it might be necessary to change them. An explanation of what the values of A-E means can be find in DETAILS.txt. 
Default values that have already been set will be normally safe, but care must be taken. Eventually, visual inspection might be the 
best way to judge the fitting as well as to provide some of the initial guesses (in particular, the values for A and B).

Regardless of using the GUI or the standard python script, the code generates 
1) A graph with the experimental data, the best fitting curve and the geometric construction used to determine it. 
2) An OUTPUT.txt file containing the results of the fitting and the calculated onset values.
3) A Summary.txt file containing the results of the statistical significance test and the calculated p-value.
