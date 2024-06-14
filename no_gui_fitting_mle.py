import sys
import numpy as np
from sklearn.gaussian_process.kernels import RBF

code_directory = "/Users/sangiole/Dropbox/Papers_data_live/Australia-immunology/fit_curves/fluorescence_fit" 

print( f"ATTENTION: Looking for python scripts in {code_directory}" )

try:
  from theory import sample_function, sample_constant, find_best_fit, calculate_onset_stat
  from theory import plot_fitted_curve, extract_and_clean_data, model_variance 
  from theory import no_signal, compare_models_using_p_value
except:
  # Change the line belows to import python package from the correct directory if needed
  sys.path.append( code_directory )
  from theory import sample_function, sample_constant, find_best_fit, calculate_onset_stat 
  from theory import plot_fitted_curve, extract_and_clean_data, model_variance 
  from theory import no_signal, compare_models_using_p_value

# Change the line belows to import data from the correct excel files and their internal pages
my_file = "Expt 10 IgM+ only IgG-dextran.xlsx"
x_name = 'bv421 IgM'
y_name = 'pe Klickmer'

summary = open( "SUMMARY.txt", 'w+' )
summary.write( f"Name of datafile = {my_file} \n" )


sheet_names = [ 
                'P1 ancestor', 
                'P1 mutated', 
                'P2 ancestor', 
                'P2 mutated', 
                'P3 ancestor', 
                'P3 mutated', 
                'P4 ancestor', 
                'P4 mutated', 
                 ] 

print( f"""This code takes the data from '{my_file}', specifically, from the sheets '{sheet_names}'
	and uses the columns '{x_name}' as X and '{y_name}' as Y data, then tries to fit
        the data using a formula related to multivalent binding. 
        (See details in the DETAILS.txt file)""" )

# PARAMETERS USED. DO NOT CHANGE UNLESS YOU KNOW WHAT YOU ARE DOING.

# Parameters for fitting using MC basin hopping
mc_runs = 8 
n_hopping = 2000
T_hopping = 3
logarithmic_fit = True

# p value to compare multivalent binding model to model where no signal is present 
p_for_significance = 0.05

# bounds (lower, upper) for the parameters inside the fitting function
bound_A = ( 0, 1000 )       # Defines the lower bound for the measured intensity
bound_B = ( 1, 5 * 10**4 )  # Approximately defines the upper bound for the measured intensity
bound_C = ( 10**(-9), 1.0 ) # Approximately ( 1 / (K_D/M) * N_bind), where N_bind is the total number of binding sites on a cell. 
                            # Order of magnitude is A_construct / A_cell, where A_cell is the surface area of a cell and 
                            # A_construct the area occupied by a single construct  
                            # Because you do not know K_D a priory, just choose a range large enough so that the fitting algo can find it.
                            #
bound_D = ( 1, 25 )         # Number of ligands per binding construct, lower bound always 1, upper bound depends on the construct itself      
bound_E = ( 10**(-9), 1.0 ) # Number concentration of binding construct times binding volume. Rule of thumb, binding volume is of the order
                      # of the size of the construct, often ~nm^3 
bounds = [ bound_A, bound_B, bound_C, bound_D, bound_E ]

for name in sheet_names:
  output_file = f'output_{name}.txt'
  stat_file = f'stat_{name}.txt'
  output_graph = f'LogLog_{name}.pdf'
 
  # Initial guess for parameters
  # Adjust based on the expected parameters if needed
  guess_A = (max( bound_A ) + min( bound_A )) / 2.0
  guess_B = (max( bound_B ) + min( bound_B )) / 2.0
  guess_C = (max( bound_C ) + min( bound_C )) / 2.0
  guess_D = (max( bound_D ) + min( bound_D )) / 2.0
  guess_E = (max( bound_E ) + min( bound_E )) / 2.0
  initial_guess = [ guess_A, guess_B, guess_C, guess_D, guess_E ]

  #Ok, now first before trying to calculate the onset, let's first check if the model is not best fit with 
  #a constant. If so, it means there is no signal for multivalent adsorption in the range of receptors/IgG 
  #sampled. In other words, the onset should be higher than the largest value of IgG present in the sampling data

  x_data_all, y_data_all = extract_and_clean_data( my_file, name, x_name, y_name )

  #Ok, so let's first fit the data so we obtain the variance around the best fit

  print( "Try fitting model where signal is assumed to exist" )
  #First try to fit a model where there should be a signal present

  optimal_parameters, onset_parameters = find_best_fit_mle( x_data, y_data, 
                   bounds, initial_guess, function_type = "multivalent",
                   kernel = RBF(),
                   alpha_gp =1e-10, # Small alpha for likelihood
                   n_restarts_optimizer_gp = 10 
                   onset_fitting = True,
                   verbose = False )
  
  best_param_multivalent = {} 
  best_param_multivalent[ "a" ] = optimal_parameters[ 0 ]
  best_param_multivalent[ "b" ] = optimal_parameters[ 1 ]
  best_param_multivalent[ "c" ] = optimal_parameters[ 2 ]
  best_param_multivalent[ "d" ] = optimal_parameters[ 3 ]
  best_param_multivalent[ "e" ] = optimal_parameters[ 4 ]
  
  print( f"Try fitting model where NO signal is assumed (constant)" )
  #Next, assume a model where there is no signal
  optimal_parameters, onset_parameters = find_best_fit_mle( x_data, y_data, 
                   bounds, initial_guess, function_type = "constant",
                   kernel = RBF(),
                   alpha_gp =1e-10, # Small alpha for likelihood
                   n_restarts_optimizer_gp = 10 
                   onset_fitting = False,
                   verbose = False )

  best_param_constant = {} 
  best_param_constant[ "a" ] = optimal_parameters[ 0 ]

  # Calculate Akaike information criterion to compare the two models

  signal_present = False 
  if delta_aic > 0:
    signal_present = True

  # This is a bit redundant but I put it for clarity
  signal_present = test_passed

  print( f"Signal encountered" )
  #Ok, now also save this data
  with open( stat_file, 'w+' ) as my_f:
    my_f.write( f"Complex model is a better representation? {signal_present} \n" )
    my_f.write( f"calculated aic for multivalent model: {aic_multi} \n" )
    my_f.write( f"calculated aic for constant signal model: {aic_const} \n" )

  if signal_present:
    function_type = "multivalent"
    tag = ""
    # In this case, save, model predict there is a signal and saves data for the onset
    with open( output_file, "w+" ) as my_f:
      my_f.write( "Optimal parameters for onset: \n" )
      for i, value in onset_parameters.items():
      my_f.write( f"{i}  {value}" )

    plot_fitted_curve_mle( x_data_all, y_data_all, 
                 optimal_parameters, 
                 onset_coeffs = onset_parameters, 
                 function_type = function_type, 
                 graph_name = output_graph, 
                 verbose = False,
                 same_scale = False,
                 mle = True )
    onset_str = f"{best_onset_coeffs[ 'onset' ]}"

  else:
    function_type = "constant"
    output_graph = output_graph[:-4] #Remove the .pdf tag at the end so I can replace it
    tag = "_SIGNAL_NOT_DETECTED.pdf"
    best_onset_coeffs = {}
    # Plot the graph. Add Tag SIGNAL_NOT_DETECTED if the signal is undetectable = model assuming no binding better 
    # fits the data
    plot_fitted_curve( x_data_all, y_data_all, 
                   optimal_parameters, 
                   onset_coeffs = onset_parameters, 
                   function_type = function_type, 
                   graph_name = output_graph + tag, 
                   verbose = False,
                   same_scale = False,
                   mle = True )
    #Basically, all you can say is that onset should be larger than max value of x sampled
    onset_str = f"> {np.max( x_data_all )}"

  
  string = f"Data series: {name}, signal present: {signal_present}, calculated onset: {onset_str}" 
  print( string ) 
  summary.write(  string + "\n" )
