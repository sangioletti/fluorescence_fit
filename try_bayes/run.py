import sys
#sys.path.append( "/Users/sangiole/Github-repos/fluorescence_fit/" )
sys.path.append( "/Users/sangiole/Dropbox/Papers_data_live/Australia-immunology/fit_curves/fluorescence_fit" )
#sys.path.append( "/Users/sangiole/Dropbox/Papers_data_live/Australia-immunology/fit_curves/fluorescence_fit/sandbox" )
from theory import sample_function, find_best_fit, calculate_onset_stat, plot_fitted_curve, extract_and_clean_data 
from theory_model_comparison import bayes_model_vs_uniform
import numpy as np
# Change the line belows to import data from the correct excel files and their internal pages
my_file = "Expt 10 IgM+ only IgG-dextran.xlsx"
x_name = 'bv421 IgM'
y_name = 'pe Klickmer'
sheet_names = [ 
                'P1 ancestor', 
                #'P1 mutated', 
                #'P2 ancestor', 
                #'P2 mutated', 
                #'P3 ancestor', 
                'P3 mutated', 
                'P4 ancestor', 
                #'P4 mutated', 
                 ] 

print( f"""This code takes the data from '{my_file}', specifically, from the sheets '{sheet_names}'
	and uses the columns '{x_name}' as X and '{y_name}' as Y data, then tries to fit
        the data using a formula related to multivalent binding. 
        (See details in the DETAILS.txt file)""" )

for name in sheet_names:
  output_file = f'output_{name}.txt'
  output_graph = f'LogLog_{name}.pdf'
  x_data, y_data = extract_and_clean_data( my_file, name, x_name, y_name )

  # bounds (lower, upper) for the parameters inside the fitting function
  minA = np.min( y_data ) 
  maxB = np.max( y_data ) 
  bound_A = ( minA, maxB )       # Defines the lower bound for the measured intensity
  bound_B = ( 0.0, maxB )  # Approximately defines the upper bound for the measured intensity
  bound_C = ( 10**(-5), 10**(-1) ) # Approximately ( 1 /N_bind), where N_bind is the total number of binding sites on a cell. 
                        # Order of magnitude is A_construct / A_cell, where A_cell is the surface area of a cell and 
                        # A_construct the area occupied by a single construct  
  bound_D = ( 1, 12 )   # Number of ligands per binding construct, lower bound always 1, upper bound depends on the construct itself      
  bound_E = ( 10**(-8), 10**(-5) ) # Number concentration of binding construct times binding volume. Rule of thumb, binding volume is of the order
                        # of the size of the construct, often ~nm^3 

  bounds = [ bound_A, bound_B, bound_C, bound_D, bound_E ]

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

  #Ok, so let's first fit the data so we obtain the variance around the best fit
  #mc_runs = 8 
  #n_hopping = 2000
  mc_runs = 1 
  n_hopping = 1000
  T_hopping = 3
  
  #Parameter for bayes sampling                                                       
  eps = 10**(-3) 
  n_blocks = 10 
  n_steps = 10**4 
  max_step = 10**5 


  print( "Try fitting model where signal is assumed to exist" )
  #First try to fit a model where there should be a signal present


  all_opt_params, weights, best_sample, minimum_loss = find_best_fit( x_data, y_data, 
                                                                                    bounds = bounds, 
                                                                                    initial_guess = initial_guess, 
                                                                                    function_type = "multivalent",
                                                                                    onset_fitting = False,
                                                                                    verbose = False,
                                                                                    mc_runs = mc_runs, 
                                                                                    n_hopping = n_hopping, 
                                                                                    T_hopping = T_hopping )
  best_param_multivalent = {} 
  best_param_multivalent[ "a" ] = all_opt_params[ best_sample ][ 0 ]
  best_param_multivalent[ "b" ] = all_opt_params[ best_sample ][ 1 ]
  best_param_multivalent[ "c" ] = all_opt_params[ best_sample ][ 2 ]
  best_param_multivalent[ "d" ] = all_opt_params[ best_sample ][ 3 ]
  best_param_multivalent[ "e" ] = all_opt_params[ best_sample ][ 4 ]

  print( f"Loss for model is: {minimum_loss}" )
  
  print( f"Try fitting model where NO signal is assumed (constant)" )
  #Next, assume a model where there is no signal
  all_opt_params_const, weights_const, best_sample_const, minimum_loss_const = find_best_fit( x_data, y_data, 
                                                                                    bounds = bounds, 
                                                                                    initial_guess = initial_guess, 
                                                                                    function_type = "constant",
                                                                                    onset_fitting = False,
                                                                                    verbose = False,
                                                                                    mc_runs = mc_runs, 
                                                                                    n_hopping = n_hopping, 
                                                                                    T_hopping = T_hopping )
  best_param_constant = {} 
  best_param_constant[ "a" ] = all_opt_params_const[ best_sample_const ][ 0 ]
  
  #Take as the variance of the data the minimum between that calculated with the two models  
  data_variance = min( minimum_loss, minimum_loss_const )

  print( f"Loss for constant model is: {minimum_loss_const}" )


  amax = np.max( bounds[ 0 ] ) 
  bmax = np.max( bounds[ 1 ] ) 

  amin = np.min( bounds[ 0 ] ) 
  bmin = np.min( bounds[ 1 ] ) 

  print( f"Max value for a = {amax}" )
  print( f"Max value for b = {bmax}" )
  print( f"Min value for a = {amin}" )
  print( f"Min value for b = {bmin}" )

  print( f"Max value for log(a+b) {np.log( amax + bmax )}" )
  print( f"Min value for log(a+b) = log( a ) = {np.log( amin )}" )

  #Estimate if model assuming binding exist is better than model with no binding

  bayes_factor, signal_present = bayes_model_vs_uniform( 
                                                         data_variance = data_variance, 
                                                         model = sample_function, 
                                                         model_parameters = best_param_multivalent, 
                                                         parameters_bounds = bounds,
                                                         constant_parameters = best_param_constant, 
                                                         constant_bounds = [ bounds[ 0 ] ],
							 x_data = x_data, 
                                                         y_data = y_data, 
                                                         prior = 'uniform', 
                                                         verbose = True, 
                                                         eps = eps, 
                                                         n_blocks = n_blocks, 
                                                         n_steps = n_steps, 
                                                         max_step = max_step 
                                                         )

  if signal_present:
    function_type = "multivalent"
    tag = ""
    # In this case, save, model predict there is a signal and saves data for the onset
    ave_opt_params, error_onset, average_onset, best_onset_coeffs =  calculate_onset_stat( x_data, 
                                                  y_data, 
                                                  all_opt_params, 
                                                  weights, best_sample, minimum_loss, 
                                                  output = output_file, verbose = False, 
                                                  save_data = True )

    plot_fitted_curve( x_data, y_data, all_opt_params, 
                 best_sample = best_sample,
                 onset_coeffs = best_onset_coeffs, 
                 function_type = function_type, 
                 graph_name = output_graph, 
                 verbose = False,
                 same_scale = False )
  else:
    function_type = "constant"
    output_graph = output_graph[:-4] #Remove the .pdf tag at the end so I can replace it
    tag = "_SIGNAL_NOT_DETECTED.pdf"
    best_onset_coeffs = {}
    # Plot the graph. Add Tag SIGNAL_NOT_DETECTED if the signal is undetectable = model assuming no binding better 
    # fits the data
    plot_fitted_curve( x_data, y_data, all_opt_params_const, 
                   best_sample = best_sample_const,
                   onset_coeffs = best_onset_coeffs, 
                   function_type = function_type, 
                   graph_name = output_graph + tag, 
                   verbose = False,
                   same_scale = False )

  print( f"Most likely model is {function_type}" )
  quit()
