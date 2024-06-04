import sys
sys.path.append( "/Users/sangiole/Dropbox/Papers_data_live/Australia-immunology/fit_curves/fluorescence_fit" )
from theory import calculate_onset, sample_function 

# Change the line belows to import data from the correct excel files and their internal pages
my_file = "Expt 10 IgM+ only IgG-dextran.xlsx"
x_name = 'bv421 IgM'
y_name = 'pe Klickmer'
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

print( f"""This code takes the data from {my_file}, specifically, from the sheets {sheet_names}
	and uses the columns {x_name} as X and {y_name} as Y data, then tries to fit
        the data using a formula related to multivalent binding (details in the DETAILS.txt 
        file)""" )

for name in sheet_names:
  output_file = f'output_{name}.txt'
  output_graph = f'LogLog_{name}.pdf'
 
  # bounds (lower, upper) for the parameters inside the fitting function
  bound_A = ( 0, 1000 )       # Defines the lower bound for the measured intensity
  bound_B = ( 1, 2 * 10**4 )  # Approximately defines the upper bound for the measured intensity
  bound_C = ( 10**(-9), 1.0 ) # Approximately ( 1 /N_bind), where N_bind is the total number of binding sites on a cell. 
                        # Order of magnitude is A_construct / A_cell, where A_cell is the surface area of a cell and 
                        # A_construct the area occupied by a single construct  
  bound_D = ( 1, 25 )         # Number of ligands per binding construct, lower bound always 1, upper bound depends on the construct itself      
  bound_E = ( 10**(-9), 1.0 ) # Number concentration of binding construct times binding volume. Rule of thumb, binding volume is of the order
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
  mc_runs = 8
  n_hopping = 2000
  T_hopping = 3
  x_data1, y_data1, all_opt_params, weights, best_sample, minimum_loss = find_best_fit( my_file, sheet_name, x_name, 
                                                                                    y_name, bounds, initial_guess, 
                                                                                    function_type = "multivalent",
                                                                                    onset_fitting = False,
                                                                                    verbose = False,
                                                                                    mc_runs = mc_runs, n_hopping = n_hopping, T_hopping = T_hopping )
  best_param_multivalent = all_opt_params[ best_sample ]
  model_variance = minimum_loss

  x_data2, y_data2, all_opt_params, weights, best_sample, minimum_loss = find_best_fit( my_file, sheet_name, x_name, 
                                                                                    y_name, bounds, initial_guess, 
                                                                                    function_type = "constant",
                                                                                    onset_fitting = False,
                                                                                    verbose = False,
                                                                                    mc_runs = mc_runs, n_hopping = n_hopping, T_hopping = T_hopping )
  best_param_constant = all_opt_params[ best_sample ]
  constant_variance = minimum_loss

  assert x_data1.all() == x_data2.all(), AssertionError( "Unexpected, all final data used should be the same" )
  assert y_data1.all() == y_data2.all(), AssertionError( "Unexpected, all final data used should be the same" )


  bayes_factor, signal_present = bayes_model_vs_uniform( model = sample_function, 
                                                         model_variance = model_variance, 
                                                         model_parameters = best_param_multivalent, 
                                                         parameters_bounds = bounds,
                                                         constant_variance = constant_variance, 
                                                         constant_parameters = best_param_constant[ "a" ], 
                                                         constant_bounds = [ bounds[ 0 ] ],
							 x_data = x_data1, 
                                                         y_data = y_data1, 
                                                         prior = 'uniform', 
                                                         verbose = False ) 

  if signal_present:
    NO NEED TO RECALCULATE IT HERE SIMPLY PLOT IT YOU HAVE ALL THE DATA NEEDED!
  else:
    print( 
    calculate_onset( my_file, name, x_name, y_name, bounds, initial_guess, 
                   output = output_file, 
                   graph_name = output_graph, 
                   verbose = True, 
                   same_scale = False,
                   mc_runs = 8, 
                   n_hopping = 2000,
                   T_hopping = 3 )
