import sys
import numpy as np
sys.path.append( "/Users/sangiole/Dropbox/Papers_data_live/Australia-immunology/fit_curves/fluorescence_fit" )
from theory import sample_function, sample_constant, find_best_fit, calculate_onset_stat
from theory import plot_fitted_curve, extract_and_clean_data, model_variance 
from theory import no_signal, compare_models_using_p_value


def run_fitting(
    my_file="Binding_curve_values.xlsx",
    x_name="X - IgM",
    y_name="Y - binding signal",
    sheet_name="P1 ancestor - IgM cells only",
    bounds_A=(0, 1000),
    guess_A = None,
    bounds_B=(1, 5 * 10**4),
    guess_B = None,
    bounds_C=(10 ** (-9), 1.0),
    guess_C = None,
    bounds_D=(1, 25),
    guess_D = None,
    bounds_E=(10 ** (-9), 1.0),
    guess_E = None,
    mc_runs = None, 
    n_hopping = None, 
    T_hopping = None,
    p_for_significance = 0.01,
    verbose = False,
):
    """
    my_file : str = Name of the excel file containing the data
    x_name : str =  Name of the column containing the X values ( IgM number )
    y_name : str =  Name of the column containing the Y values ( Fluorescence )
    sheet_name : str = Name of the specific sheet containing the data

    - bounds (lower, upper) for the parameters inside the fitting function

    bound_A = ( 0, 1000 )       # Defines the lower bound for the measured intensity
    bound_B = ( 1, 2 * 10**4 )  # Approximately defines the upper bound for the
                                # measured intensity
    bound_C = ( 10**(-9), 1.0 ) # Approximately ( 1 /N_bind), where N_bind is the total
                                # number of binding sites on a cell.
                                # Order of magnitude is A_construct / A_cell, where
                                # A_cell is the surface area of a cell and
                                # A_construct the area occupied by a single construct
    bound_D = ( 1, 25 )         # Number of ligands per binding construct, lower bound always 1,
                                # upper bound depends on the construct itself
    bound_E = ( 10**(-9), 1.0 ) # Number concentration of binding construct times
                                # binding volume.
                                # Rule of thumb, binding volume is of the order
                                # of the size of the construct, often ~nm^3"""

    print( f"""This code takes the data from {my_file}, specifically, from the sheet: {sheet_name}
           and uses the columns {x_name} as X and {y_name} as Y data, then tries to fit
           the data using a formula related to multivalent binding as well as a simple model assuming 
           no signal (null hypothesis). It then uses f statistics with a p value of {p_for_significance} to 
           decide about the statistical significance of the multivalent model vs no-signal. If the model
           is deemed significant (signal present), calculates quantities like the onset of adsorption.
           Always returns also a graph of the data vs best fitting (constant line in case no signal hypothesis 
           is calculated as more likely to represent the data.
           (Additional details in the DETAILS.txt file)""" )

    name = sheet_name
    output_file = f"OUTPUT_{name}.txt"
    output_graph = f"LogLogPlot_{name}.pdf"
    bounds = [bounds_A, bounds_B, bounds_C, bounds_D, bounds_E]

    # Initial guess for parameters
    # Adjust based on the expected parameters if needed
    if guess_A is None:
      guess_A = (max(bounds_A) + min(bounds_A)) / 2.0
    if guess_B is None:
      guess_B = (max(bounds_B) + min(bounds_B)) / 2.0
    if guess_C is None:
      guess_C = (max(bounds_C) + min(bounds_C)) / 2.0
    if guess_D is None:
      guess_D = (max(bounds_D) + min(bounds_D)) / 2.0
    if guess_E is None:
      guess_E = (max(bounds_E) + min(bounds_E)) / 2.0
    
    # If None set to default value used for paper
    if mc_runs is None:
      mc_runs = 8 
    if n_hopping is None:
      n_hopping = 2000
    if T_hopping is None:
      T_hopping = 3
    if verbose is None:
      verbose = False

    initial_guess = [guess_A, guess_B, guess_C, guess_D, guess_E]
    print( f"A: interval {bounds_A}, initial guess: {guess_A}" )
    print( f"B: interval {bounds_B}, initial guess: {guess_B}" )
    print( f"C: interval {bounds_C}, initial guess: {guess_C}" )
    print( f"D: interval {bounds_D}, initial guess: {guess_D}" )
    print( f"E: interval {bounds_E}, initial guess: {guess_E}" )

    print( "Parameters for running Basin Hopping Monte Carlo optimisation" )
    print( f"Number of MC runs {mc_runs}")
    print( f"Number of hopping steps x MC run {n_hopping}")
    print( f"Effective temperature for MC run: {T_hopping}")
    print( f"Chose p value for significance w.r.t. no signal (null hypothesis) {p_for_significance}")
    print( f"Maximum verbosity of output? {verbose}")

    print( "Try fitting model where signal is assumed to exist" )
    
    x_data_all, y_data_all = extract_and_clean_data( my_file, name, x_name, y_name )

    #First try to fit a model where there should be a signal present
    all_opt_params, weights, best_sample, minimum_loss = find_best_fit( 
                                                                      x_data = x_data_all, 
                                                                      y_data = y_data_all, 
                                                                      bounds = bounds, 
                                                                      initial_guess = initial_guess, 
                                                                      function_type = "multivalent",
                                                                      onset_fitting = True,
                                                                      verbose = verbose,
                                                                      mc_runs = mc_runs, 
                                                                      n_hopping = n_hopping, 
                                                                      T_hopping = T_hopping )
    best_param_multivalent = {} 
    best_param_multivalent[ "a" ] = all_opt_params[ best_sample ][ 0 ]
    best_param_multivalent[ "b" ] = all_opt_params[ best_sample ][ 1 ]
    best_param_multivalent[ "c" ] = all_opt_params[ best_sample ][ 2 ]
    best_param_multivalent[ "d" ] = all_opt_params[ best_sample ][ 3 ]
    best_param_multivalent[ "e" ] = all_opt_params[ best_sample ][ 4 ]
  
    print( f"Try fitting model where NO signal is assumed (constant)" )
    #Next, assume a model where there is no signal
    all_opt_params_const, weights_const, best_sample_const, minimum_loss_const = find_best_fit( 
                                                                                    x_data = x_data_all, 
                                                                                    y_data = y_data_all, 
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

    # Calculate the f-statistic of the model and from that the p_value. Uses p_min to decide if the test is passed or not
    f_stat, p_value, test_passed = compare_models_using_p_value( 
                                                       func1 = no_signal, 
                                                       params1 = best_param_constant, 
                                                       func2 = sample_function, 
                                                       params2 = best_param_multivalent, 
                                                       x_data = x_data_all, 
                                                       y_data = y_data_all, 
                                                       p_min = p_for_significance, 
                                                       logarithmic = True
                                                      )

    # This is a bit redundant but I put it for clarity
    signal_present = test_passed

    print( f"Signal encountered: calculated f_statistics = {f_stat}, p_value = {p_value}, p used for significance {p_for_significance}" )
    #Ok, now also save this data
    stat_file = f'stat_{sheet_name}.txt'
    with open( stat_file, 'w+' ) as my_f:
      my_f.write( f"Complex model is a better representation? {signal_present} \n" )
      my_f.write( f"calculated f_statistics: {f_stat} \n" )
      my_f.write( f"calculated p_value: {p_value} \n" )
      my_f.write( f"p used for significance: {p_for_significance} \n" )

    if signal_present:
      function_type = "multivalent"
      tag = ""
      # In this case, save, model predict there is a signal and saves data for the onset
      ave_opt_params, error_onset, average_onset, best_onset_coeffs =  calculate_onset_stat( x_data_all, 
                                                  y_data_all, 
                                                  all_opt_params, 
                                                  weights, best_sample, minimum_loss, 
                                                  output = output_file, verbose = False, 
                                                  save_data = True )

      plot_fitted_curve( x_data_all, y_data_all, all_opt_params, 
                 best_sample = best_sample,
                 onset_coeffs = best_onset_coeffs, 
                 function_type = function_type, 
                 graph_name = output_graph, 
                 verbose = False,
                 same_scale = False )
      onset_str = f"{best_onset_coeffs[ 'onset' ]}"
    else:
      function_type = "constant"
      output_graph = output_graph[:-4] #Remove the .pdf tag at the end so I can replace it
      tag = "_SIGNAL_NOT_DETECTED.pdf"
      best_onset_coeffs = {}
      # Plot the graph. Add Tag SIGNAL_NOT_DETECTED if the signal is undetectable = model assuming no binding better 
      # fits the data
      plot_fitted_curve( x_data_all, y_data_all, all_opt_params_const, 
                   best_sample = best_sample_const,
                   onset_coeffs = best_onset_coeffs, 
                   function_type = function_type, 
                   graph_name = output_graph + tag, 
                   verbose = False,
                   same_scale = False )
      #Basically, all you can say is that onset should be larger than max value of x sampled
      onset_str = f"> {np.max( x_data_all )}"

  
    summary = open( "SUMMARY.txt", 'w+' )
    summary.write( f"Name of datafile = {my_file} \n" )
    string = f"Data series: {name}, signal present: {signal_present}, Calculated p-value: {p_value} \n"
    string2 = f"Calculated onset: {onset_str}" 
    print( string ) 
    print( string2 ) 
    summary.write(  string + "\n" )
    summary.write(  string2 )
    summary.close()

    return
