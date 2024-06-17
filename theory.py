import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
from scipy.stats import f

# Define the function to fit
def sample_function( x, a, b, c, d, e):
    '''This is the fitting function used to fit the fluorescence
    intensity signal as a function of "x", the number of IgM present on a cell
    surface'''
    if ( 1.0 + c * x ).any() <= 0 and d % 1 != 0:  
        print(f"Warning: Trying to compute a complex number with x={x} and exponent={b}")
        return np.nan  # Return NaN or some other appropriate value
    p1 = ( 1.0 + c * x )**d
    return a + b * e * p1 / ( 1.0 + e * p1 )

def sample_constant( x, a, b, c, d, e ):
    '''This is the fitting function used to fit the fluorescence
    intensity signal as a function of "x", the number of IgM present on a cell
    surface'''
    return np.ones( len(x) ) * a 

def sample_log( x, a, b, c, d, e):
    '''Just returns the logarithm of the sampling function'''
    return np.log( sample_function( x, a, b, c, d, e ) )

def sample_log_constant( x, a, b, c, d, e):
    '''Just returns the logarithm of the sampling function'''
    return np.log( sample_constant( x, a, b, c, d, e ) )

def logarithmic_derivative( x, a, b, c, d, e ):
    '''Logarithmic derivative of the sampling function, defined as
    dln(f)/dln(x)
    '''
    der_sample = b * c * d * e * ( 1.0 + c * x )**(d-1.0) / (1.0 + e * (1.0 + c* x)**d )**2
    y = sample_function( x,a,b,c,d,e) 
    return x/y * der_sample
    
def model_variance( x_data, y_data, model, params, logarithmic = True ):
    '''Calculate the model variance with respect to experimental data'''
    if logarithmic:
      result = np.mean( (np.log( y_data ) - np.log( model(x_data, **params)))**2 ) 
    else: 
      result = np.mean( ( y_data - model( x_data, **params))**2 ) 
    return result


def middle( x, a, b, c, d, e ):
    '''Average between the maximum and minimum value of the logarithm  
    of the sampling function'''
    my_min = a + b * e / ( 1.0 + e )
    my_max = a + b
    log_min = np.log( my_min ) 
    log_max = np.log( my_max ) 
    log_ave = ( log_min + log_max ) / 2.0
    return log_ave

def find_middle( x, a, b, c, d, e ):
    '''Function used to solve the equation log(f) - x0 = 0, where f is the sampling function
    and x0 the mid point defined by "middle"'''
    return np.log( sample_function( x, a, b, c, d, e ) ) - middle( x, a, b, c, d, e )

def onset_value( x, y_value_min, b, alpha ):
    return np.log( b * x**alpha ) - np.log( y_value_min )
    
def calculate_onset_from_fit( x_data, y_data, opt_params, verbose = False ):
    '''This function simply take the input experimental data and the parameters of the fitted model
    and use it to calculate the onset of adsorption'''
    # Find mid point between max and minimum in logarithmic space and calculate the logarithmic derivative 
    # at that point.
    # Starting guess for mid point
    #x0 = np.max( x_data ) - np.min( x_data )
    root = scipy.optimize.bisect(find_middle, np.min( x_data), np.max( x_data ), args = tuple( opt_params ) )

    # Next line needed to format in a way I can feed this to function, probably there is a smarter way
    alpha = logarithmic_derivative( root, *opt_params )

    aa = opt_params[ 0 ]
    bb = opt_params[ 1 ]
    ee = opt_params[ 4 ]

    y_value_min = aa + bb * ee / ( 1.0 + ee )
    y_value_max = aa + bb 

    #Define the curve which in a log-log plot looks like a line with the right slope
    log_mid_point = ( np.log( y_value_min ) + np.log( y_value_max ) ) / 2.0 
    y0 = np.exp(1)**( log_mid_point )  # y value of the mid point
    b_coeff = y0 / (root**alpha)
    #y_derivative = b_coeff * x_data**alpha 
    args_onset_value = ( y_value_min, b_coeff, alpha )

    #Find onset using bisection method
    onset = scipy.optimize.bisect( onset_value, 0, np.max( x_data), args = args_onset_value )
    if verbose:
      print( f'x value corresponding to the mid point between max and min fluorescence on log-log plot = {root}' )
      print( f'y value for the midpoint is {y0}' )
      print( f'Derivative at midpoint is: {alpha}' )
      print( f"Onset found at {onset}" )

    return onset, b_coeff, alpha, y0, y_value_min


def extract_and_clean_data( my_file, sheet_name, x_name, y_name ):
    try:
      data = pd.read_excel( my_file, sheet_name = sheet_name )
    except:
      data = pd.read_csv( my_file )
      print( "Excel file not found, assuming csv" )

    x_data = data[x_name].values  
    y_data = data[y_name].values

    #Clean the data to remove values with negative intensity. Alternatively, one can simply shift all values so 
    #that the minimum intensity is positive, necessary for taking the logarithm later
    condition = y_data > 0.0
    x_data = np.extract( condition, x_data ) 
    y_data = np.extract( condition, y_data )

    return x_data, y_data

    
def find_best_fit( x_data, y_data, bounds, initial_guess, function_type = "multivalent",
                   onset_fitting = False,
                   verbose = False,
                   mc_runs = 8, n_hopping = 2000, T_hopping = 3 ):


    #Redefine bound on parameter B to be slightly higher than max_y if needed
    if bounds[ 1 ][ 1 ] < np.max( y_data ):
      boundsB = list( bounds[ 1 ] ) 
      bounds[ 1 ] = ( boundsB[ 0 ], np.max( y_data ) ) 
      print( f"Max y observed is {np.max( y_data )}" )
      print( f"Redefined bound on B to be at least as large as y_data, new upper bound is {bounds[1][1]}" )
   
    # Not good python practice...but here we define objective function within another function 
    def objective_function( params ):
        '''Define the objective function to minimize (average residuals between data and model) via the 
        basin hopping Monte Carlo procedure'''
        return np.sqrt( np.sum((np.log( y_data ) - sample_log( x_data, *params))**2) / len( y_data ) )
    
    def objective_constant( params ):
        '''This is to fit the data using a constant function ( = no signal detected )'''
        return np.sqrt( np.sum((np.log( y_data ) - sample_log_constant( x_data, *params))**2) / len( y_data ) )

    num_param = 5 # This is the number of parameters in our fitting function
    all_opt_params = np.zeros( ( mc_runs, num_param + 5 ) )
    weights = np.ones( mc_runs )

    if function_type == "multivalent":
      loss_function = objective_function
    elif function_type == "constant":
      loss_function = objective_constant
    else:
      raise NotImplementedError
    
    #Set initial state for best loss and best sample
    minimum_loss = np.inf
    best_sample = -1

    for i in range( mc_runs ):
      # Perform basin hopping optimization
      print( f'Bounds for parameters: {bounds}' )

      result = basinhopping( loss_function, initial_guess, niter = n_hopping, T = T_hopping, minimizer_kwargs={"bounds" : bounds})
      if result.fun < minimum_loss:
        print( f"Found improved solution with residual {result.fun}, parameter values {result.x}" )
        best_sample = i
        minimum_loss = result.fun
      weights[ i ] = 1.0 / result.fun

      all_opt_params[ i, :-5 ] = result.x

      if function_type == "multivalent" and onset_fitting:
          try:
              onset, b_coeff, alpha, y0, y_min_value = calculate_onset_from_fit( x_data, y_data, 
                                                                    all_opt_params[ i, :-5 ], 
                                                                    verbose = verbose )
          except ValueError:
              print( """Onset not found for this specific git, no worries, continue.
                      Problem only if no fit at all is found""" )
              onset, b_coeff, alpha, y0, y_min_value = [ 0.0, 0.0, 0.0, 0.0, 0.0 ] 
         
          all_opt_params[ i, -1 ] = onset
          all_opt_params[ i, -2 ] = b_coeff 
          all_opt_params[ i, -3 ] = alpha
          all_opt_params[ i, -4 ] = y0
          all_opt_params[ i, -5 ] = y_min_value
        

    return all_opt_params, weights, best_sample, minimum_loss


def calculate_onset_stat( x_data, y_data, all_opt_params, weights, best_sample, minimum_loss, 
		   output = 'output.txt', verbose = False,
                   save_data = True ):
    
    ave_opt_params = np.average( all_opt_params, axis = 0, weights = weights ) 
    error_onset = np.sqrt( np.var( all_opt_params[ -1 ], axis = 0 ) ) 
    average_onset = ave_opt_params[ -1 ]
    
    best_onset_coeffs = {}  
    best_onset_coeffs[ 'onset' ] = all_opt_params[ best_sample, -1 ]
    best_onset_coeffs[ 'b_coeff'] = all_opt_params[ best_sample, -2 ]
    best_onset_coeffs[ 'alpha'] = all_opt_params[ best_sample, -3 ]
    best_onset_coeffs[ 'y0'] = all_opt_params[ best_sample, -4 ]
    best_onset_coeffs[ 'y_value_min'] = all_opt_params[ best_sample, -5 ]

    if save_data:
    #Save fitting data to file and calculate averages of the fits found during MC procedure

      save_fit_data( all_opt_params, best_sample, error_onset, average_onset, best_onset_coeffs,
                   minimum_loss, 
                   function_type = "multivalent",
		   output = output, verbose = False )

    return ave_opt_params, error_onset, average_onset, best_onset_coeffs


def full_fitting( my_file, sheet_name, x_name, y_name, bounds, initial_guess,
                   function_type = "multivalent", onset_fitting = True, 
		   output = 'output.txt', graph_name = 'LogLog.pdf', verbose = False,
		   same_scale = False, mc_runs = 8, n_hopping = 2000, T_hopping = 3, save_data = True ):
    ''''Does both fitting of the curve, calculate the onset and plot the graphs'''
    x_data, y_data = extract_and_clean_data( my_file, sheet_name, x_name, y_name )

    all_opt_params, weights, best_sample, minimum_loss = find_best_fit( x_data, y_data, 
                   bounds, initial_guess, function_type = function_type, onset_fitting = onset_fitting,
                   output = output, graph_name = graph_name, verbose = verbose,
                   same_scale = same_scale, mc_runs = mc_runs, n_hopping = n_hopping, T_hopping = T_hopping )

    plot_fitted_curve( x_data, y_data, all_opt_params, weights, best_sample, 
                   function_type = "multivalent", onset_fitting = True, 
		   output = 'output.txt', graph_name = 'LogLog.pdf', verbose = False,
		   same_scale = False )
    
    ##Now average the onset and save the data
    if function_type == 'multivalent':
      calculate_onset_stat( x_data, y_data, all_opt_params, weights, best_sample, minimum_loss, 
   		   output = 'output.txt', verbose = False,
                   save_data = True )

    return

def plot_fitted_curve( x_data, y_data, all_opt_params, 
                   onset_coeffs, best_sample, 
                   function_type = "multivalent", 
		   graph_name = 'LogLog.pdf', verbose = False,
		   same_scale = False ):
    
    #Just a sanity check
    assert function_type == 'multivalent' or function_type == 'constant', AssertionError( f"Function type not recognized {function_type}" )

    ''''Take results of fitting and plot it'''

    if function_type == "multivalent":
      onset = onset_coeffs[ 'onset' ] 
      b_coeff = onset_coeffs[ 'b_coeff'] 
      alpha = onset_coeffs[ 'alpha'] 
      y0 = onset_coeffs[ 'y0'] 
      y_value_min = onset_coeffs[ 'y_value_min']
 
    # Ok, now prepare to make the plots
    x = np.logspace( np.log10( np.min( x_data )), np.log10( np.max( x_data ) ), 100 )

    if function_type == "multivalent":
      y_min = np.ones( 100 ) * y_value_min
      y_mid = np.ones( 100 ) * y0
      y_onset = np.logspace( -1,5, 100 )
      x_onset = np.ones( 100 ) * onset
      y_derivative = b_coeff * x_data**alpha 
    
      # Plot the best fit function 
      y_fit = sample_function( x, 
                             a = all_opt_params[ best_sample, 0 ], 
                             b = all_opt_params[ best_sample, 1 ], 
                             c = all_opt_params[ best_sample, 2 ], 
                             d = all_opt_params[ best_sample, 3 ], 
                             e = all_opt_params[ best_sample, 4 ] )
    elif function_type == "constant": 
      # Plot the best fit function 
      y_fit = sample_constant( x, 
                             a = all_opt_params[ best_sample, 0 ], 
                             b = all_opt_params[ best_sample, 1 ], 
                             c = all_opt_params[ best_sample, 2 ], 
                             d = all_opt_params[ best_sample, 3 ], 
                             e = all_opt_params[ best_sample, 4 ] )

    plt.xscale( 'log' )
    plt.yscale( 'log' )
    if same_scale:
      yrange = np.max( y_data ) / np.min( y_data )
      plt.xlim( np.min( x_data ), yrange * np.min( x_data ) )
    else:
      plt.xlim( np.min( x_data ), np.max( x_data ))

    plt.ylim( np.min( y_data ), np.max( y_data ) )
    plt.plot( x_data, y_data, 'g.', ms = 1.0, label = 'Exp.' )  # Experimental data
    plt.plot( x, y_fit, 'b-', ms = 0.6, label = 'Best fit' )   # Fitted curve
    if function_type == "multivalent":
      plt.plot( x_data, y_derivative, 'r-.', ms = 0.3, label = 'Tangent' ) # Plot of the tangent in the log-log plot
      plt.plot( x, y_mid, 'k-', ms = 0.3, label = 'Mid-point' )   # Horizontal line at the mid point in the log-log plot 
      plt.plot( x_onset, y_onset, 'k--', ms = 0.3, label = "Onset" ) # A vertical line at the onset
      plt.plot( x, y_min, 'k-', ms = 0.3, label = 'Minimum' )     # Horizontal line at the minimum (extrapolated at x = 0 )

    plt.legend()
    plt.savefig( graph_name )
    plt.close()

    return

def save_fit_data( all_opt_params, best_sample, error_onset, average_onset, onset_coeffs, 
                   minimum_loss, 
                   function_type = "multivalent",
		   output = 'output.txt', verbose = False,
                   save = True ):
    ''''Take result of fitting_calculate average and save it in a file'''
    
    if function_type == "multivalent":
      onset = onset_coeffs[ 'onset' ] 
      b_coeff = onset_coeffs[ 'b_coeff'] 
      alpha = onset_coeffs[ 'alpha'] 
      y0 = onset_coeffs[ 'y0'] 
      y_value_min = onset_coeffs[ 'y_value_min'] 
    
    # Print the optimized parameters
    if verbose:
      print("Parameter for best solution:", all_opt_params[ best_sample ] )

    with open( output, 'w+' ) as my_f:
      par = [ "A", "B", "C", "D", "E" ]
      if function_type == "multivalent":
        my_f.write( "Intensity approximated with fitting function A + B * E * ( 1.0 + C * x )^D / [ 1.0 + E * ( 1.0 + C * x )^D ] \n" )
        for name, val in zip( par, all_opt_params[ best_sample ] ):
          my_f.write( f"{name}  {val} \n" ) 
        #print("Optimized Parameters:", result.x)
        my_f.write( f'Logarithmic Derivative at midpoint (superselectivity parameter): {alpha} \n' )
        my_f.write( f'Onset for best solution at x = {onset} \n' )
        my_f.write( f"Residual on the logarithm: {minimum_loss} \n" ) 
      elif function_type == "constant":
        my_f.write( "Intensity approximated with constant fitting function f(x) = A \n" )
        my_f.write( f"A {all_opt_params[ best_sample ][ 0 ]} \n" ) 
        my_f.write( f"Residual on the logarithm: {minimum_loss} \n" )

    return 

def f_statistic_from_fitted_functions(func1, params1, func2, params2, x_data, y_data, 
                                      logarithmic = True, verbose = False ):
    """
    Calculates the F-statistic to compare two fitted functions.

    Args:
        func1: First fitted function (callable).
        params1: Parameters of the first function.
        func2: Second fitted function (callable).
        params2: Parameters of the second function.
        x_data: Array of independent variable values.
        y_data: Array of observed dependent variable values.

    Returns:
        F-statistic value (float).
    """

    if logarithmic:
      # Calculate predicted values and residuals on the logarithms of the values
      log_y_pred1 = np.log( func1(x_data, **params1) )
      log_y_pred2 = np.log( func2(x_data, **params2) )
      residuals1 = np.log( y_data ) - log_y_pred1
      residuals2 = np.log( y_data ) - log_y_pred2
    else:
      # Calculate predicted values and residuals for each model
      y_pred1 = func1(x_data, *params1) 
      y_pred2 = func2(x_data, *params2)
      residuals1 = y_data - y_pred1
      residuals2 = y_data - y_pred2

    # Calculate the sum of squares for each model
    rss1 = np.sum(residuals1**2)
    rss2 = np.sum(residuals2**2)

    assert len( params2 ) > len( params1 ), AssertionError( "Model 1 should be the simpler (less parameters)")
    assert rss1 > rss2, AssertionError( """Model 1 should have higher variance, it is the simpler model.
                                         MC procedure might not find best fit, increase mc_runs and n_hopping.
                                         If problem persists, it signals the more complex model is not
                                         better than fitting with a constant = no signal""")

    # Determine degrees of freedom (assuming func2 has more parameters)
    n = len(x_data) 
    p2 = len(params2)  # N of parameters for more complex model
    p1 = len(params1)  # N of parameters for simpler model
   
    n_degree_freedom = n - p2 # Number of degree of freedom for complex model 
    delta_para = p2 - p1  # Additional degrees of freedom for full model

    # Calculate the F-statistic
    f_stat = ( (rss1 - rss2) / rss2 ) * n_degree_freedom / delta_para

    if verbose:
      print( f"rss1 {rss1}" ) 
      print( f"rss2 {rss2}" ) 
      print( f"p1 {p1}" ) 
      print( f"p2 {p2}" ) 
      print( f"n degree of freedom {n_degree_freedom}" ) 
      print( f"delta_para (4 expected) {delta_para}" ) 
      print( f"FINAL f_stat {f_stat}" ) 
    
    return f_stat, delta_para, n_degree_freedom

def calculate_p_value(f_statistic, dfn, dfd ):
    """Calculates the p-value for a given F-statistic.

    Args:
        f_statistic: The F-statistic value.
        dfn: Degrees of freedom for the numerator. 
             this is the difference in the number of parameters between the full vs simplified model.
        dfd: Degrees of freedom for the denominatori. When using F-statistics for model comparison,
             this is the difference between the number of data points and the number of parameters 
             in the full model (i.e., how many degrees of freedom are left)

    Returns:
        The p-value associated with the F-statistic.
    """

    p_value = 1 - st.f.cdf(f_statistic, dfn, dfd)  # Using survival function (sf) for better accuracy
    return p_value

def compare_models_using_p_value( func1, params1, func2, params2, x_data, y_data, p_min, 
                                  logarithmic = True, verbose = False ):
    """Takes two fitted functions as input, the data from which they were fitted and a minimum p-value for significance.
    Then calculates the f-statistics and determined the p-value. If p-value is less than the set minimum, return
    test_passed = True, along with the p_value
    """
    
    try:
      #assert len( params1 ) > len( params2 )
      assert len( params1 ) < len( params2 )
      #The following is only executed if previous assertion is True
      f1 = func1
      p1 = params1
      f2 = func2
      p2 = params2
    except AssertionError:
      print( "PROBLEM: complex  model should have more parameters, not less...: " )
      raise AssertionError
    
    if verbose:
      print( f"parameters {p1}" )  
      print( f"function 1 (constant expected) {f1}" )  
      print( f"parameters model 2 {p2}" )  
      print( f"function 2 (multivalent expected) {f2}" )  
      print( f"xdata {x_data[::100]}" )  
      print( f"ydata {y_data[::100]}" )  
    #First calculate f_statistics:
    f_stat, dfn, dfd  = f_statistic_from_fitted_functions(f1, p1, f2, p2, x_data, y_data, 
                        logarithmic = True, verbose = verbose )
    p_value = calculate_p_value(f_stat, dfn, dfd )
   
    if verbose:
      print( f"fstat {f_stat}" )
      print( f"dfn {dfn}" )
      print( f"dfd {dfd}" )
      print( f"p-value {p_value}" )

    test_passed = p_value < p_min

    return f_stat, p_value, test_passed

def no_signal( x, a ):
    '''Fitting function of the fluorescence vs B-cell receptors assuming no signal =
    no binding at all is present (only fit baseline).
    This function is the same as sample_constant, just built to take 1 parameter only.
    Note that sample_constant takes 5 parameters in input but 4 (b-e) are irrelevant
    it was only used to have an homogeneous way of passing data.
    To be merged with no_signal in the future.'''

    return np.ones( len(x) ) * a 

if __name__ == "__main__":
  dfn = 4
  dfd = 3260

  f_stat_tot = np.linspace( 0.0, 300, 100 )
  
  for i in f_stat_tot:
    print( f"{i}" )
    pval = calculate_p_value(i, dfn, dfd )
    print( f"fstat {i}, p value {pval}" )
