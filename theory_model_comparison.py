import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
#from theory import sample_function, sample_constant, sample_log, sample_log_constan

def no_signal( x, a ):
    return a 

def bayes_model_vs_uniform( data_variance, model, model_parameters, parameters_bounds,  
                            constant_parameters, constant_bounds, 
                            x_data, y_data, prior = 'uniform', integral_on_grid = False, verbose = False, logarithmic = True,
                            eps = 10**(-5), n_blocks = 10, n_steps = 10**4, max_step = 10**7, 
                            ):

  #Calculate and then compare the (unnormalised) probability that the model is correct, given the data. Because normalisation constant
  #is the same = P( data ) prior probability of the data, the ratio of the unnormalised probability defines the most probable model
  model_posterior, error_model = model_evidence( model, data_variance, model_parameters, parameters_bounds, x_data, y_data, prior = 'uniform', 
                 logarithmic = logarithmic, verbose = verbose,
                 eps = eps, n_blocks = n_blocks, n_steps = n_steps, max_step = max_step, integral_on_grid = integral_on_grid )
  uniform_posterior, error_uniform = model_evidence( no_signal, data_variance, constant_parameters, constant_bounds, x_data, y_data, prior = 'uniform',
                 logarithmic = logarithmic,  verbose = verbose, 
                 eps = eps, n_blocks = n_blocks, n_steps = n_steps, max_step = max_step, integral_on_grid = integral_on_grid )
  
  bayes_coeff = model_posterior / uniform_posterior
  model_valid = bayes_coeff > 1

  # Warn in case difference between evidence not significant
  if abs( model_posterior - uniform_posterior ) < abs( error_model ) + abs( error_uniform ):
    raise AssertionError( f"Validity of estimation uncertain, reduce value of eps parameter. Current value is eps = {eps}" ) 

  return bayes_coeff, model_valid

def gaussian_likelyhood( model, sigma2, parameters, x_data, y_data, logarithmic = True, verbose = False ):
  '''Calculate the likelyhood of the data given a model and its parameters.
  If logarithmic == True it assumes that the error sigma2 is the variance obtained when fitting
  the logarithm of the data instead of the data itself. This has nothing to do with using later the 
  log of the likelyhood, which is done simply to handle a larger range of values
  '''
  if logarithmic:
    y_model = np.log( model( x_data, **parameters ) )
    y_data = np.log( y_data )
  
  #This part is useless so I remove it
  #log_prefactor = len( x_data ) * np.log( 1.0 / np.sqrt( 2 * np.pi * sigma2 ) )
  dy2 = -( ( y_model - y_data )**2 / ( 2 * sigma2 ) )
  #log_likelyhood = log_prefactor + dy2.sum() 
  #log_likelyhood = dy2.sum() 
  log_likelyhood = np.mean( dy2 )
  #if verbose:
  print( f"Parameters {parameters}" )
  print( f"x in input before likelyhood {x_data[::20]}" ) 
  print( f"y_data {y_data[::20]}" )
  print( f"y_model {y_model[::20]}" )
  print( f"sigma2 {sigma2}" )
  print( f"dy2 {dy2[::20]}" )
  print( f"log of the likelyhood {log_likelyhood}" )
  
  return np.exp( log_likelyhood )

def model_evidence( model, sigma2, model_parameters, parameters_bounds, x_data, y_data, prior = 'uniform', logarithmic = True, verbose = False,
                    eps = 10**(-5), n_blocks = 10, n_steps = 10**4, max_step = 10**7, integral_on_grid = False ):
  assert prior == "uniform", AssertionError( "Only flat priors implemented, check function to generate random sample" )

  if integral_on_grid:
    result, error = grid_integral( parameters_bounds, model, sigma2, x_data, y_data, logarithmic = logarithmic, prior = prior, eps = eps, verbose = verbose )
  else:
    result, error = mc_integral( model, sigma2, model_parameters, parameters_bounds, x_data, y_data, logarithmic = logarithmic, 
                 eps = eps, n_blocks = n_blocks, n_steps = n_steps, max_step = max_step,
                 verbose = verbose 
                )

  print( f"MODEL EVIDENCE is: {result}" )
  return result, error

def mc_integral( model, sigma2, model_parameters, bounds, x_data, y_data, logarithmic = True, 
                 eps = 10**(-3), n_blocks = 2, n_steps = 20, max_step = 10**4, verbose = False 
                ):
  rng = np.random.default_rng()  # Create a random number generator

  hyper_volume = 1.0
  for i in range( len( bounds ) ):
    hyper_volume *= np.max( bounds[ i ] ) - np.min( bounds[ i ] )
  assert hyper_volume > 0.0, AssertionError( f"Integration space should > 0.0, value: {hyper_volume}" )
  converged = False
  #print( f"n_blocks {n_blocks}, max_step  {max_step}" ) 
  estimate = np.zeros( ( n_blocks, max_step ) )
  max_count = int( max_step / n_steps )
  count = 0
  while True:
    for block in range( n_blocks ):
      for step in range( n_steps ):
          estimate[ block, count * n_steps + step ] += generate_random_sample( model, sigma2, 
                                                                               model_parameters, 
                                                                               bounds, 
                                                                               x_data, y_data,
                                                                               rng.uniform, 
                                                                               logarithmic = logarithmic
                                                                              )

    count += 1
    mean_over_steps = np.mean( estimate, axis = 1 )
    print( f"Mean over steps is: {mean_over_steps}" )
    mean_over_blocks = np.mean( mean_over_steps ) 
    var = np.var( mean_over_blocks , dtype=np.float32 )

    #Needed if likelyhood is too small or will never converge
    if ( mean_over_steps == 0 ).all():
      print( "Very unlikely model" )
      rel_error = 0
      converged = True
      break

    error = np.sqrt( var )
    rel_error = error / mean_over_blocks
    if verbose: 
      print( f"mean_over_steps {mean_over_steps}" )
      print( f"mean_over_blocks {mean_over_blocks}" )
      print( f"Mean and variance of the function {mean_over_blocks} {var}" )
      print( f"Relative error = { rel_error }, accepted max error: {eps}" )
      print( f"Cycle = {count}" )
    if rel_error < eps:
      converged = True
      break
    if count >= max_count:
      break

  if converged:
    result = mean_over_blocks * hyper_volume
  else:
    result = None

  print( f"Convergence achieved = {converged}, mean = {mean_over_blocks}" )
  print( f"Value of the integral (mean*hypervolume): {result}, relative error: {rel_error}" ) 

  return result, error 

def generate_random_sample( model, sigma2, model_parameters, bounds, x_data, y_data, rng, logarithmic = True ):
    #Generate a random sample for the model parameters, assuming a flat/uniform distribution
    #within the hyper-cube defined by bounds. To be used later for MC integration
    parameters = {}
    for i, value in enumerate( model_parameters.keys() ):
      #Pick a value for the parameter within the bounds
      parameters[ value ] = rng( low = np.min( bounds[ i ] ), high = np.max( bounds[ i ] ), size = 1 )
    sample = gaussian_likelyhood( model, sigma2, parameters, x_data, y_data, logarithmic = logarithmic )

    #Now we are assuming here uniform priors, in which case you have that p( theta ) = prod_i 1/L_i
    #where is the length of the interval for parameter "i"
    for i in range( len( bounds ) ):
      L = np.max( bounds[ i ] ) - np.min( bounds[ i ] )
      sample *= 1.0 / L

    return sample

def grid_function( a, b, c, d, e, model, sigma2, x_data, y_data, logarithmic = True, prior = "uniform" ):
    assert prior == "uniform", AssertionError( """This implementation only works if the prior for the parameters
                                               values are uniform, check the math. This is because when the prior are uniform,
                                               the evidence is exactly equal to the mean of the data likelyhood over the
                                               region defining the parameters' domain""" )
    par = {}
    par[ 'a' ] = a
    par[ 'b' ] = b
    par[ 'c' ] = c
    par[ 'd' ] = d
    par[ 'e' ] = e

    data_likelyhood = gaussian_likelyhood( model, sigma2, par, x_data, y_data, logarithmic = logarithmic )

    return data_likelyhood

def grid_integral( bounds, model, sigma2, x_data, y_data, 
                   logarithmic = True, prior = "uniform", eps = 10**(-3), verbose = False ):

    function_args = ( model, sigma2, x_data, y_data, logarithmic, prior )
    bounds = tuple( bounds )
    
    result, error = integrate.nquad( grid_function, bounds, args = function_args, opts = {'epsrel' : eps} )

    return result, error

if __name__ == "__main__":

  verbose = False
  #Test function to see if calculation of model likelyhood works
  rng = np.random.default_rng()   # Create a random number generator

  #First thing to do is to generate noisy data from an exact model
  def model( x, a, b ):
    return np.sin( x * a ) + b
  #First thing to do is to generate noisy data from an exact model
  def model2( x, a, b ):
    return x * a + b

  n_points = 10
  x_data = np.linspace( 0, 2 * np.pi, n_points )
  parameters = { "a":1.0, "b": 1.0 }
  noise_list = [ 0.01, 0.1, 0.2 ]
  my_styles = [ 'r-', 'g-', 'k-' ]

  for noise_scale, style in zip( noise_list, my_styles ):
    print( f"Noise value: {noise_scale}" ) 
    noise = rng.normal( loc = 0.0, scale = noise_scale, size = n_points ) 
    #print( f"X =  {x_data}" ) 
    y_data = model( x_data, **parameters )
    #print( f"Data BEFORE noise: {y_data}" ) 
    y_data += noise
    #print( f"Data AFTER noise: {y_data}" ) 

    #Now calculated evidence for model
    bounds = [ ( -4, 4 ), ( -4, 4 ) ] 
    logarithmic = False
    first_estimate = gaussian_likelyhood( model, noise_scale, parameters, x_data, y_data, verbose = verbose, logarithmic = logarithmic ) 
    print( f"Gaussian likelyhood of data with exact parameters {first_estimate}" )
    evidence = model_evidence( model, sigma2 = noise_scale, model_parameters = parameters, 
             parameters_bounds = bounds, 
               x_data = x_data, 
               y_data = y_data,
               logarithmic = logarithmic 
               )
    print( f"Evidence for model1: {evidence}" )

    first_estimate = gaussian_likelyhood( model2, noise_scale, parameters, x_data, y_data, verbose = verbose, logarithmic = logarithmic )
    print( f"Gaussian likelyhood of data with exact parameters, CONSTANT MODEL {first_estimate}" )
    evidence = model_evidence( model2, sigma2 = noise_scale, model_parameters = parameters, 
               parameters_bounds = bounds, 
               x_data = x_data, 
               y_data = y_data, 
               logarithmic = logarithmic 
               )
    print( f"Evidence for model2: {evidence}" )

    #Make a plot just to be able to check...
    plt.xlabel( 'x' )
    plt.ylabel( 'y' )
    plt.title( 'sin( x ) + 1 with noise' )
    plt.plot( x_data, y_data, style )

  plt.legend() 
  plt.savefig( "Check.pdf" )
  plt.close()

