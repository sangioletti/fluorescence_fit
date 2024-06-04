import numpy as np
import matplotlib.pyplot as plt
#from theory import sample_function, sample_constant, sample_log, sample_log_constan

def no_signal( x, a ):
    return a 

def bayes_model_vs_uniform( model, model_variance,model_parameters, parameters_bounds,  
                            constant_variance, constant_parameters, constant_bounds, 
                            x_data, y_data, prior = 'uniform', verbose = False )

  #Calculate and then compare the (unnormalised) probability that the model is correct, given the data. Because normalisation constant
  #is the same = P( data ) prior probability of the data, the ratio of the unnormalised probability defines the most probable model
  model_posterior = model_evidence( model, model_variance, model_parameters, parameters_bounds, x_data, y_data, prior = 'uniform', verbose = verbose )
  uniform_posterior = model_evidence( no_signal, constant_variance, constant_parameters, constant_bounds, x_data, y_data, prior = 'uniform', verbose = verbose )
  
  bayes_coeff = model_posterior / uniform_posterior
  model_valid = bayes_coeff > 1

  return bayes_coeff, model_valid

def gaussian_likelyhood( model, sigma2, parameters, x_data, y_data, verbose = False ):
  '''Calculate the likelyhood of the data given a model and its parameters'''
  y_model = model( x_data, **parameters )
  log_prefactor = len( x_data ) * np.log( 1.0 / np.sqrt( 2 * np.pi * sigma2 ) )
  dy2 = -( ( y_model - y_data )**2 / ( 2 * sigma2 ) )
  log_likelyhood = log_prefactor + dy2.sum() 
  if verbose:
    print( f"Parameters {parameters}" )
    print( f"x in input before likelyhood {x_data}" ) 
    print( f"y_model {y_model}" )
    print( f"y_data {y_data}" )
    print( f"dy2 {dy2}" )
    print( f"log of the likelyhood {log_likelyhood}" )
  
  return np.exp( log_likelyhood )

def model_evidence( model, sigma2, model_parameters, parameters_bounds, x_data, y_data, prior = 'uniform', verbose = False ):
  assert prior == "uniform", AssertionError( "Only flat priors implemented, check function to generate random sample" )
  result = mc_integral( model, sigma2, model_parameters, bounds, x_data, y_data, 
                 eps = 10**(-5), n_blocks = 10, n_steps = 10**4, max_step = 10**7,
                 verbose = verbose 
                )
  return result

def mc_integral( model, sigma2, model_parameters, bounds, x_data, y_data, 
                 eps = 10**(-5), n_blocks = 2, n_steps = 20, max_step = 10**4, verbose = False 
                ):
  rng = np.random.default_rng()  # Create a random number generator

  hyper_volume = 1.0
  for i in range( len( bounds ) ):
    hyper_volume *= np.max( bounds[ i ] ) - np.min( bounds[ i ] )
  assert hyper_volume > 0.0, AssertionError( f"Integration space should > 0.0, value: {hyper_volume}" )
  converged = False 
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
                                                                               rng.uniform
                                                                              )
    count += 1
    mean_over_steps = np.mean( estimate, axis = 1 )
    mean_over_blocks = np.mean( mean_over_steps ) 
    var = np.var( mean_over_blocks , dtype=np.float32 )
    rel_error = np.sqrt( var ) / mean_over_blocks
    if verbose: 
      print( f"mean_over_steps {mean_over_steps}" )
      print( f"mean_over_blocks {mean_over_blocks}" )
      print( f"Mean and variance of the function {mean_over_blocks} {var}" )
      print( f"Relative error = { rel_error }, accepted max error: {eps}" )
    if rel_error < eps:
      converged = True
      break
    if count >= max_count:
      break

  if converged:
    result = mean_over_blocks * hyper_volume
  else:
    result = None 

  return result 

def generate_random_sample( model, sigma2, model_parameters, bounds, x_data, y_data, rng ):
    #Generate a random sample for the model parameters, assuming a flat/uniform distribution
    #within the hyper-cube defined by bounds. To be used later for MC integration
    parameters = {}
    for i, value in enumerate( model_parameters.keys() ):
      #Pick a value for the parameter within the bounds
      parameters[ value ] = rng( low = np.min( bounds[ i ] ), high = np.max( bounds[ i ] ), size = 1 )
    sample = gaussian_likelyhood( model, sigma2, parameters, x_data, y_data )

    #Now we are assuming here uniform priors, in which case you have that p( theta ) = prod_i 1/L_i
    #where is the length of the interval for parameter "i"
    for i in range( len( bounds ) ):
      L = np.max( bounds[ i ] ) - np.min( bounds[ i ] )
      sample *= 1.0 / L

    return sample

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
  
    first_estimate = gaussian_likelyhood( model, noise_scale, parameters, x_data, y_data, verbose = verbose )
    print( f"Gaussian likelyhood of data with exact parameters {first_estimate}" )
    evidence = model_evidence( model, sigma2 = noise_scale, model_parameters = parameters, 
             parameters_bounds = bounds, 
               x_data = x_data, 
               y_data = y_data 
               )
    print( f"Evidence for model1: {evidence}" )

    first_estimate = gaussian_likelyhood( model2, noise_scale, parameters, x_data, y_data, verbose = verbose )
    print( f"Gaussian likelyhood of data with exact parameters, CONSTANT MODEL {first_estimate}" )
    evidence = model_evidence( model2, sigma2 = noise_scale, model_parameters = parameters, 
               parameters_bounds = bounds, 
               x_data = x_data, 
               y_data = y_data 
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
