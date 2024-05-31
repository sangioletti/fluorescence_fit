import numpy as np
import matplotlib.pyplot as plt
#from theory import sample_function, sample_constant, sample_log, sample_log_constant

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

def model_evidence( model, sigma2, model_parameters, parameters_bounds, x_data, y_data, prior = 'uniform' ):
  assert prior == "uniform", AssertionError( "Only flat priors implemented, check function to generate random sample" )
  result = mc_integral( model, sigma2, model_parameters, bounds, x_data, y_data, 
                 eps = 10**(-5), n_blocks = 10, n_steps = 10**4, max_step = 10**7 
                )
  return result

def mc_integral( model, sigma2, model_parameters, bounds, x_data, y_data, 
                 eps = 10**(-5), n_blocks = 2, n_steps = 20, max_step = 10**4 
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
    print( f"mean_over_steps {mean_over_steps}" )
    mean_over_blocks = np.mean( mean_over_steps ) 
    print( f"mean_over_blocks {mean_over_blocks}" )
    var = np.var( mean_over_blocks , dtype=np.float32 )
    rel_error = np.sqrt( var ) / mean_over_blocks 
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
  noise_scale = 0.1
  noise = rng.normal( loc = 0.0, scale = noise_scale, size = n_points ) 
  print( f"X =  {x_data}" ) 
  y_data = model( x_data, **parameters )
  print( f"Data BEFORE noise: {y_data}" ) 
  y_data += noise
  print( f"Data AFTER noise: {y_data}" ) 

  #Make a plot just to be able to check...
  plt.xlabel( 'x' )
  plt.ylabel( 'y' )
  plt.title( 'sin( x ) + 1 with noise' )
  plt.plot( x_data, y_data, 'r-' ) 
  plt.savefig( "Check.pdf" )
  plt.close() 
 
  bounds = [ ( -1, 2 ), ( -2, 2 ) ] 
  
  first_estimate = gaussian_likelyhood( model, noise_scale, parameters, x_data, y_data, verbose = True )
  print( f"First estimate {first_estimate}" )
  evidence = model_evidence( model, sigma2 = noise_scale, model_parameters = parameters, 
               parameters_bounds = bounds, 
               x_data = x_data, 
               y_data = y_data 
               )
  print( f"Evidence for the model is {evidence}" )
  evidence = model_evidence( model2, sigma2 = noise_scale, model_parameters = parameters, 
               parameters_bounds = bounds, 
               x_data = x_data, 
               y_data = y_data 
               )
  print( f"Evidence for model2 is {evidence}" )
