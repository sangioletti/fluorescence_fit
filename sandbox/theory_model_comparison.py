import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
import scipy
#from theory import sample_function, sample_constant, sample_log, sample_log_constant

def gaussian_likelyhood( model, sigma2, parameters, x_data, y_data ):
  '''Calculate the likelyhood of the data given a model and its parameters'''
  y_model = model( x_data, **parameters )
  log_prefactor = len( x_data ) * np.log( 1.0 / np.sqrt( 2 * np.pi * sigma2 ) )
  dy2 = -( ( y_model - y_data ) / ( 2 * sigma2 ) )**2
  log_likelyhood = log_prefactor + dy2.sum() 
  
  return np.exp( log_likelyhood )

def model_evidence( model, sigma2, model_parameters, parameters_bounds, x_data, y_data ):
  result = mc_integral( model, sigma2, model_parameters, bounds, x_data, y_data, 
                 eps = 10**(-5), n_blocks = 10, n_steps = 10**4, max_step = 10**7 
                )
  return result

def mc_integral( model, sigma2, model_parameters, bounds, x_data, y_data, 
                 eps = 10**(-5), n_blocks = 10, n_steps = 10**4, max_step = 10**7 
                ):
  rng = np.random.default_rng()  # Create a random number generator

  hyper_volume = 1.0
  for i in range( bounds ):
    hyper_volume *= np.max( bounds[ i ] ) - np.min( bounds[ i ] )
  assert hyper_volume > 0.0, AssertionError( f"Integration space should > 0.0, value: {hyper_volume}" )
  error = np.inf
  converged = False 
  estimate = np.zeros( ( block, max_steps ) )
  max_count = int( max_step / n_steps )
  count = 0
  while True:
    for block in range( n_blocks ):
      for step in range( n_steps ):
          estimate[ block, count * n_steps + step ] += generate_random_sample( model, sigma2, 
                                                                               model_parameters, 
                                                                               bounds, 
                                                                               x_data, y_data,
                                                                               rng.uniform() 
                                                                              )
    count += 1
    mean_over_steps = np.mean( estimate, axis = 1 ) 
    mean_over_blocks = np.mean( mean_over_steps ) 
    var = np.var( mean_over_blocks )
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
      parameters[ key ] = rng( low = np.min( bounds[ i ] ), high = np.max( bounds[ i ] ), size = 1 )
    sample = gaussian_likelyhood( model, sigma2, parameters, x_data, y_data )
    return sample

if __name__ == "__main__":
  #Test function to see if calculation of model likelyhood works
  rng = np.random.default_rng()   # Create a random number generator
  loc = 0
  scale = 1

  #First thing to do is to generate noisy data from an exact model
  def model( x, a, b ):
    return np.sin( x * a ) + b

  n_points = 100
  x_data = np.linspace( 0, 2 * np.pi, n_points )
  parameters = { "a":1, "b": 1 }
  noise_scale = 1
  noise = rng.normal( loc = 0.0, scale = noise_scale, size = n_points ) 
  y_data = model( x_data, **parameters ) + noise
  
  parameters = { "a":-1, "b":0 } #The actual values should be irrelevant, only keys are needed 
  parameters_bounds = [ ( -10, 10 ), ( -10, 10 ) ] 
  
  evidence = model_evidence( model, sigma2 = scale, parameters, parameters_bounds, x_data, y_data )
  print( f"Evidence for the model is {evidence}" )
