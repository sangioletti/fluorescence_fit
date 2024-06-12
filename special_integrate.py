import sys
sys.path.append( "/Users/sangiole/Github-repos/fluorescence_fit/" )
import numpy as np
import scipy as sp
import itertools
from theory_model_comparison import gaussian_likelyhood


def generate_grid( bounds, n_points ):
  my_list = []
  for i in range( len( bounds ) ):
    #print( bounds[ i ][ 0 ] )
    #print( bounds[ i ][ 1 ] )
    my_list.append(  np.linspace( start = bounds[ i ][ 0 ], stop = bounds[ i ][ 1 ], num = n_points[ i ] ) )
  all_combinations = list( itertools.product( *my_list ) )
  all_combinations = np.array( all_combinations )

  return all_combinations

def average_likelyhood_over_grid( model, sigma2, par, x_data, y_data, bounds, n_points, n_count = 10**4, verbose = False ):
  parameters = generate_grid( bounds, n_points )
  res = np.zeros( len( parameters ) )
  for i, value in enumerate( parameters ):
    par = {} 
    par[ "a" ] = value[ 0 ] 
    par[ "b" ] = value[ 1 ] 
    par[ "c" ] = value[ 2 ] 
    par[ "d" ] = value[ 3 ] 
    par[ "e" ] = value[ 4 ] 
    res[ i ] = gaussian_likelyhood( model, sigma2, par, x_data, y_data, logarithmic = True, verbose = False )
    if i % n_count == 0 and verbose:
      print( f"Point {i}, par = {par}" )
  
  return np.mean( res )

if __name__ == "__main__":
  bounds = ( (0, 2), (0, 2) )
  n_points = ( 3, 3 )
  print( generate_grid( bounds, n_points ) )
  
  bounds = ( (0, 2), (0, 2) )
  n_points = ( 3, 3, 3 )
  aa = generate_grid( bounds, n_points ) 
  #print( aa )
  #print( len( aa ) )

  def model( x, a, b, c, d, e ):
    p1 = ( 1.0 + c*x )**d
    return a + b * e * p1 / ( 1.0 + e * p1 )
  
  def model2( x, a, b, c, d, e ):
    return a * x + b + c + d + e

  n_data = [ 10**3, 10**4, 10**5, 10**6 ]
  sigma2 = 0.5 
  for nn in n_data:
    x_data = np.zeros( nn )
    par = {}
    par[ "a" ] = 3.0
    par[ "b" ] = 1.0
    par[ "c" ] = 1.0
    par[ "d" ] = 1.0
    par[ "e" ] = 1.0
    y_data = par["a" ] * x_data + par["b"]
    ba = (0, 4 )
    bb = ( 0, 2 )
    bounds = ( ba, bb, bb, bb, bb )
    n_points = ( 10, 10, 10, 10, 10 )
    result = average_likelyhood_over_grid( model2, sigma2, par, x_data, y_data, bounds, n_points )
    print( result )
    
