import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF
from theory import sample_log

def calculate MLE( param, x_data, y_data, initial_guess, bounds, logarithmic = True,
                   kernel = RBF(), 
                   alpha_gp =1e-10, # Small alpha for likelihood
                   n_restarts_optimizer_gp = 10 ) 
 
  if logarithmic:
    y_data = np.log( y_data )
    
  # GP for Likelihood
  kernel = kernel # Choose an appropriate kernel for your problem
  gp = GaussianProcessRegressor( kernel = kernel, 
                                 alpha=1e-10, 
                                 n_restarts_optimizer = 10 ) # Small alpha for likelihood

  # Negative Log-Likelihood Function
  def neg_log_likelihood( param ):
    gp.fit( x_data, y_data - model(x_data, param) ) # GP models residuals
    return -gp.log_marginal_likelihood( gp.kernel_.theta ) # Negative log-likelihood

  # MLE Optimization
  result = minimize(neg_log_likelihood, initial_guess, bounds = bounds, method='L-BFGS-B')
  estimated_parameter = result.x

  print("Estimated Parameters:", estimated_parameters)

  return estimated_parameter

def AIC( log_likelyhood, param ):
  k = len( param )
  return 2 * k - 2 * log_likelyhood 
