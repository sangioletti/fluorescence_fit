import numpy as np
import scipy.stats as st
from scipy.stats import f

def f_statistic_from_fitted_functions(func1, params1, func2, params2, x_data, y_data, logarithmic = True ):
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

    # Determine degrees of freedom (assuming func2 has more parameters)
    df1 = len(x_data) - len(params2)  # Degrees of freedom for full model
    df2 = len(params2) - len(params1)  # Additional degrees of freedom for full model

    # Calculate the F-statistic
    f_stat = ((rss1 - rss2) / df2) / (rss2 / df1)
    
    return f_stat

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

def compare_models_using_p_value( func1, params1, func2, params2, x_data, y_data, p_min, logarithmic = True ):
    """Takes two fitted functions as input, the data from which they were fitted and a minimum p-value for significance.
    Then calculates the f-statistics and determined the p-value. If p-value is less than the set minimum, return
    test_passed = True, along with the p_value
    """
    
    try:
      assert len( params1 ) > len( params2 )
      #The following is only executed if previous assertion is True
      f1 = func1
      p1 = params1
      f2 = func2
      p2 = params2
    except AssertionError:
      print( "WARNING: You have inverted the complex (more parameters) vs simple (less parameters) model in the input." )
      print( "I will continue but I will invert their definitions for you. Checking might be better though" )
      f2 = func1
      p2 = params1
      f1 = func2
      p1 = params2
      
    #First calculate f_statistics:
    f_stat = f_statistic_from_fitted_functions(f1, p1, f2, p2, x_data, y_data, logarithmic = True )
    dfn = len( p1 ) - len( p2 ) 
    dfd = len( x_data ) - len( p1 ) 
    p_value = calculate_p_value(f_stat, dfn, dfd )

    test_passed = p_value < p_min

    return p_value, test_passed
