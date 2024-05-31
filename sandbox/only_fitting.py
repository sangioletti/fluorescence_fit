from theory import calculate_onset, find_best_fit, graphical_fitting 

# Change the line belows to import data from the correct excel files and their internal pages
my_file = "Binding_curve_values.xlsx"
x_name = 'X - IgM'
y_name = 'Y - binding signal'
sheet_names = [ 'P2 ancestor - IgM cells only', 'P3 ancestor - IgM cells only' ]

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

  graphical_fitting( my_file, name, x_name, y_name, bounds, initial_guess,
                   function_type = "constant", onset_fitting = False,
                   output = 'output.txt', graph_name = 'LogLog.pdf', verbose = False,
                   same_scale = False, mc_runs = 8, n_hopping = 2000, T_hopping = 3 )
