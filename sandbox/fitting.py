from theory import calculate_onset 


def run_fitting(
    my_file="Binding_curve_values.xlsx",
    x_name="X - IgM",
    y_name="Y - binding signal",
    #sheet_names=["P2 ancestor - IgM cells only", "P3 ancestor - IgM cells only"],
    sheet_name="P1 ancestor - IgM cells only",
    bounds_A=(0, 1000),
    guess_A = None,
    bounds_B=(1, 2 * 10**4),
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
    verbose = False,
):
    """
    my_file : str = Name of the excel file containing the data
    x_name : str =  Name of the column containing the X values ( IgM number )
    y_name : str =  Name of the column containing the Y values ( Fluorescence )
    sheet_names : str = Name of the specific sheet containing the data

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

    # print( f"""This code takes the data from {my_file}, specifically, from the sheets {sheet_names}
    # and uses the columns {x_name} as X and {y_name} as Y data, then tries to fit
    # the data using a formula related to multivalent binding (details in the DETAILS.txt
    # file)""" )

    # for name in sheet_names:
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
    print( f"Maximum verbosity of output? {verbose}")

    calculate_onset(
        my_file,
        name,
        x_name,
        y_name,
        bounds,
        initial_guess,
        output=output_file,
        graph_name=output_graph,
        verbose=verbose,
        same_scale=False,
        mc_runs=mc_runs, 
        n_hopping=n_hopping, 
        T_hopping=T_hopping,
    )

    return
