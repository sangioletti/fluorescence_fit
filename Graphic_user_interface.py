import tkinter as tk
from fitting import run_fitting

# Wrapper function to get input from GUI and run the actual function
def run_function():
    my_file_value = my_file.get()
    my_sheet_value = sheet_name.get()
    x_name_value = x_name.get()
    y_name_value = y_name.get()

    # Use eval to convert the string to a tuple
    bounds_A_value = eval(bounds_A.get())  
    guess_A_value = eval(guess_A.get())  
    bounds_B_value = eval(bounds_B.get())
    guess_B_value = eval(guess_B.get())  
    bounds_C_value = eval(bounds_C.get())
    guess_C_value = eval(guess_C.get())  
    bounds_D_value = eval(bounds_D.get())
    guess_D_value = eval(guess_D.get())  
    bounds_E_value = eval(bounds_E.get())
    guess_E_value = eval(guess_E.get()) 

    mc_runs_value = eval( mc_runs.get() )
    n_hop_value = eval( n_hop.get() )
    temp_value = eval( temp.get() )
    verbosity_value = eval( verbosity.get() )

    # Call the actual fitting function
    result = run_fitting(
        my_file=my_file_value,
        sheet_name = my_sheet_value,
        x_name = x_name_value,
        y_name = y_name_value,
        bounds_A = bounds_A_value,
        guess_A = guess_A_value,
        bounds_B = bounds_B_value,
        guess_B = guess_B_value,
        bounds_C = bounds_C_value,
        guess_C = guess_C_value,
        bounds_D = bounds_D_value,
        guess_D = guess_D_value,
        bounds_E = bounds_E_value,
        guess_E = guess_E_value,
        mc_runs = mc_runs_value,
        n_hopping = n_hop_value,
        T_hopping = temp_value,
        verbose = verbosity_value,
    )

    return result

root = tk.Tk()
root.title("Graphic User Interface - Leave None if unknown")

# Define labels and entry widgets for parameters
tk.Label(root, text="Name of Excel file").pack()
my_file = tk.Entry(root)
my_file.pack()
my_file.insert(0, "Binding_curve_values.xlsx")  # Default value

tk.Label(root, text="Name of Excel sheet").pack()
sheet_name = tk.Entry(root)
sheet_name.pack()
sheet_name.insert(0, "P2 ancestor - IgM cells only")  # Default value

tk.Label(root, text="Name of X values column").pack()
x_name = tk.Entry(root)
x_name.pack()
x_name.insert(0, "X - IgM")  # Default value

tk.Label(root, text="Name of Y values column").pack()
y_name = tk.Entry(root)
y_name.pack()
y_name.insert(0, "Y - binding signal")  # Default value

tk.Label(root, text="Bounds for A value").pack()
bounds_A = tk.Entry(root)
bounds_A.pack()
bounds_A.insert(0, "(0, 10**4)")  # Default value

tk.Label(root, text="Initial gues for A value").pack()
guess_A = tk.Entry(root)
guess_A.pack()
guess_A.insert(0, "None")  # Default value
    
tk.Label(root, text="Bounds for B value").pack()
bounds_B = tk.Entry(root)
bounds_B.pack()
bounds_B.insert(0, "(0, 2 * 10**4)")  # Default value

tk.Label(root, text="Initial gues for B value").pack()
guess_B = tk.Entry(root)
guess_B.pack()
guess_B.insert(0, "None")  # Default value

tk.Label(root, text="Bounds for C value").pack()
bounds_C = tk.Entry(root)
bounds_C.pack()
bounds_C.insert(0, "(10**(-9), 1.0)" )  # Default value

tk.Label(root, text="Initial gues for C value").pack()
guess_C = tk.Entry(root)
guess_C.pack()
guess_C.insert(0, "None")  # Default value

tk.Label(root, text="Bounds for D value").pack()
bounds_D = tk.Entry(root)
bounds_D.pack()
bounds_D.insert(0, "(1, 25)")  # Default value

tk.Label(root, text="Initial guess for D value").pack()
guess_D = tk.Entry(root)
guess_D.pack()
guess_D.insert(0, "None")  # Default value

tk.Label(root, text="Bounds for E value").pack()
bounds_E = tk.Entry(root)
bounds_E.pack()
bounds_E.insert(0, "(10**(-9), 1)")  # Default value

tk.Label(root, text="Initial guess for E value").pack()
guess_E = tk.Entry(root)
guess_E.pack()
guess_E.insert(0, "None")  # Default value

tk.Label(root, text="Number of independent Monte Carlo runs").pack()
mc_runs = tk.Entry(root)
mc_runs.pack()
mc_runs.insert(0, "None")  # Default value

tk.Label(root, text="Number of hopping steps x Monte Carlo run").pack()
n_hop = tk.Entry(root)
n_hop.pack()
n_hop.insert(0, "None")  # Default value

tk.Label(root, text="Temperature for Monte Carlo  basin hopping").pack()
temp = tk.Entry(root)
temp.pack()
temp.insert(0, "None")  # Default value

tk.Label(root, text="Maximum verbosity of output?").pack()
verbosity = tk.Entry(root)
verbosity.pack()
verbosity.insert(0, "False")  # Default value
    
# Button to run the function
run_button = tk.Button(root, text="Run Function", command=run_function)
run_button.pack()

# Start the GUI event loop
root.mainloop()
