#%%
import numpy as np
import pandas as pd
from gurobipy import Model, GRB
from input_data_and_functions_basecase_MC import *
from math import ceil


max_ntc = 1000
Number_of_days = 30*6
Number_of_hours = 24*Number_of_days

# Input
use_NP = False
use_PTC = False
use_line = False
Basecase = True

GSK = 'flat'

Basecase_fixed_results_path = f'D-1_CGM_results_{GSK}/' + str(Number_of_hours) + '_hours'
if not os.path.exists(Basecase_fixed_results_path):
    os.makedirs(Basecase_fixed_results_path)
    
if Basecase:
    generation = pd.read_csv(f'D-1_MC_results_{GSK}\\{Number_of_hours}_hours\\d_1_gen.csv', index_col=0)
    curtailment = pd.read_csv(f'D-1_MC_results_{GSK}\\{Number_of_hours}_hours\\d_1_curt.csv', index_col=0)
    
T = generation.index
# Initialize arrays for various variables
d_2_curt = np.zeros((len(T), len(N)), dtype=float)  # Curtailment
d_2_delta = np.zeros((len(T), len(N)), dtype=float)  # Delta values
d_2_nod_inj = np.zeros((len(T), len(N)), dtype=float)  # Nodal injections
d_2_line_f = np.zeros((len(T), len(L)), dtype=float)  # Line flows
d_2_gen = np.zeros((len(T), len(P)), dtype=float)  # Generation
d_2_gen_costs = np.zeros((len(T), len(Z)), dtype=float)  # Generation costs
d_2_curt_costs = np.zeros((len(T), len(Z)), dtype=float)  # Curtailment costs
d_2_nodal_price = np.zeros((len(T), len(N)), dtype=float)  # Nodal prices
d_2_np = np.zeros((len(T), len(Z_FBMC)), dtype=float)  # Nodal price for FBMC zones
d_2_export = np.zeros((len(T), len(Z_not_in_FBMC)), dtype=float)  # Exports to non-FBMC zones
d_2_objective_per_hour = np.zeros((len(T)), dtype=float)  # Objective per hour
d_1_line_f_slack_pos = np.zeros((len(T), len(L)), dtype=float)  # Line flow slack variables (positive)
d_1_line_f_slack_neg = np.zeros((len(T), len(L)), dtype=float)  # Line flow slack variables (negative)

# Find the maximum marginal cost
max_mc = find_maximum_mc()

# Curtailment cost is the ceiling of the maximum marginal cost
cost_curt = ceil(max_mc)

# Define hours per horizon and days foresight
hours_per_horizon = 24*7 #4 * 168  # Example: 4 weeks of hourly intervals
days_foresight = 1  # Days of foresight

# Print outputs for verification
print(f"Maximum marginal cost: {max_mc}")
print(f"Curtailment cost: {cost_curt}")
print(f"Hours per horizon: {hours_per_horizon}")
print(f"Days foresight: {days_foresight}")


# Loop through horizons
for horizon in range(1, ceil(len(T) / hours_per_horizon) + 1):  # Python range is inclusive of start and exclusive of end
    print(f"Horizon: {horizon}/{ceil(len(T) / hours_per_horizon)}")
    
    # Calculate the subset of time steps (Tsub)
    start_index = (horizon - 1) * hours_per_horizon  # Use zero-based indexing
    end_index = min(horizon * hours_per_horizon, len(T))  # Ensure we do not exceed the length of T
    
    # Extract the actual time steps from T based on the calculated indices
    Tsub = T[start_index:end_index]  # Use slicing to get the subset of T

    # Print or use Tsub
    print(f"Tsub for Horizon {horizon}: {Tsub.tolist()}")  
    
    # Initialize the Gurobi model
    m = Model("Optimization_Model")

    # Create upper bound dictionary for CURT
    ub_CURT = {(t, n): get_renew(t, n) for t in Tsub for n in N}

    # Add CURT variables with precomputed upper bounds
    CURT = m.addVars(
        [(t, n) for t in Tsub for n in N],
        lb=0,
        ub={(t, n): ub_CURT[t, n] for t in Tsub for n in N},  # Explicitly reference the dictionary
        name="CURT"
    )
    DELTA = m.addVars(
        [(t, n) for t in Tsub for n in N], 
        lb=-GRB.INFINITY, 
        ub=GRB.INFINITY, 
        name="DELTA"
    )
    NOD_INJ = m.addVars(
        [(t, n) for t in Tsub for n in N], 
        lb=-GRB.INFINITY, 
        ub=GRB.INFINITY, 
        name="NOD_INJ"
    )
    LINE_F = m.addVars(
        [(t, l) for t in Tsub for l in L], 
        lb=-GRB.INFINITY, 
        ub=GRB.INFINITY, 
        name="LINE_F"
    )
        # Create upper bound dictionary for GEN
    ub_GEN = {(t, p): get_gen_up(p) for t in Tsub for p in P}

    # Add GEN variables with precomputed upper bounds
    GEN = m.addVars(
        [(t, p) for t in Tsub for p in P],
        lb=0,
        ub={(t, p): ub_GEN[t, p] for t in Tsub for p in P},  # Explicitly index ub_GEN
        name="GEN"
    )
    GEN_COSTS = m.addVars(
        [(t, z) for t in Tsub for z in Z], 
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY, 
        name="GEN_COSTS"
    )
    CURT_COSTS = m.addVars(
        [(t, z) for t in Tsub for z in Z], 
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY, 
        name="CURT_COSTS"
    )
    NP = m.addVars(
        [(t, z) for t in Tsub for z in Z_FBMC], 
        lb=-GRB.INFINITY, 
        ub=GRB.INFINITY, 
        name="NP"
    )
    EXPORT = m.addVars(
        [(t, z) for t in Tsub for z in Z_not_in_FBMC], 
        lb=-max_ntc,
        ub=max_ntc,
        name="EXPORT"
    )

    OBJECTIVE_PER_HOUR = m.addVars(
        [(t) for t in Tsub],
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="OBJECTIVE_PER_HOUR"
    )
    
    # Define objective function
    m.setObjective(
        sum(GEN[t, p] * get_mc(p) for t in Tsub for p in P) +
        sum(CURT[t, n] * cost_curt*1.1 for t in Tsub for n in N),
        GRB.MINIMIZE
    )

    print("Variables and objective function set up.")
 
    for t in Tsub:
        m.addConstr(
            OBJECTIVE_PER_HOUR[t] == sum(GEN[t, p] * get_mc(p) for p in P) +
            sum(CURT[t, n] * cost_curt for n in N),
            name=f"OBJECTIVE_PER_HOUR_{t}"
        )

    # Define constraints
    # Add constraints for generation costs
    for t in Tsub:
        for z in Z:
            m.addConstr(
                sum(GEN[t, p] * get_mc(p) for p in p_in_z[z]) == GEN_COSTS[t, z],
                name=f"costs_gen_{t}_{z}"
            )
    print("Built constraints costs_gen.")

    # Add constraints for curtailment costs
    for t in Tsub:
        for z in Z:
            m.addConstr(
                sum(CURT[t, n] * cost_curt for n in n_in_z[z]) == CURT_COSTS[t, z],
                name=f"costs_curt_{t}_{z}"
            )
    print("Built constraints costs_curt.")
    
        # Add nodal balance constraints
    for t in Tsub:
        for n in N:
            m.addConstr(
                sum(GEN[t, p] for p in p_at_n[n]) + get_renew(t, n)
                - NOD_INJ[t, n] - CURT[t, n]
                ==
                get_dem(t, n),
                name=f"nodal_balance_{t}_{n}"
            )
    print("Built constraints nodal_balance.")

    # Add export balance constraints for zones outside of FBMC
    for t in Tsub:
        for z in Z_not_in_FBMC:
            m.addConstr(
                EXPORT[t, z] == sum(NOD_INJ[t, n] for n in n_in_z[z]),
                name=f"export_balance_abroad_{t}_{z}"
            )
    print("Export balance outside of FBMC.")
    
    # Add net position constraints for zones within FBMC
    for t in Tsub:
        for z in Z_FBMC:
            # Determine the zones (zz) to consider for exports
            if z in Z_FBMC:
                zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]
            else:
                zz_list = z_to_z[z]

            # Add the constraint
            m.addConstr(
                NP[t, z] == sum(NOD_INJ[t, n] for n in n_in_z[z]) +
                            sum(EXPORT[t, zz] for zz in zz_list),
                name=f"net_positions_{t}_{z}"
            )
    print("Net positions within FBMC.")
    
                
    # Add nodal injection constraints
    for t in Tsub:
        for n in N:
            m.addConstr(
                NOD_INJ[t, n] == sum(B_mat[(n, nn)] * DELTA[t, nn] for nn in N),
                name=f"nodal_injection_{t}_{n}"
            )
    print("Built constraints nodal_injection.")


    # Add line flow constraints
    for t in Tsub:
        for l in L:
            m.addConstr(
                LINE_F[t, l] ==     
                sum(H_mat[(l, n)] * DELTA[t, n] for n in N),
                name=f"line_flow_{t}_{l}"
            )
    print("Built constraints line_flow.")

 ### Out commented to get results, depite of potential line cap violations ###
 


    # Fix the slack node (e.g., DELTA[t, 68] = 0 for all t in Tsub)
    for t in Tsub:
        m.addConstr(DELTA[t, 50] == 0, name=f"fix_slack_node_{t}")
    print("Built constraints FIX SLACK NODE.")

    #Fix generation to the generation from MC
    for t in Tsub:
        for p in P:
            m.addConstr(GEN[t, p] == generation.loc[t, str(p)], name=f"fix_gen_{t}_{p}")
            
    #Fix curtailment to the curtailment from MC
    for t in Tsub:
        for n in N:
            m.addConstr(CURT[t, n] == curtailment.loc[t, str(n)], name=f"fix_curt_{t}_{n}")
        
    # Print confirmation that constraints are done
    print("Constraints done.")

    # Optimize the model
    m.optimize()

    # Check and print the optimization status
    status = m.status
    if status == GRB.OPTIMAL:
        print("Optimization completed successfully. Status: OPTIMAL")
    elif status == GRB.INFEASIBLE:
        print("Model is infeasible. Status: INFEASIBLE")
    elif status == GRB.UNBOUNDED:
        print("Model is unbounded. Status: UNBOUNDED")
    else:
        print(f"Optimization stopped with status: {status}")
        
            # Extract results and store them in the respective numpy arrays
    for t_idx, t in enumerate(Tsub):

        d_2_objective_per_hour[start_index+t_idx] = OBJECTIVE_PER_HOUR[t].X  # Get value of OBJECTIVE_PER_HOUR

        for n_idx, n in enumerate(N):
            d_2_curt[start_index+t_idx, n_idx] = CURT[t, n].X  # Get value of CURT
            d_2_delta[start_index+t_idx, n_idx] = DELTA[t, n].X  # Get value of DELTA
            d_2_nod_inj[start_index+t_idx, n_idx] = NOD_INJ[t, n].X  # Get value of NOD_INJ
            d_2_nodal_price[start_index+t_idx, n_idx] = m.getConstrByName(f"nodal_balance_{t}_{n}").Pi  # Dual price of nodal balance constraint

        for l_idx, l in enumerate(L):
            d_2_line_f[start_index+t_idx, l_idx] = LINE_F[t, l].X  # Get value of LINE_F

        for p_idx, p in enumerate(P):
            d_2_gen[start_index+t_idx, p_idx] = GEN[t, p].X  # Get value of GEN

        for z_idx, z in enumerate(Z):
            d_2_gen_costs[start_index+t_idx, z_idx] = GEN_COSTS[t, z].X  # Get value of GEN_COSTS
            d_2_curt_costs[start_index+t_idx, z_idx] = CURT_COSTS[t, z].X  # Get value of CURT_COSTS

        for z_idx, z in enumerate(Z_FBMC):
            d_2_np[start_index+t_idx, z_idx] = NP[t, z].X  # Get value of NP

        for z_idx, z in enumerate(Z_not_in_FBMC):
            d_2_export[start_index+t_idx, z_idx] = EXPORT[t, z].X  # Get value of EXPORT   
            
    # Clear the model after extracting results
    m.dispose()


# Create DataFrame for CURT
df_curt = pd.DataFrame(d_2_curt, index=T, columns=N)
df_curt.index.name = "Time Step"
df_curt.columns.name = "Node"

# Create DataFrame for DELTA
df_delta = pd.DataFrame(d_2_delta, index=T, columns=N)
df_delta.index.name = "Time Step"
df_delta.columns.name = "Node"

# Create DataFrame for NOD_INJ
df_nod_inj = pd.DataFrame(d_2_nod_inj, index=T, columns=N)
df_nod_inj.index.name = "Time Step"
df_nod_inj.columns.name = "Node"

# Create DataFrame for Nodal Prices
df_nodal_price = pd.DataFrame(d_2_nodal_price, index=T, columns=N)
df_nodal_price.index.name = "Time Step"
df_nodal_price.columns.name = "Node"

# Create DataFrame for LINE_F
df_line_f = pd.DataFrame(d_2_line_f, index=T, columns=L)
df_line_f.index.name = "Time Step"
df_line_f.columns.name = "Line"

# Create DataFrame for GEN
df_gen = pd.DataFrame(d_2_gen, index=T, columns=P)
df_gen.index.name = "Time Step"
df_gen.columns.name = "Generator"

# Create DataFrame for GEN_COSTS
df_gen_costs = pd.DataFrame(d_2_gen_costs, index=T, columns=Z)
df_gen_costs.index.name = "Time Step"
df_gen_costs.columns.name = "Zone"

# Create DataFrame for CURT_COSTS
df_curt_costs = pd.DataFrame(d_2_curt_costs, index=T, columns=Z)
df_curt_costs.index.name = "Time Step"
df_curt_costs.columns.name = "Zone"

# Create DataFrame for NP (Net Position)
df_np = pd.DataFrame(d_2_np, index=T, columns=Z_FBMC)
df_np.index.name = "Time Step"
df_np.columns.name = "Zone (FBMC)"

# Create DataFrame for EXPORT
df_export = pd.DataFrame(d_2_export, index=T, columns=Z_not_in_FBMC)
df_export.index.name = "Time Step"
df_export.columns.name = "Zone (Not in FBMC)"

# Create DataFrame for OBJECTIVE_PER_HOUR
df_objective_per_hour = pd.DataFrame(d_2_objective_per_hour, index=T)
df_objective_per_hour.index.name = "Time Step"
df_objective_per_hour.columns.name = "Objective"


# save df_export to csv in results folder
df_export.to_csv(f'{Basecase_fixed_results_path}\\df_export.csv')
df_np.to_csv(f'{Basecase_fixed_results_path}\\df_np.csv')
df_gen_costs.to_csv(f'{Basecase_fixed_results_path}\\df_gen_costs.csv')
df_gen.to_csv(f'{Basecase_fixed_results_path}\\df_gen.csv')
df_line_f.to_csv(f'{Basecase_fixed_results_path}\\df_line_f.csv')
df_nodal_price.to_csv(f'{Basecase_fixed_results_path}\\df_nodal_price.csv')
df_nod_inj.to_csv(f'{Basecase_fixed_results_path}\\df_nod_inj.csv')
df_delta.to_csv(f'{Basecase_fixed_results_path}\\df_delta.csv')
df_curt.to_csv(f'{Basecase_fixed_results_path}\\df_curt.csv')
df_curt_costs.to_csv(f'{Basecase_fixed_results_path}\\df_curt_costs.csv')
df_objective_per_hour.to_csv(f'{Basecase_fixed_results_path}\\df_objective_per_hour.csv')




overload_df = pd.DataFrame(index=T, columns=L)
for t in T: 
    for l in L:
        if df_line_f.loc[t, l] > get_line_cap(l):
            overload_df.loc[t, l] = df_line_f.loc[t, l] - get_line_cap(l)
        if df_line_f.loc[t, l] < -get_line_cap(l):
            overload_df.loc[t, l] = df_line_f.loc[t, l] + get_line_cap(l)
            

number_of_overloads_per_hour_df = pd.DataFrame(index=T, columns=['Number of overloads'])
for t in T:
    number_of_overloads_per_hour_df.loc[t, 'Number of overloads'] = overload_df.loc[t].count()

number_of_overloads_per_line_df = pd.DataFrame(index=L, columns=['Number of overloads'])

for l in L:
    number_of_overloads_per_line_df.loc[l, 'Number of overloads'] = overload_df[l].count()
#save overload_df to csv in results folder
overload_df.to_csv(f'{Basecase_fixed_results_path}\\overload_df.csv')

# %%