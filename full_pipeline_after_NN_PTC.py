#%%
import numpy as np
import pandas as pd
from gurobipy import Model, GRB
from input_data_and_functions_basecase_MC import *
from math import ceil

max_ntc = 1000
Number_of_days = 30*6
Number_of_hours = 24*Number_of_days

# What should be fixed?
use_NP = False
use_PTC = True

# Find the maximum marginal cost
max_mc = find_maximum_mc()

# Curtailment cost is the ceiling of the maximum marginal cost
cost_curt = ceil(max_mc)

# Define hours per horizon and days foresight
hours_per_horizon = 24*7 #4 * 168  # Example: 4 weeks of hourly intervals
days_foresight = 1  # Days of foresight

#########################
###   FIXED D-2 CGM   ###
###        PTC        ###
#########################

if use_PTC:
    Basecase_fixed_results_path = 'D-2_base_case_fixed_PTC_results/' + str(Number_of_hours) + '_hours'
    if not os.path.exists(Basecase_fixed_results_path):
        os.makedirs(Basecase_fixed_results_path)
    #load predictions
    predictions = pd.read_csv(f'ML_results\\{Number_of_hours}_hours\\predictions_PTC_50_epochs.csv', index_col=0)

PTC_mapping = pd.read_csv(f'D-1_MC_results_flat\\{Number_of_hours}_hours\\PTC_mapping.csv', index_col=0)


# Iterate through each column in the predictions DataFrame
for column in predictions.columns:
    #look up predictions.columns in PTC_mapping['PTC_line'] and get all the lines that are in same row in column PTC_mapping['BranchID']
    Branch_IDs = PTC_mapping[PTC_mapping['PTC'] == column]['BranchID']
    #if Branch_IDs is empty, then 
    if column == '9->10' or column == '55->119' or column == '89->120':
        PTC_line_caps = max_ntc
        predictions[column] = predictions[column].clip(lower=-PTC_line_caps, upper=PTC_line_caps)
    else:
        # Initialize the total PTC line capacities
        PTC_line_caps = 0
        
        # Iterate through each Branch_ID in the Branch_IDs Series
        for Branch_ID in Branch_IDs:
            # Convert Branch_ID string to a list of integers (handles cases like '[63, 209]')
            branch_ids = [int(id.strip()) for id in Branch_ID.strip('[]').split(',')]
            
            # Add the capacities of all branch IDs to the total PTC_line_caps
            PTC_line_caps += sum(get_line_cap(id) for id in branch_ids)
        
        # Clip the values in the column to be within [-PTC_line_caps, PTC_line_caps]
        predictions[column] = predictions[column].clip(
            lower=-PTC_line_caps * (1 - frm),
            upper=PTC_line_caps * (1 - frm)
        )
         
T = predictions.index
# Initialize arrays for various variables
d_2_curt = np.zeros((len(T), len(N)), dtype=float)  # Curtailment
d_2_delta = np.zeros((len(T), len(N)), dtype=float)  # Delta values
d_2_nod_inj = np.zeros((len(T), len(N)), dtype=float)  # Nodal injections
d_2_line_f = np.zeros((len(T), len(L)), dtype=float)  # Line flows
d_2_PTC_slack_pos = np.zeros((len(T), len(predictions.columns)), dtype=float)  # PTC slack variables (positive)
d_2_PTC_slack_neg = np.zeros((len(T), len(predictions.columns)), dtype=float)  # PTC slack variables (negative)  
d_2_gen = np.zeros((len(T), len(P)), dtype=float)  # Generation
d_2_gen_costs = np.zeros((len(T), len(Z)), dtype=float)  # Generation costs
d_2_curt_costs = np.zeros((len(T), len(Z)), dtype=float)  # Curtailment costs
d_2_nodal_price = np.zeros((len(T), len(N)), dtype=float)  # Nodal prices
d_2_np = np.zeros((len(T), len(Z_FBMC)), dtype=float)  # Nodal price for FBMC zones
d_2_export = np.zeros((len(T), len(Z_not_in_FBMC)), dtype=float)  # Exports to non-FBMC zones
d_2_objective_per_hour = np.zeros((len(T)), dtype=float)  # Objective per hour


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
    prediction_PTC_slack_pos = m.addVars(
        [(t, l) for t in Tsub for l in predictions.columns],
        lb=0,
        ub=GRB.INFINITY,
        name="PTC_slack_pos"    
    )
    prediction_PTC_slack_neg = m.addVars(
        [(t, l) for t in Tsub for l in predictions.columns],
        lb=0,
        ub=GRB.INFINITY,
        name="PTC_slack_neg"
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
        sum(CURT[t, n] * cost_curt for t in Tsub for n in N) +
        sum(prediction_PTC_slack_neg[t,PTC_line]*(cost_curt)*5 + prediction_PTC_slack_pos[t,PTC_line]*(cost_curt)*5 for t in Tsub for PTC_line in predictions.columns),
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
                LINE_F[t, l] == sum(H_mat[(l, n)] * DELTA[t, n] for n in N),
                name=f"line_flow_{t}_{l}"
            )
    print("Built constraints line_flow.")

 
    # Add positive line capacity constraints for all lines
    for t in Tsub:
        for l in L:
            m.addConstr(
                LINE_F[t, l] <= get_line_cap(l) * (1 - frm),  # Positive capacity limit with reliability margin
                name=f"line_cap_pos_cnec_{t}_{l}"
            )
    print("Built constraints line_cap_pos for CNECs.")

    # Add negative line capacity constraints for all lines
    for t in Tsub:
        for l in L:
            m.addConstr(
                LINE_F[t, l] >= -get_line_cap(l) * (1 - frm),  # Negative capacity limit with reliability margin
                name=f"line_cap_neg_cnec_{t}_{l}"
            )
    print("Built constraints line_cap_neg for CNECs.")

    # Fix the slack node (e.g., DELTA[t, 68] = 0 for all t in Tsub)
    for t in Tsub:
        m.addConstr(DELTA[t, 50] == 0, name=f"fix_slack_node_{t}")
    print("Built constraints FIX SLACK NODE.")

    for t in Tsub:
        for PTC_lines in predictions.columns:
            Branch_IDs = PTC_mapping[PTC_mapping['PTC'] == PTC_lines]['BranchID']
            Branch_IDs_ints = [int(id.strip()) for id in Branch_IDs.iloc[0].strip('[]').split(',')]
        
            m.addConstr(sum(LINE_F[t, l] for l in Branch_IDs_ints) == predictions.loc[t, PTC_lines] + prediction_PTC_slack_pos[t, PTC_lines] - prediction_PTC_slack_neg[t, PTC_lines], name=f"PTC_line_{t}_{PTC_lines}")
    print("Built constraints FIX FMBC CROSS BORDER LINES.")
        
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
            
        #extract line flow slack variables
        for l_idx, l in enumerate(predictions.columns):
            d_2_PTC_slack_pos[start_index+t_idx, l_idx] = prediction_PTC_slack_pos[t, l].X
            d_2_PTC_slack_neg[start_index+t_idx, l_idx] = prediction_PTC_slack_neg[t, l].X
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

# Create DataFrame for LINE_F_slack_pos
df_line_f_slack_pos = pd.DataFrame(d_2_PTC_slack_pos, index=T, columns=predictions.columns)
df_line_f_slack_pos.index.name = "Time Step"
df_line_f_slack_pos.columns.name = "Line"

# Create DataFrame for LINE_F_slack_neg
df_line_f_slack_neg = pd.DataFrame(d_2_PTC_slack_neg, index=T, columns=predictions.columns)
df_line_f_slack_neg.index.name = "Time Step"
df_line_f_slack_neg.columns.name = "Line"

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
df_line_f_slack_pos.to_csv(f'{Basecase_fixed_results_path}\\df_PTC_slack_pos.csv')
df_line_f_slack_neg.to_csv(f'{Basecase_fixed_results_path}\\df_PTC_slack_neg.csv')



########################
###   FIXED D-1 MC   ###
###        PTC       ###
########################

if use_NP:
    MC_fixed_results_path = 'D-1_MC_fixed_NP_results/' + str(Number_of_hours) + '_hours'
    if not os.path.exists(MC_fixed_results_path):
        os.makedirs(MC_fixed_results_path)
    d_2_np = pd.read_csv(f'D-2_base_case_fixed_NP_results\\{Number_of_hours}_hours\\df_np.csv', index_col="Time Step")
    d_2_line_f = pd.read_csv(f'D-2_base_case_fixed_NP_results\\{Number_of_hours}_hours\\df_line_f.csv', index_col="Time Step")

if use_PTC:
    MC_fixed_results_path = 'D-1_MC_fixed_PTC_results/' + str(Number_of_hours) + '_hours'
    if not os.path.exists(MC_fixed_results_path):
        os.makedirs(MC_fixed_results_path)
    d_2_np = pd.read_csv(f'D-2_base_case_fixed_PTC_results\\{Number_of_hours}_hours\\df_np.csv', index_col="Time Step")
    d_2_line_f = pd.read_csv(f'D-2_base_case_fixed_PTC_results\\{Number_of_hours}_hours\\df_line_f.csv', index_col="Time Step")

T = d_2_np.index

d_2_np = d_2_np.to_numpy()
d_2_line_f = d_2_line_f.to_numpy()

# Calculate PTDF_Z
PTDF_Z = np.matmul(PTDF, gsk_mc)  # Multiply PTDF with gsk_mc

# Subset PTDF_Z for Critical Network Elements (CNEC)
CNEC_indices = [L.index(l) for l in CNEC if l in L]  # Find indices of CNEC in L
PTDF_Z_CNEC = PTDF_Z[CNEC_indices, :]  # Subset rows corresponding to CNEC

#save the PTDF_Z_CNEC to a csv file
PTDF_Z_CNEC_df = pd.DataFrame(PTDF_Z_CNEC, columns=Z_FBMC, index=CNEC)
PTDF_Z_CNEC_df.to_csv(f'{MC_fixed_results_path}/PTDF_Z_CNEC.csv')


# Curtailment cost
cost_curt_mc = 0

# Define dimensions
T_len = len(T)  # Total time steps
N_len = len(N)  # Number of nodes
P_len = len(P)  # Number of generators
Z_FBMC_len = len(Z_FBMC)  # Number of FBMC zones
Z_len = len(Z)  # Total zones
Z_not_in_FBMC_len = len(Z_not_in_FBMC)  # Zones not in FBMC

# Create empty matrices to store values
d_1_curt = np.zeros((T_len, N_len), dtype=float)  # Curtailment
d_1_gen = np.zeros((T_len, P_len), dtype=float)  # Generation
d_1_np = np.zeros((T_len, Z_FBMC_len), dtype=float)  # Net position for FBMC zones
d_1_dump_dem = np.zeros((T_len, Z_len), dtype=float)  # Dumped demand
d_1_gen_costs = np.zeros((T_len, Z_len), dtype=float)  # Generation costs
d_1_curt_costs = np.zeros((T_len, Z_len), dtype=float)  # Curtailment costs
d_1_nodal_price = np.zeros((T_len, Z_FBMC_len), dtype=float)  # Nodal prices for FBMC zones
d_1_nodal_price_abroad = np.zeros((T_len, Z_not_in_FBMC_len), dtype=float)  # Nodal prices abroad
d_1_obj_per_hour = np.zeros((T_len), dtype=float)  # Objective value per hour
d_1_dual_flow_on_cnes_pos = np.zeros((T_len, len(CNEC)), dtype=float)  # Dual variable for flow_on_cnes_pos
d_1_dual_flow_on_cnes_neg = np.zeros((T_len, len(CNEC)), dtype=float)  # Dual variable for flow_on_cnes_neg

export_data = []  # List to store export data

# Precompute RAM_pos and RAM_neg for each time step and CNEC
RAM_pos = {}
RAM_neg = {}

for t in T:
    for j in CNEC:
        # Compute RAM_pos and RAM_neg
        RAM_pos[(t, j)] = get_line_cap(j) * (1 - frm) - d_2_line_f[t-len(T) - 1, L.index(j)]
        RAM_neg[(t, j)] = -get_line_cap(j) * (1 - frm) - d_2_line_f[t-len(T) - 1, L.index(j)]
# Create a list to store RAM data
ram_data = []

# Populate the RAM data for each (t, j)
for t in T:
    for j in CNEC:
        ram_data.append({
            "Time Step": t,
            "CNEC": j,
            "RAM_Pos": RAM_pos[(t, j)],
            "RAM_Neg": RAM_neg[(t, j)]
        })

# Convert the list to a pandas DataFrame
df_ram = pd.DataFrame(ram_data)


# Loop through horizons
for horizon in range(1, ceil(len(T) / hours_per_horizon) + 1):  # Python range is inclusive of start and exclusive of end
    print(f"Horizon: {horizon}/{ceil(len(T) / hours_per_horizon)}")
    
    # Calculate the subset of time steps (Tsub)
    start_index = (horizon - 1) * hours_per_horizon  # Use zero-based indexing
    end_index = min(horizon * hours_per_horizon, len(T))  # Ensure we do not exceed the length of T
    
    # Extract the actual time steps from T based on the calculated indices
    Tsub = T[start_index:end_index]  # Use slicing to get the subset of T

    # Print or use Tsub
    print(f"Tsub for Horizon {horizon}: {Tsub[:]}...")  # Print first 10 elements for brevity

    # Initialize the Gurobi model
    m = Model("Optimization_Model")
    ub_CURT = {(t, n): get_renew(t, n) for t in Tsub for n in N}

    # Define variables
    CURT = m.addVars(
        [(t, n) for t in Tsub for n in N],
        lb=0,
        ub={(t, n): ub_CURT[t, n] for t in Tsub for n in N},  # Explicitly reference the dictionary
        name="CURT"
    )

    NP = m.addVars(
        [(t, z) for t in Tsub for z in Z_FBMC],
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="NP"
    )
    ub_GEN = {(t, p): get_gen_up(p) for t in Tsub for p in P}

    GEN = m.addVars(
        [(t, p) for t in Tsub for p in P],
        lb=0,
        ub={(t, p): ub_GEN[t, p] for t in Tsub for p in P},  # Explicitly index ub_GEN
        name="GEN"
    )

    EXPORT = m.addVars(
        [(t, z, zz) for t in Tsub for z in Z for zz in z_to_z[z]],
        lb=0,
        ub=max_ntc,
        name="EXPORT"
    )

    GEN_COSTS = m.addVars(
        [(t, z) for t in Tsub for z in Z],
        lb=0,
        ub=GRB.INFINITY,
        name="GEN_COSTS"
    )

    CURT_COSTS = m.addVars(
        [(t, z) for t in Tsub for z in Z],
        lb=0,
        ub=GRB.INFINITY,
        name="CURT_COSTS"
    )

    OBJECTIVE_PER_HOUR = m.addVars(
        [(t) for t in Tsub],
        lb=-GRB.INFINITY,
        ub=GRB.INFINITY,
        name="OBJECTIVE_PER_HOUR"
    )

    print("Variables done.")
    # Define the objective function
    m.setObjective(
        sum(GEN[t, p] * get_mc(p) for t in Tsub for p in P) +
        sum(CURT[t, n] * cost_curt_mc for t in Tsub for n in N)+
        sum(EXPORT[t, z, zz]*0.0000001 for t in Tsub for z in Z for zz in z_to_z[z]),
        GRB.MINIMIZE
    )

    # Add constraints for objective per hour
    for t in Tsub:
        m.addConstr(
            OBJECTIVE_PER_HOUR[t] == sum(GEN[t, p] * get_mc(p) for p in P) +
            sum(CURT[t, n] * cost_curt_mc for n in N),
            name=f"objective_per_hour_{t}"
        )
    
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
                sum(CURT[t, n] * cost_curt_mc for n in n_in_z[z]) == CURT_COSTS[t, z],
                name=f"costs_curt_{t}_{z}"
            )
    print("Built constraints costs_curt.")

    # Add zonal balance constraints
    for t in Tsub:
        for z in Z_FBMC:
            # Determine zones to consider for exports and imports
            if z in Z_FBMC:
                zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]
            else:
                zz_list = z_to_z[z]

            # Add the zonal balance constraint
            m.addConstr(
                sum(GEN[t, p] for p in p_in_z[z]) +  # Generation in the zone
                sum(get_renew(t, n) for n in n_in_z[z]) -  # Renewable generation
                sum(CURT[t, n] for n in n_in_z[z]) -  # Curtailment
                sum(EXPORT[t, z, zz] for zz in zz_list) +  # Exports from zone z
                sum(EXPORT[t, zz, z] for zz in zz_list) -  # Imports to zone z
                NP[t, z] ==  # Net position in the zone
                sum(get_dem(t, n) for n in n_in_z[z]),  # Demand in the zone
                name=f"zonal_balance_{t}_{z}"
            )
    print("Built constraints zonal_balance.")
    
    # Add zonal balance constraints for zones outside FBMC
    for t in Tsub:
        for z in Z_not_in_FBMC:
            # Determine zones to consider for exports and imports
            if z in Z_FBMC:
                zz_list = [zz for zz in z_to_z[z] if zz not in Z_FBMC]
            else:
                zz_list = z_to_z[z]

            # Add the zonal balance constraint for zones outside FBMC
            m.addConstr(
                sum(GEN[t, p] for p in p_in_z[z]) +  # Generation in the zone
                sum(get_renew(t, n) for n in n_in_z[z]) -  # Renewable generation
                sum(CURT[t, n] for n in n_in_z[z]) -  # Curtailment
                sum(EXPORT[t, z, zz] for zz in zz_list) +  # Exports from zone z
                sum(EXPORT[t, zz, z] for zz in zz_list) ==  # Imports to zone z
                sum(get_dem(t, n) for n in n_in_z[z]),  # Demand in the zone
                name=f"zonal_balance_abroad_{t}_{z}"
            )
    print("Built constraints zonal_balance abroad.")
    
    # Add net position sum-zero constraint inside FBMC
    for t in Tsub:
        m.addConstr(
            sum(NP[t, z] for z in Z_FBMC) == 0,  # Sum of net positions across FBMC zones equals zero
            name=f"net_position_fbmc_{t}"
        )
    print("Net positions sum-zero inside of FBMC.")

    # Add flow constraints on CNECs (positive)
    for t in Tsub:
        for j in CNEC:
            m.addConstr(
                sum(
                    PTDF_Z_CNEC[CNEC.index(j), Z_FBMC.index(z_fb)] *
                    (NP[t, z_fb] - d_2_np[t-1-len(T), Z.index(z_fb)])
                    for z_fb in Z_FBMC
                )
                <= RAM_pos[(t, j)],
                name=f"flow_on_cnes_pos_{t}_{j}"
            )
    print("Flows on CNECs (pos).")

    # Add flow constraints on CNECs (negative)
    for t in Tsub:
        for j in CNEC:
            m.addConstr(
                sum(
                    PTDF_Z_CNEC[CNEC.index(j), Z_FBMC.index(z_fb)] *
                    (NP[t, z_fb] - d_2_np[t-1-len(T), Z.index(z_fb)])
                    for z_fb in Z_FBMC
                )
                >= RAM_neg[(t, j)],
                name=f"flow_on_cnes_neg_{t}_{j}"
            )
    print("Flows on CNECs (neg).")

    # Print a message confirming all constraints are added
    print("Constraints done.")

    # Optimize the model
    m.optimize()

    # Check the optimization status
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

        d_1_obj_per_hour[start_index+t_idx] = OBJECTIVE_PER_HOUR[t].X  # Get value of OBJECTIVE_PER_HOUR

        for n_idx, n in enumerate(N):
            d_1_curt[start_index+t_idx, n_idx] = CURT[t, n].X  # Get value of CURT

        for z_idx, z in enumerate(Z_FBMC):
            d_1_np[start_index+t_idx, z_idx] = NP[t, z].X  # Get value of NP

        for p_idx, p in enumerate(P):
            d_1_gen[start_index+t_idx, p_idx] = GEN[t, p].X  # Get value of GEN

        for z_idx, z in enumerate(Z):
            d_1_gen_costs[start_index+t_idx, z_idx] = GEN_COSTS[t, z].X  # Get value of GEN_COSTS
            d_1_curt_costs[start_index+t_idx, z_idx] = CURT_COSTS[t, z].X  # Get value of CURT_COSTS

        for z_idx, z in enumerate(Z_FBMC):
            # Dual prices for zonal_balance constraints
            d_1_nodal_price[start_index+t_idx, z_idx] = m.getConstrByName(f"zonal_balance_{t}_{z}").Pi
        for j in CNEC:
            d_1_dual_flow_on_cnes_pos[start_index+t_idx, CNEC.index(j)] = m.getConstrByName(f"flow_on_cnes_pos_{t}_{j}").Pi
            d_1_dual_flow_on_cnes_neg[start_index+t_idx, CNEC.index(j)] = m.getConstrByName(f"flow_on_cnes_neg_{t}_{j}").Pi

        for z_idx, z in enumerate(Z_not_in_FBMC):
            # Dual prices for zonal_balance_abroad constraints
            d_1_nodal_price_abroad[start_index+t_idx, z_idx] = m.getConstrByName(f"zonal_balance_abroad_{t}_{z}").Pi
    #extract the export values
        for z in Z:
            for zz in z_to_z[z]:
                export_data.append({
                    "Time Step": t,
                    "Zone From": z,
                    "Zone To": zz,
                    "Export": EXPORT[t, z, zz].X
                })
    # Clear the model reference
    m.dispose()

    
# export d_1_curt, d_1_np, d_1_gen, d_1_nodal_price, d_1_obj_per_hour
d_1_curt_df = pd.DataFrame(d_1_curt, columns=N, index=range(1, d_1_curt.shape[0] + 1))
d_1_np_df = pd.DataFrame(d_1_np, columns=Z_FBMC, index=range(1, d_1_np.shape[0] + 1))
d_1_gen_df = pd.DataFrame(d_1_gen, columns=P, index=range(1, d_1_gen.shape[0] + 1))
d_1_nodal_price_df = pd.DataFrame(d_1_nodal_price, columns=Z_FBMC, index=range(1, d_1_nodal_price.shape[0] + 1))
d_1_obj_per_hour_df = pd.DataFrame(d_1_obj_per_hour, columns=['Objective Value'], index=range(1, d_1_obj_per_hour.shape[0] + 1))
d_1_dual_flow_on_cnes_pos_df = pd.DataFrame(d_1_dual_flow_on_cnes_pos, columns=CNEC, index=range(1, d_1_dual_flow_on_cnes_pos.shape[0] + 1))
d_1_dual_flow_on_cnes_neg_df = pd.DataFrame(d_1_dual_flow_on_cnes_neg, columns=CNEC, index=range(1, d_1_dual_flow_on_cnes_neg.shape[0] + 1))

# Save the DataFrames to CSV files
d_1_curt_df.to_csv(f'{MC_fixed_results_path}/d_1_curt.csv')
d_1_np_df.to_csv(f'{MC_fixed_results_path}/d_1_np.csv')
d_1_gen_df.to_csv(f'{MC_fixed_results_path}/d_1_gen.csv')
d_1_nodal_price_df.to_csv(f'{MC_fixed_results_path}/d_1_nodal_price.csv')
d_1_obj_per_hour_df.to_csv(f'{MC_fixed_results_path}/d_1_obj_per_hour.csv')
d_1_dual_flow_on_cnes_pos_df.to_csv(f'{MC_fixed_results_path}/d_1_dual_variable_ram_pos.csv')
d_1_dual_flow_on_cnes_neg_df.to_csv(f'{MC_fixed_results_path}/d_1_dual_variable_ram_neg.csv')


# make a dataframe out of the export data
export_df = pd.DataFrame(export_data)
D1_flows = np.matmul(PTDF_Z_CNEC,np.transpose(d_1_np))
D1_flows = np.transpose(D1_flows)
D1_flows_df = pd.DataFrame(D1_flows, columns=CNEC, index=range(1, d_1_np.shape[0] + 1))

#based on cross_border_lines list extract the columns from the flows_test_tranposed_df which are in the cross_border_lines list
cross_border_flows = D1_flows_df[cross_border_lines]


# Exports from FB zones to non-FBMC zones
exports_fb_to_non_fb = []
for t in range(1, len(T)+1):  # Iterate over time steps
    for z in Z_FBMC:  # Iterate over FB zones
        for zz in Z_not_in_FBMC:  # Iterate over non-FBMC zones
            exports_fb_to_non_fb.append({
                "Time Step": t,
                "Zone From": z,
                "Zone To": zz,
                "Export": export_df.loc[
                (export_df["Time Step"] == t) & 
                (export_df["Zone From"] == z) & 
                (export_df["Zone To"] == zz), "Export"
            ].values[0] if not export_df.loc[
                (export_df["Time Step"] == t) & 
                (export_df["Zone From"] == z) & 
                (export_df["Zone To"] == zz), "Export"
            ].empty else 0
            })

# Extract optimized values for exports from non-FBMC zones to FB zones
exports_non_fb_to_fb = []
for t in range(1, len(T)+1):  # Iterate over time steps
    for z in Z_FBMC:  # Iterate over FB zones
        for zz in Z_not_in_FBMC:  # Iterate over non-FBMC zones
            exports_non_fb_to_fb.append({
                "Time Step": t,
                "Zone From": zz,
                "Zone To": z,
                "Export": export_df.loc[
                (export_df["Time Step"] == t) &
                (export_df["Zone From"] == zz) &
                (export_df["Zone To"] == z), "Export"
            ].values[0] if not export_df.loc[
                (export_df["Time Step"] == t) &
                (export_df["Zone From"] == zz) &
                (export_df["Zone To"] == z), "Export"
            ].empty else 0
            })
            
# Convert the lists to DataFrames
df_exports_fb_to_non_fb = pd.DataFrame(exports_fb_to_non_fb)
df_exports_non_fb_to_fb = pd.DataFrame(exports_non_fb_to_fb)

# Create combined columns for 'Zone From' -> 'Zone To'
df_exports_fb_to_non_fb["Zone Pair"] = df_exports_fb_to_non_fb["Zone From"] + " -> " + df_exports_fb_to_non_fb["Zone To"]
df_exports_non_fb_to_fb["Zone Pair"] = df_exports_non_fb_to_fb["Zone From"] + " -> " + df_exports_non_fb_to_fb["Zone To"]

# Pivot table for exports from FB zones to non-FBMC zones
pivot_fb_to_non_fb = df_exports_fb_to_non_fb.pivot(index="Time Step", columns="Zone Pair", values="Export")
pivot_fb_to_non_fb = pivot_fb_to_non_fb[['1 -> Import/Export_1','2 -> Import/Export_2','3 -> Import/Export_3']]

# Pivot table for exports from non-FBMC zones to FB zones
pivot_non_fb_to_fb = df_exports_non_fb_to_fb.pivot(index="Time Step", columns="Zone Pair", values="Export")
pivot_non_fb_to_fb = pivot_non_fb_to_fb[['Import/Export_1 -> 1','Import/Export_2 -> 2','Import/Export_3 -> 3']]

# Calculate cross-border flows to the import/export zones
cross_border_flows_adjacent_zones = pivot_fb_to_non_fb - pivot_non_fb_to_fb.values

all_cross_border_flows = pd.concat([cross_border_flows, cross_border_flows_adjacent_zones], axis=1) 
# set index of all_cross_border_flows to be the same as T
all_cross_border_flows.index = T
all_cross_border_flows.to_parquet(f'{MC_fixed_results_path}/Y_line.parquet')

######################################
### MAP FROM INDIVIDUAL LINE FLOWS ###
###            TO PTCs             ###
######################################

# Step 2: Create a mapping for each (FromBus, ToBus) to a set of valid BranchIDs
PTC_mapping = (
    df_branch[df_branch['BranchID'].isin(cross_border_lines)]
    .groupby(['FromBus', 'ToBus'])['BranchID']
    .apply(list)
    .to_dict()
)

PTC_mapping_df = pd.DataFrame([{"FromBus": k[0], "ToBus": k[1], "BranchID": v} for k, v in PTC_mapping.items()])

# Step 1: Create a mapping of columns in cross_border_flows to (FromBus, ToBus) pairs
column_to_bus_mapping = {
    branch_id: (from_bus, to_bus)
    for (from_bus, to_bus), branch_ids in PTC_mapping.items()
    for branch_id in branch_ids
}

# Step 2: Initialize a dictionary to hold aggregated flows for each (FromBus, ToBus)
PTC_flows_dict = {}

# Step 3: Aggregate flows for each (FromBus, ToBus) using vectorized operations
for (from_bus, to_bus) in set(column_to_bus_mapping.values()):
    # Get all branch IDs associated with this (FromBus, ToBus) pair
    related_columns = [
        branch_id for branch_id, buses in column_to_bus_mapping.items()
        if buses == (from_bus, to_bus)
    ]
    # Sum flows for these columns if any exist
    if related_columns:
        PTC_flows_dict[(from_bus, to_bus)] = cross_border_flows[related_columns].sum(axis=1)

# Convert the dictionary to a DataFrame
PTC_flows = pd.DataFrame(PTC_flows_dict)


# Rename the columns in `aggregated_flows` to a single column format "X->Y"
PTC_flows.columns = [f"{from_bus}->{to_bus}" for from_bus, to_bus in PTC_flows.columns]

# Function to handle opposite flow directions
def consolidate_opposite_flows(df):
    # Create a copy of the DataFrame to avoid modifying the original directly
    df = df.copy()
    columns_to_check = list(df.columns)  # List of columns to iterate over

    while columns_to_check:
        col = columns_to_check.pop(0)  # Get and remove the first column from the list
        from_bus, to_bus = col.split("->")  # Extract the bus pair in the form "X->Y"
        reverse_col = f"{to_bus}->{from_bus}"  # Find the reverse flow column "Y->X"

        # If the reverse column exists in the DataFrame
        if reverse_col in df.columns:
            # Add the reverse flow (multiplied by -1) to the current flow
            df[col] += -df[reverse_col]

            # Drop the reverse flow column and remove it from the columns_to_check list
            df.drop(columns=[reverse_col], inplace=True)
            if reverse_col in columns_to_check:
                columns_to_check.remove(reverse_col)

    return df

# Apply the updated function to the aggregated_flows DataFrame
PTC_flows = consolidate_opposite_flows(PTC_flows)



####################################
###   CONNECT LINES WITH ZONES   ###
###         MAPPING TABLE        ###
####################################

# Split PTC_flows into FromBus and ToBus
PTC_flows_split = pd.DataFrame(PTC_flows.columns.str.split("->", expand=True).to_list(), columns=["FromBus", "ToBus"])
PTC_flows_split["FromBus"] = PTC_flows_split["FromBus"].astype(str)  # Keep as strings for MultiIndex compatibility
PTC_flows_split["ToBus"] = PTC_flows_split["ToBus"].astype(str)  # Keep as strings for MultiIndex compatibility

# Assign these as MultiIndex columns to PTC_flows
PTC_flows.columns = pd.MultiIndex.from_frame(PTC_flows_split)

# Join df_branch with df_bus to get zone information for FromBus and ToBus
from_bus_zones = df_bus.set_index("BusID")["Zone"]
to_bus_zones = df_bus.set_index("BusID")["Zone"]

# Create a mapping of zones for each bus pair in PTC_flows
flows_with_zones = PTC_flows.copy()

# Extract levels from the MultiIndex
from_bus = flows_with_zones.columns.get_level_values("FromBus").astype(int)
to_bus = flows_with_zones.columns.get_level_values("ToBus").astype(int)

# Map zones using df_bus
zone_from = from_bus.map(from_bus_zones)
zone_to = to_bus.map(to_bus_zones)

# Reassign MultiIndex with ZoneFrom and ZoneTo added
flows_with_zones.columns = pd.MultiIndex.from_tuples([
    (f, t, zf, zt) for f, t, zf, zt in zip(from_bus, to_bus, zone_from, zone_to)
], names=["FromBus", "ToBus", "ZoneFrom", "ZoneTo"])


# Create a DataFrame for the new flows
manual_flows_data = {
    (9, 10, 1, "Import/Export_1"): all_cross_border_flows["1 -> Import/Export_1"],
    (55, 119, 2, "Import/Export_2"): all_cross_border_flows["2 -> Import/Export_2"],
    (89, 120, 3, "Import/Export_3"): all_cross_border_flows["3 -> Import/Export_3"]
}

# Convert the dictionary into a DataFrame with MultiIndex columnse
manual_flows = pd.DataFrame(manual_flows_data, index=flows_with_zones.index)

# Set the MultiIndex for the manual flows
manual_flows.columns = pd.MultiIndex.from_tuples(
    manual_flows_data.keys(), names=["FromBus", "ToBus", "ZoneFrom", "ZoneTo"]
)

# Combine the manual flows with the existing flows_with_zones
flows_with_zones_updated = pd.concat([flows_with_zones, manual_flows], axis=1)


###################################
###   CALCULATE NET POSITIONS   ###
###################################

# Calculate net positions of the zones (without considering adjacent zones)
# These sum up to 0
zone_exports = flows_with_zones.T.groupby(level="ZoneFrom").sum().T
zone_imports = flows_with_zones.T.groupby(level="ZoneTo").sum().T
net_positions_FBMC = zone_exports.subtract(zone_imports, fill_value=0)


# Calculate net positions of the zones
# adding also the adjacent zones
zone_exports_2 = flows_with_zones_updated.T.groupby(level="ZoneFrom").sum().T
zone_imports_2 = flows_with_zones_updated.T.groupby(level="ZoneTo").sum().T
net_positions_with_adjacent_zones = zone_exports_2.subtract(zone_imports_2, fill_value=0)

for zone in [1, 2, 3]:
    if str(zone) in net_positions_with_adjacent_zones.columns and zone in net_positions_with_adjacent_zones.columns:
        # Combine numeric and string representations
        net_positions_with_adjacent_zones[zone] += net_positions_with_adjacent_zones[str(zone)]
        # Remove the string version of the zone
        net_positions_with_adjacent_zones.drop(columns=str(zone), inplace=True)


Line_z2z_mapping_table = pd.DataFrame(
    [{"FromBus": col[0], "ToBus": col[1], "ZoneFrom": col[2], "ZoneTo": col[3], "PTC": f"{col[0]}->{col[1]}"} 
     for col in flows_with_zones_updated.columns]
)


# Extract the data and update the column names to only use the "Flattened" (X->Y) format
PTC_flows = flows_with_zones_updated.copy(deep=True)
PTC_flows.columns = [f"{col[0]}->{col[1]}" for col in flows_with_zones_updated.columns]

# write out PTC_flows as parquet 
PTC_flows.index = T
PTC_flows.to_parquet(f'{MC_fixed_results_path}/Y_PTC.parquet')
net_positions_FBMC.index = T
net_positions_FBMC.to_parquet(f'{MC_fixed_results_path}/Y_NP_FBMC.parquet')
net_positions_with_adjacent_zones.index = T
net_positions_with_adjacent_zones.to_parquet(f'{MC_fixed_results_path}/Y_NP_with_adjacent_zones.parquet')

# merge PTC_mapping_df with data_with_flattened_columns on FromBus and ToBus
final_line_mapping_table = pd.merge(PTC_mapping_df, Line_z2z_mapping_table, on=["FromBus", "ToBus"], how = 'right')


# manually add branches to virtual bidding zone PTCs
final_line_mapping_table["BranchID"] = final_line_mapping_table.apply(
    lambda row: [6, 309] if row["PTC"] == "9->10" else
                [305, 306] if row["PTC"] == "55->119" else
                [308, 307] if row["PTC"] == "89->120" else row["BranchID"],
    axis=1
)

# export df_ram as parquet
df_ram.to_parquet(f'{MC_fixed_results_path}/RAM.parquet')


#####################
###    D-1 CGM    ###
###      PTC      ###
#####################

Basecase_fixed_results_path = 'D-1_CGM_fixed_PTC_results_flat/' + str(Number_of_hours) + '_hours'
if not os.path.exists(Basecase_fixed_results_path):
    os.makedirs(Basecase_fixed_results_path)

PTC_mapping = pd.read_csv(f'D-1_MC_results_flat\\{Number_of_hours}_hours\\PTC_mapping.csv', index_col=0)
#load predictions
if use_PTC:
    generation = pd.read_csv(f'D-1_MC_fixed_PTC_results\\{Number_of_hours}_hours\\d_1_gen.csv', index_col=0)
    curtailment = pd.read_csv(f'D-1_MC_fixed_PTC_results\\{Number_of_hours}_hours\\d_1_curt.csv', index_col=0)

generation.index = predictions.index
curtailment.index = predictions.index
     
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
        sum(CURT[t, n] * cost_curt for t in Tsub for n in N),
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
                LINE_F[t, l]  ==     
                sum(H_mat[(l, n)] * DELTA[t, n] for n in N),
                name=f"line_flow_{t}_{l}"
            )
    print("Built constraints line_flow.")

    # Fix the slack node (e.g., DELTA[t, 68] = 0 for all t in Tsub)
    for t in Tsub:
        m.addConstr(DELTA[t, 50] == 0, name=f"fix_slack_node_{t}")
    print("Built constraints FIX SLACK NODE.")

    #Fix generation to the generation from MC
    for t in Tsub:
        for p in P:
            m.addConstr(GEN[t, p] == generation.loc[t, str(p)], name=f"fix_gen_{t}_{p}")

    #fix curtailment to the curtailment from MC
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
