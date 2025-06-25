#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Get the current script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

Number_of_days = 30*6
Number_of_hours = 24 * Number_of_days


NP50 = False
NPperf = True
PTC50 = False
PTC300 = False
PTCperf = False

#create a folder for the results/plots
results_folder = os.path.join(script_dir, "Shapley_results")
if not os.path.exists(results_folder):
    os.makedirs(results_folder)
########################
###   LOAD INDICES   ###
########################
if NP50 or NPperf:
    comparison_NP_path = os.path.join(
        "..", "ML_results", f"{Number_of_hours}_hours", "comparison_NP.csv")
    comparison_NP_path = os.path.abspath(os.path.join(script_dir, comparison_NP_path))

    comparison_NP = pd.read_csv(comparison_NP_path, index_col='Timestep')
    comparison_NP = comparison_NP.drop(columns='Unnamed: 0')

    # Fixing index and column names
    index_of_interest = np.unique(comparison_NP.index)
    zone_cols = ['Zone 1', 'Zone 2', 'Zone 3']
if PTC50 or PTCperf or PTC300:
    if PTC50 or PTCperf:
        comparison_PTC_path = os.path.join(
        "..", "ML_results", f"{Number_of_hours}_hours", "comparison_PTC_50_epochs.csv")
    elif PTC300:
        comparison_PTC_path = os.path.join(
        "..", "ML_results", f"{Number_of_hours}_hours", "comparison_PTC_300_epochs.csv")
    comparison_PTC_path = os.path.abspath(os.path.join(script_dir, comparison_PTC_path))

    comparison_PTC = pd.read_csv(comparison_PTC_path, index_col='Timestep')
    comparison_PTC = comparison_PTC.drop(columns='Unnamed: 0')

    # Fixing index and column names
    index_of_interest = np.unique(comparison_PTC.index)
    zone_cols = ['Zone 1', 'Zone 2', 'Zone 3']
    PTC_lines = np.unique(comparison_PTC['Line'].values)
    PTC_lines = PTC_lines.tolist()

####################################
###   SET GSK FOR LOADING DATA   ###
####################################

GSK = '_flat'

FBME_mean = True
FBME_all = False

########################
###   LOAD D-1 CGM   ###
########################
if NP50:
    D1_CGM_NP_path = os.path.join(
        "..", f"D-1_MC_fixed_NP_results{GSK}", f"{Number_of_hours}_hours", "d_1_np.csv")
    D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))
elif NPperf:
    D1_CGM_NP_path = os.path.join(
        "..", f"D-1_MC_perfect_NP_results{GSK}", f"{Number_of_hours}_hours", "d_1_np.csv")
    D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))
elif PTC50:
    D1_CGM_NP_path = os.path.join(
        "..", f"D-1_MC_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "d_1_np.csv")
    D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))
elif PTC300:
    D1_CGM_NP_path = os.path.join(
        "..", f"D-1_MC_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours", "d_1_np.csv")
    D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))
elif PTCperf:
    D1_CGM_NP_path = os.path.join(
        "..", f"D-1_MC_perfect_PTC_results", f"{Number_of_hours}_hours", "d_1_np.csv")
    D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))

try:
    D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
    D1_CGM_NP.index = index_of_interest
    D1_CGM_NP.columns = zone_cols
    print('Flat D-1 NP found')
except:
    print('Flat D-1 NP not found')

# columns
cols_D1_NP = ['Zone 1 (D1 NP)', 'Zone 2 (D1 NP)', 'Zone 3 (D1 NP)']
D1_CGM_NP.columns = cols_D1_NP


###########################
###   LOAD D-2 CGM NP   ###
###########################
if NP50:
    D2_CGM_NP_path = os.path.join(
        "..", f"D-2_base_case_fixed_NP_results{GSK}", f"{Number_of_hours}_hours", "df_np.csv")
    D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
elif NPperf:
    D2_CGM_NP_path = os.path.join(
        "..", f"D-2_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours", "df_np.csv")
    D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
elif PTC50:
    D2_CGM_NP_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "df_np.csv")
    D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
elif PTC300:
    D2_CGM_NP_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours", "df_np.csv")
    D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
elif PTCperf:
    D2_CGM_NP_path = os.path.join(
        "..", f"D-2_CGM_perfect_PTC_results", f"{Number_of_hours}_hours", "df_np.csv")
    D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))

try:
    D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
    D2_CGM_NP.index = index_of_interest
    print('Flat D-2 NP found')
except:
    print('Flat D-2 NP not found')

# columns
cols_D2_NP = ['Zone 1 (D2 NP)', 'Zone 2 (D2 NP)', 'Zone 3 (D2 NP)']
D2_CGM_NP.columns = cols_D2_NP


#############################################
###   CALCULATE D1 NP - D2 NP DIFFERENCE  ###
#############################################

cols_NP_diff = ['Zone 1 (D1 NP - D2 NP)', 'Zone 2 (D1 NP - D2 NP)', 'Zone 3 (D1 NP - D2 NP)']
D1_D2_NP_diff = pd.DataFrame(D1_CGM_NP.values - D2_CGM_NP.values, columns=cols_NP_diff, index=index_of_interest)

############################
###   LOAD D-1 CGM GEN   ###
############################
if NP50:
    D1_CGM_GEN_path = os.path.join(
        "..", f"D-1_MC_fixed_NP_results{GSK}", f"{Number_of_hours}_hours", "d_1_gen.csv")
    D1_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D1_CGM_GEN_path))
elif NPperf:
    D1_CGM_GEN_path = os.path.join(
        "..", f"D-1_MC_perfect_NP_results{GSK}", f"{Number_of_hours}_hours", "d_1_gen.csv")
    D1_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D1_CGM_GEN_path))
elif PTC50:
    D1_CGM_GEN_path = os.path.join(
        "..", f"D-1_MC_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "d_1_gen.csv")
    D1_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D1_CGM_GEN_path))
elif PTC300:
    D1_CGM_GEN_path = os.path.join(
        "..", f"D-1_MC_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours", "d_1_gen.csv")
    D1_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D1_CGM_GEN_path))
elif PTCperf:
    D1_CGM_GEN_path = os.path.join(
        "..", f"D-1_MC_perfect_PTC_results", f"{Number_of_hours}_hours", "d_1_gen.csv")
    D1_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D1_CGM_GEN_path))



try:
    D1_CGM_GEN = pd.read_csv(D1_CGM_GEN_path, index_col=0)
    D1_CGM_GEN.index = index_of_interest
    print('Flat D-1 generation found')
except:
    print('Flat D-1 generation not found')

gen_mapping_to_node_path = os.path.join(
    "..", "data", "df_gen_final.csv")
gen_mapping_to_node_path = os.path.abspath(os.path.join(script_dir, gen_mapping_to_node_path))

gen_mapping_to_node = pd.read_csv(gen_mapping_to_node_path, sep=';')

new_gen_columns_names = []

# make new column names as Gen: "column" + Node gen_mapping_to_node['OnBus'].loc[gen_mapping_to_node['GenID'] == int(column)].values[0]
for column in D1_CGM_GEN.columns:
    new_gen_columns_names.append(f"Gen {column} at Node {gen_mapping_to_node['OnBus'].loc[gen_mapping_to_node['GenID'] == int(column)].values[0]} ({gen_mapping_to_node['Type'].loc[gen_mapping_to_node['GenID'] == int(column)].values[0]})")

# columns
add_this_to_each_column = ' (D1)'
cols_gen = [col + add_this_to_each_column for col in new_gen_columns_names]
D1_CGM_GEN.columns = cols_gen

############################
###   LOAD D-2 CGM GEN   ###
############################
if NP50:
    D2_CGM_GEN_path = os.path.join(
        "..", f"D-2_base_case_fixed_NP_results{GSK}", f"{Number_of_hours}_hours", "df_gen.csv")
    D2_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D2_CGM_GEN_path))
elif NPperf:
    D2_CGM_GEN_path = os.path.join(
        "..", f"D-2_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours", "df_gen.csv")
    D2_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D2_CGM_GEN_path))
elif PTC50:
    D2_CGM_GEN_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "df_gen.csv")
    D2_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D2_CGM_GEN_path))
elif PTC300:
    D2_CGM_GEN_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours", "df_gen.csv")
    D2_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D2_CGM_GEN_path))
elif PTCperf:
    D2_CGM_GEN_path = os.path.join(
        "..", f"D-2_CGM_perfect_PTC_results", f"{Number_of_hours}_hours", "df_gen.csv")
    D2_CGM_GEN_path = os.path.abspath(os.path.join(script_dir, D2_CGM_GEN_path))



try:
    D2_CGM_GEN = pd.read_csv(D2_CGM_GEN_path, index_col=0)
    D2_CGM_GEN.index = index_of_interest
    print('Flat D-2 generation found')
except:
    print('Flat D-2 generation not found')

# from C:\Users\SørenNielsenSardeman\OneDrive - Nordic RCC\Documents\Master Thesis\Master_thesis_Part2\Python_repo\data\df_gen_final.csv

new_gen_columns_names = []
# make new column names as Gen: "column" + Node gen_mapping_to_node['OnBus'].loc[gen_mapping_to_node['GenID'] == int(column)].values[0]
for column in D2_CGM_GEN.columns:
    new_gen_columns_names.append(f"Gen {column} at Node {gen_mapping_to_node['OnBus'].loc[gen_mapping_to_node['GenID'] == int(column)].values[0]} ({gen_mapping_to_node['Type'].loc[gen_mapping_to_node['GenID'] == int(column)].values[0]})")

# columns
add_this_to_each_column = ' (D2)'
cols_gen = [col + add_this_to_each_column for col in new_gen_columns_names]
D2_CGM_GEN.columns = cols_gen
    


##############################
###   LOAD D-2 CGM SLACK   ###
##############################
if NP50:
    df_NP_slack_neg_path = os.path.join(
        "..", f"D-2_base_case_fixed_NP_results{GSK}", f"{Number_of_hours}_hours", "df_NP_slack_neg.csv")
    df_NP_slack_neg_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_neg_path))

    df_NP_slack_pos_path = os.path.join(
        "..", f"D-2_base_case_fixed_NP_results{GSK}", f"{Number_of_hours}_hours", "df_NP_slack_pos.csv")
    df_NP_slack_pos_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_pos_path))
elif NPperf:
    df_NP_slack_neg_path = os.path.join(
        "..", f"D-2_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours", "df_NP_slack_neg.csv")
    df_NP_slack_neg_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_neg_path))

    df_NP_slack_pos_path = os.path.join(
        "..", f"D-2_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours", "df_NP_slack_pos.csv")
    df_NP_slack_pos_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_pos_path))

elif PTC50:
    df_NP_slack_neg_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "df_PTC_slack_neg.csv")
    df_NP_slack_neg_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_neg_path))

    df_NP_slack_pos_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "df_PTC_slack_pos.csv")
    df_NP_slack_pos_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_pos_path))
elif PTC300:
    df_NP_slack_neg_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours", "df_PTC_slack_neg.csv")
    df_NP_slack_neg_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_neg_path))

    df_NP_slack_pos_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours", "df_PTC_slack_pos.csv")
    df_NP_slack_pos_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_pos_path))
elif PTCperf:
    df_NP_slack_neg_path = os.path.join(
        "..", f"D-2_CGM_perfect_PTC_results", f"{Number_of_hours}_hours", "df_NP_slack_neg.csv")
    df_NP_slack_neg_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_neg_path))

    df_NP_slack_pos_path = os.path.join(
        "..", f"D-2_CGM_perfect_PTC_results", f"{Number_of_hours}_hours", "df_NP_slack_pos.csv")
    df_NP_slack_pos_path = os.path.abspath(os.path.join(script_dir, df_NP_slack_pos_path))


try:
    if NP50:
        df_NP_slack_neg = pd.read_csv(df_NP_slack_neg_path, index_col=0)
        df_NP_slack_neg.index = index_of_interest

        df_NP_slack_pos = pd.read_csv(df_NP_slack_pos_path, index_col=0)
        df_NP_slack_pos.index = index_of_interest
    elif PTC50 or PTC300:
        df_NP_slack_neg = pd.read_csv(df_NP_slack_neg_path, index_col=0)
        df_NP_slack_neg.index = index_of_interest

        df_NP_slack_pos = pd.read_csv(df_NP_slack_pos_path, index_col=0)
        df_NP_slack_pos.index = index_of_interest
    print('Flat D-2 slack found')
except:
    print('Flat D-2 slack not found')

# columns
if NP50:
    cols_slack_neg = ['Zone 1 (D2 NP slack neg)', 'Zone 2 (D2 NP slack neg)', 'Zone 3 (D2 NP slack neg)']
    cols_slack_pos = ['Zone 1 (D2 NP slack pos)', 'Zone 2 (D2 NP slack pos)', 'Zone 3 (D2 NP slack pos)']
    df_NP_slack_pos.columns = cols_slack_pos
    df_NP_slack_neg.columns = cols_slack_neg
elif PTC50 or PTC300:
    cols_slack_neg = PTC_lines 
    # add slack neg to each column name
    cols_slack_neg = [col + ' (PTC slack neg)' for col in cols_slack_neg]
    cols_slack_pos = PTC_lines
    # add slack pos to each column name
    cols_slack_pos = [col + ' (PTC slack pos)' for col in cols_slack_pos]

    df_NP_slack_pos.columns = cols_slack_pos
    df_NP_slack_neg.columns = cols_slack_neg

#################################
###   CGMA PREDICTION ERROR   ###
#################################
if NP50:
    comparison_NP['CGMA Prediction Error'] = comparison_NP['Actual'] - comparison_NP['Predicted']

    # Pivot the comparison_NP dataframe
    CGMA_prediction_error = comparison_NP.reset_index().pivot(
        index='Timestep', columns='Line', values='CGMA Prediction Error')
    cols_CGMA_pred_error = ['Zone 1 (CGMA error)', 'Zone 2 (CGMA error)', 'Zone 3 (CGMA error)']
    CGMA_prediction_error.columns = cols_CGMA_pred_error
    CGMA_prediction_error = CGMA_prediction_error.abs()
elif PTC50 or PTC300:
    comparison_PTC['CGMA Prediction Error'] = comparison_PTC['Actual'] - comparison_PTC['Predicted']
    # Pivot the comparison_PTC dataframe
    CGMA_prediction_error = comparison_PTC.reset_index().pivot(
        index='Timestep', columns='Line', values='CGMA Prediction Error')
    cols_CGMA_pred_error = PTC_lines 
    # add CGMA error to each column name
    cols_CGMA_pred_error = [col + ' (CGMA error)' for col in cols_CGMA_pred_error]
    CGMA_prediction_error.columns = cols_CGMA_pred_error
    CGMA_prediction_error = CGMA_prediction_error.abs()


#########################################
###   RENEWABLE PRODUCTION PER ZONE   ###
#########################################

renewable_per_hour_per_zone_path = os.path.join(
    "..", "Analysis", "renewable_per_hour_per_zone.csv")
renewable_per_hour_per_zone_path = os.path.abspath(os.path.join(script_dir, renewable_per_hour_per_zone_path))

renewable_per_hour_per_zone = pd.read_csv(renewable_per_hour_per_zone_path, index_col=0)
renewable_per_hour_per_zone.index = renewable_per_hour_per_zone.index + 1
renewable_per_hour_per_zone = renewable_per_hour_per_zone.loc[index_of_interest]

# cols 
cols_renewable = ['Zone 1 (Renewable)', 'Zone 2 (Renewable)', 'Zone 3 (Renewable)', 'I/E 1 (Renewable)', 'I/E 2 (Renewable)', 'I/E 3 (Renewable)']
renewable_per_hour_per_zone.columns = cols_renewable

########################
###   LOAD PER ZONE  ###
########################

demand_path = os.path.join(
    "..", "data", "df_bus_load_added_abroad_final.csv")
demand_path = os.path.abspath(os.path.join(script_dir, demand_path))
bus_path = os.path.join(
    "..", "data", "df_bus_final.csv")
bus_path = os.path.abspath(os.path.join(script_dir, bus_path))

# Import data
df_bus_load = pd.read_csv(demand_path)
df_bus_load.index = df_bus_load.index + 1
df_bus = pd.read_csv(bus_path)

# Create sets
N = df_bus["BusID"].tolist()
Z = sorted(df_bus["Zone"].unique())
n_in_z = {z: [n for n in N if df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0] == z] for z in Z}

# Calculate load per zone
load_per_hour_per_zone = np.zeros((df_bus_load.shape[0], len(Z)))
for i, z in enumerate(Z):
    load_sum_temp = 0
    for n in n_in_z[z]:
        load_sum_temp += df_bus_load[str(n)]
    load_per_hour_per_zone[:, i] = load_sum_temp
    
load_per_hour_per_zone = pd.DataFrame(load_per_hour_per_zone, columns=Z)
load_per_hour_per_zone.index = load_per_hour_per_zone.index + 1
load_per_hour_per_zone = load_per_hour_per_zone.loc[index_of_interest]

# cols
cols_load = ['Zone 1 (Load)', 'Zone 2 (Load)', 'Zone 3 (Load)', 'I/E 1 (Load)', 'I/E 2 (Load)', 'I/E 3 (Load)']
load_per_hour_per_zone.columns = cols_load

##############################
###   NET DEMAND PER ZONE  ###
##############################

# Calculate net demand per hour per zone
cols_ND = ['Zone 1 (Net Demand)', 'Zone 2 (Net Demand)', 'Zone 3 (Net Demand)', 'I/E 1 (Net Demand)', 'I/E 2 (Net Demand)', 'I/E 3 (Net Demand)']
net_demand_per_hour_per_zone = pd.DataFrame(load_per_hour_per_zone.values - renewable_per_hour_per_zone.values, columns=cols_ND, index=index_of_interest)


################################
###   CURTAILMENT IN ZONES   ###
################################
if NP50:
    df_curt_path = os.path.join(
        "..", f"D-2_base_case_fixed_NP_results{GSK}", f"{Number_of_hours}_hours", "df_curt.csv")
    df_curt_path = os.path.abspath(os.path.join(script_dir, df_curt_path))
elif NPperf:
    df_curt_path = os.path.join(
        "..", f"D-2_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours", "df_curt.csv")
    df_curt_path = os.path.abspath(os.path.join(script_dir, df_curt_path))
elif PTC50:
    df_curt_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "df_curt.csv")
    df_curt_path = os.path.abspath(os.path.join(script_dir, df_curt_path))
elif PTC300:
    df_curt_path = os.path.join(
        "..", f"D-2_base_case_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours", "df_curt.csv")
    df_curt_path = os.path.abspath(os.path.join(script_dir, df_curt_path))
elif PTCperf:
    df_curt_path = os.path.join(
        "..", f"D-2_CGM_perfect_PTC_results", f"{Number_of_hours}_hours", "df_curt.csv")
    df_curt_path = os.path.abspath(os.path.join(script_dir, df_curt_path))


df_curt = pd.read_csv(df_curt_path, index_col=0)
df_curt.index = index_of_interest

# map to zones using n_in_z
df_curt_per_zone = {}
for z in Z:
    curt_sum_temp = 0
    for n in n_in_z[z]:
        curt_sum_temp += df_curt[str(n)]
    df_curt_per_zone[z] = curt_sum_temp

df_curt = pd.DataFrame(df_curt_per_zone)

# drop 3 last columns
df_curt = df_curt.iloc[:, :-3]

# cols
cols_curt = ['Zone 1 (Curtailment)', 'Zone 2 (Curtailment)', 'Zone 3 (Curtailment)']
df_curt.columns = cols_curt


##############################
###   GET FBME (Y-LABEL)   ###
##############################
if NP50:
    FBME_path = os.path.join(
        "..", "Analysis", f"FBME_{Number_of_hours}_hours{GSK}_fixed_NP", f"FBME{GSK}.parquet")
    FBME_path = os.path.abspath(os.path.join(script_dir, FBME_path))
elif NPperf:
    FBME_path = os.path.join(
        "..", "Analysis", f"FBME_{Number_of_hours}_hours{GSK}_perfect_NP", f"FBME{GSK}.parquet")
    FBME_path = os.path.abspath(os.path.join(script_dir, FBME_path))
elif PTC50:
    FBME_path = os.path.join(
        "..", "Analysis", f"FBME_{Number_of_hours}_hours{GSK}_fixed_PTC_50_epochs", f"FBME{GSK}.parquet")
    FBME_path = os.path.abspath(os.path.join(script_dir, FBME_path))
elif PTC300:
    FBME_path = os.path.join(
        "..", "Analysis", f"FBME_{Number_of_hours}_hours{GSK}_fixed_PTC_300_epochs", f"FBME{GSK}.parquet")
    FBME_path = os.path.abspath(os.path.join(script_dir, FBME_path))
elif PTCperf:
    FBME_path = os.path.join(
        "..", "Analysis", f"FBME_{Number_of_hours}_hours{GSK}_perfect_PTC", f"FBME{GSK}.parquet")
    FBME_path = os.path.abspath(os.path.join(script_dir, FBME_path))

FBME = pd.read_parquet(FBME_path)

if FBME_mean:
    FBME = FBME.abs().mean(axis=1)

# WHICH CNEC HAS THE ABSOLUTE HIGHEST ERROR?
if FBME_all:
    abs_mean_FBME = FBME.abs().mean()
    abs_mean_FBME = abs_mean_FBME.sort_values(ascending=False)

    # Top 10 CNECs with highest error
    top10_CNEC_with_highest_error = abs_mean_FBME.head(10)

    # Top 1 CNEC with highest error
    CNEC_with_highest_error = FBME[abs_mean_FBME.index[0]]

    chosen_CNEC = 0 # choose which CNEC to analyze, 0 is the first one, 1 is the second one, etc.
    # CNEC 136
    name_of_CNEC = abs_mean_FBME.index[chosen_CNEC]

    top10_name_of_CNEC = abs_mean_FBME.index[:10]

    # column index of CNEC is 44
    column_idx_of_CNEC = FBME.columns.get_loc(name_of_CNEC)
    top10_column_idx_of_CNEC = [FBME.columns.get_loc(c) for c in top10_name_of_CNEC]

###############################
###   MAKE INPUT MATRIX X   ###
###############################

X = pd.concat([#D1_CGM_NP, 
                D2_CGM_NP, 
                #D1_D2_NP_diff, 
                #D1_CGM_GEN, 
                D2_CGM_GEN, # kan være fed
                #df_NP_slack_pos, 
                #df_NP_slack_neg, 
                #df_curt, # kan være fed?
                renewable_per_hour_per_zone, # kan være fed
                #load_per_hour_per_zone, # kan være fed
                net_demand_per_hour_per_zone # kan være fed?
                ], axis=1)
if NP50 or PTC50 or PTC300:
#add CGMA prediction error to X
    X = pd.concat([X, CGMA_prediction_error], axis=1)

# First 80% is TRAIN DATA, last 20% is TEST DATA
X_train = X.iloc[:int(0.8*len(X))]
X_test = X.iloc[int(0.8*len(X)):]

# Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
Min_Max_Scaler = MinMaxScaler()
X_train_scaled = Min_Max_Scaler.fit_transform(X_train)
X_test_scaled = Min_Max_Scaler.transform(X_test)


################################
###   MAKE TARGET Y (FBME)   ###
################################

# First 80% is TRAIN DATA, last 20% is TEST DATA
Y = FBME.values

Y_train = Y[:int(0.8*len(Y))]
Y_test = Y[int(0.8*len(Y)):]

# Standardize
from sklearn.preprocessing import StandardScaler
Standard_Scaler = StandardScaler()

if FBME_mean:
    Y_train = Y_train.reshape(-1, 1)
    Y_test = Y_test.reshape(-1, 1)

Y_train_scaled = Standard_Scaler.fit_transform(Y_train)
Y_test_scaled = Standard_Scaler.transform(Y_test)




import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import numpy as np


# Convert the data into DMatrix format (optimized for XGBoost)
dtrain = xgb.DMatrix(X_train_scaled, label=Y_train_scaled)
dtest = xgb.DMatrix(X_test_scaled, label=Y_test_scaled)

# XGBoost parameters to handle large data and prevent overfitting
params = {
    'objective': 'reg:squarederror',  # Regression task
    'eval_metric': 'rmse',            # Evaluation metric
    'learning_rate': 0.05,            # Controls the learning rate
    'subsample': 0.8,                 # Subsample ratio of the training data
    'colsample_bytree': 0.8,          # Subsample ratio of columns when constructing each tree
    'min_child_weight': 3,            # Minimum sum of instance weight needed in a child
    'max_depth': 6,                   # Maximum depth of a tree
    'lambda': 1.0,                    # L2 regularization term
    'alpha': 0.1,                     # L1 regularization term
    'seed': 42                        # For reproducibility
}

# Training the model
xgb_model = xgb.train(params, dtrain, num_boost_round=100)


# Predictions
Y_train_pred = xgb_model.predict(dtrain)
Y_test_pred = xgb_model.predict(dtest)

if FBME_mean:
    # Inverse scaling for evaluation
    Y_train_pred_inverse = Standard_Scaler.inverse_transform(Y_train_pred.reshape(-1, 1))
    Y_test_pred_inverse = Standard_Scaler.inverse_transform(Y_test_pred.reshape(-1, 1))
if FBME_all:
    # Inverse scaling for evaluation
    Y_train_pred_inverse = Standard_Scaler.inverse_transform(Y_train_pred)
    Y_test_pred_inverse = Standard_Scaler.inverse_transform(Y_test_pred)

# Evaluation
train_mse = mean_squared_error(Y_train, Y_train_pred_inverse)
test_mse = mean_squared_error(Y_test, Y_test_pred_inverse)

train_mae = np.mean(np.abs(Y_train - Y_train_pred_inverse))
test_mae = np.mean(np.abs(Y_test - Y_test_pred_inverse))

print(f'Training MSE: {train_mse}')
print(f'Testing MSE: {test_mse}')
print(f'Training MAE: {train_mae}')
print(f'Testing MAE: {test_mae}')
#plot a timeseries of the first column of Y_test and Y_test_pred


if FBME_all:
    colly = 5
    plt.figure(figsize=(15, 5))
    plt.plot(Y_test[:, colly], label='Actual')
    plt.plot(Y_test_pred_inverse[:, colly], label='Predicted')
    plt.xlabel('Hour')
    plt.ylabel('FBME [MW]')
    plt.legend()
    plt.show()

if FBME_mean:
    plt.figure(figsize=(15, 5))
    plt.plot(Y_test, label='Actual')
    plt.plot(Y_test_pred_inverse, label='Predicted')
    plt.xlabel('Hour')
    plt.ylabel('Absolute FBME [MW]')
    plt.legend()
    plt.tight_layout()

    if NP50:
        plt.title('FBME Mean - NP50')
        filename = os.path.join(results_folder, 'FBME_mean_NP50.png')
    elif PTC50:
        plt.title('FBME Mean - PTC50')
        filename = os.path.join(results_folder, 'FBME_mean_PTC50.png')
    elif NPperf:
        plt.title('FBME Mean - NPperf')
        filename = os.path.join(results_folder, 'FBME_mean_NPperf.png')
    elif PTCperf:
        plt.title('FBME Mean - PTCperf')
        filename = os.path.join(results_folder, 'FBME_mean_PTCperf.png')
    elif PTC300:
        plt.title('FBME Mean - PTC300')
        filename = os.path.join(results_folder, 'FBME_mean_PTC300.png')
    plt.savefig(filename, dpi=300)
    plt.show()


######################
### SHAPLEY VALUES ###
######################

import shap
from joblib import Parallel, delayed

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)

def compute_shap_for_batch(data):
    return explainer.shap_values(data)

batch_size = 100

# Prepare data for SHAP calculations
X_train_df_shapley = pd.DataFrame(X_train_scaled.copy())

# Convert any non-numeric columns to numeric (if possible) or drop them
X_train_df_shapley = X_train_df_shapley.apply(pd.to_numeric, errors='coerce')
X_train_df_shapley = X_train_df_shapley.dropna(axis=1)

# Re-create SHAP explainer after ensuring all columns are numeric
explainer = shap.TreeExplainer(xgb_model)
batches = [X_train_df_shapley.iloc[i:i + batch_size] for i in range(0, len(X_train_df_shapley), batch_size)]

# Compute SHAP values in parallel
shap_values_batches = Parallel(n_jobs=4)(delayed(compute_shap_for_batch)(batch) for batch in batches)

shap_values = np.vstack(shap_values_batches)
##########################################
### SUMMARY PLOT OVERALL BEST FEATURES ###
##########################################

if FBME_all:
    # Plotting most important features for the CNEC with the highest error

    #shap.summary_plot(shap_values[:,:,top10_column_idx_of_CNEC[chosen_CNEC]], X_train_df_shapley, plot_type='bar', max_display=100)
    chosen_CNEC = 0 # choose which CNEC to analyze, 0 is the first one, 1 is the second one, etc.
    # CNEC 136
    name_of_CNEC = abs_mean_FBME.index[chosen_CNEC]

    top10_name_of_CNEC = abs_mean_FBME.index[:10]

    # column index of CNEC is 44
    column_idx_of_CNEC = FBME.columns.get_loc(name_of_CNEC)
    top10_column_idx_of_CNEC = [FBME.columns.get_loc(c) for c in top10_name_of_CNEC]
    # Plotting most important features for the CNEC with the highest error
    plt.figure()
    shap.summary_plot(shap_values[:,:,column_idx_of_CNEC], X_train_df_shapley, show=False)
    plt.title(f"Most important features for CNEC {top10_name_of_CNEC[chosen_CNEC]}")
    plt.show()


if FBME_mean:
    plt.figure()
    shap.summary_plot(shap_values, X_train_df_shapley, show=False, max_display=10)

    if NP50:
        plt.title('SHAP Summary Plot - NP50')
        filename = os.path.join(results_folder, 'shap_summary_plot_NP50.png')
    elif PTC50:
        plt.title('SHAP Summary Plot - PTC50')
        filename = os.path.join(results_folder, 'shap_summary_plot_PTC50.png')
    elif NPperf:
        plt.title('SHAP Summary Plot - NPperf')
        filename = os.path.join(results_folder, 'shap_summary_plot_NPperf.png')
    elif PTCperf:
        plt.title('SHAP Summary Plot - PTCperf')
        filename = os.path.join(results_folder, 'shap_summary_plot_PTCperf.png')
    elif PTC300:
        plt.title('SHAP Summary Plot - PTC300')
        filename = os.path.join(results_folder, 'shap_summary_plot_PTC300.png')
    
    plt.tight_layout()
    print(f"Saving figure to: {filename}")
    plt.savefig(filename, dpi=300)
    plt.show()



#%%
##########################################
### SUMMARY PLOT OVERALL BEST FEATURES ###
##########################################
top_n_features = 20
# Calculate mean SHAP values across all classes
shap_values_mean_across_classes = np.mean(shap_values, axis=1)

shap_mean_df = pd.DataFrame(shap_values_mean_across_classes, columns=X_train_df_shapley.columns)
top_features = shap_mean_df.abs().mean().sort_values(ascending=False).head(top_n_features).index
shap_values_top_overall = shap_mean_df[top_features]

# %%
plt.figure()
shap.summary_plot(shap_values_top_overall.values, X_train_df_shapley[top_features], show=False)
plt.title(f"SHAP Summary Plot for Top {top_n_features} Overall Important Features Across All Flows")
plt.show()


# %%
# Calculate residuals (errors)
train_errors = (Y_train - Y_train_pred_inverse).flatten()
test_errors = (Y_test - Y_test_pred_inverse).flatten()

# 1. Feature-Error Correlation
feature_error_corr = pd.DataFrame(X_train_scaled).corrwith(pd.Series(train_errors))

# sort feature_error_corr from highest to lowest¨
feature_error_corr = feature_error_corr.sort_values(ascending=False)

plt.figure(figsize=(10, 26))
sns.barplot(x=feature_error_corr.values, y=feature_error_corr.index)
plt.title('Correlation between Input Features and Training Errors')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.tight_layout()
plt.show()

