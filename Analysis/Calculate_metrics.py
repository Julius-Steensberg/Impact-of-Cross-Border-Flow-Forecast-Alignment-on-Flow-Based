#%%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Number_of_days = 30 * 6
Number_of_hours = Number_of_days * 24

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_csv(file_path):
    return pd.read_csv(os.path.abspath(os.path.join(script_dir, file_path)), index_col=0)

ML_prediction_NP = load_csv(f"../ML_results/{Number_of_hours}_hours/predictions_NP.csv")
overload_fixed_NP_flat = load_csv(f"../D-1_CGM_fixed_NP_results_flat/{Number_of_hours}_hours/overload_df.csv")
overload_basecase_flat = load_csv(f"../D-1_CGM_results_flat/{Number_of_hours}_hours/overload_df.csv")
overload_perfect_NP_flat = load_csv(f"../D-1_CGM_perfect_NP_results_flat/{Number_of_hours}_hours/overload_df.csv")
overload_perfect_PTC_flat = load_csv(f"../D-1_CGM_perfect_PTC_results_flat/{Number_of_hours}_hours/overload_df.csv")
#overload_fixed_PTC_flat = load_csv(f"../D-1_CGM_fixed_PTC_results_flat/{Number_of_hours}_hours/overload_df.csv")
overload_fixed_PTC_flat_50_epochs = load_csv(f"../D-1_CGM_fixed_PTC_results_flat_50_epochs/{Number_of_hours}_hours/overload_df.csv")
overload_fixed_PTC_flat_300_epochs = load_csv(f"../D-1_CGM_fixed_PTC_results_flat_300_epochs/{Number_of_hours}_hours/overload_df.csv")

#take the index from ML_prediction_NP and extract the corresponding rows from overload_fixed_NP_flat
overload_basecase_flat = overload_basecase_flat.loc[ML_prediction_NP.index]

PTDF_Z_CNE_path_flat = os.path.join(
    "..", "D-1_MC_fixed_NP_results_flat", f"{Number_of_hours}_hours", "PTDF_Z_CNEC.csv")
PTDF_Z_CNE_path_flat = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path_flat))
PTDF_Z_CNE_flat = pd.read_csv(PTDF_Z_CNE_path_flat, index_col=0)
CNEC_lines_list_flat = (PTDF_Z_CNE_flat.index).to_list()



def calculate_total_number_of_overloads(overload_df):
    return overload_df.count().sum()

def calculate_summed_overloads(overload_df):
    return overload_df.abs().sum().sum()

def calculate_total_number_of_overloads_for_CNEC_lines(overload_df, CNEC_lines_list):
    # Ensure CNEC list and DataFrame columns have the same type
    if overload_df.columns.dtype == 'object':  # If columns are stored as strings
        CNEC_lines_list = [str(line) for line in CNEC_lines_list]  # Convert list to strings
        
    else:  # If columns are numeric
        CNEC_lines_list = [int(line) for line in CNEC_lines_list]  # Convert list to integers
    
    # Find valid columns that exist in the DataFrame
    valid_columns = [col for col in CNEC_lines_list if col in overload_df.columns]
    
    if not valid_columns:
        print("⚠️ Warning: No matching CNEC lines found in overload DataFrame.")
    
    # Return count of non-null values in valid CNEC columns
    return overload_df[valid_columns].count().sum()

def calculate_total_number_of_overloads_for_non_CNEC_lines(overload_df, CNEC_lines_list):
    
    # Ensure CNEC list and DataFrame columns have the same type
    if overload_df.columns.dtype == 'object':  # If columns are stored as strings
        CNEC_lines_list = [str(line) for line in CNEC_lines_list]  # Convert list to strings
        
    else:  # If columns are numeric
        CNEC_lines_list = [int(line) for line in CNEC_lines_list]  # Convert list to integers
    
    # Find valid columns that exist in the DataFrame
    valid_columns = [col for col in overload_df.columns if col not in CNEC_lines_list]
    
    if not valid_columns:
        print("⚠️ Warning: No matching non-CNEC lines found in overload DataFrame.")
    
    # Return count of non-null values in valid CNEC columns
    return overload_df[valid_columns].count().sum()





overload_data = {
       "NP (Prediction)": {
        "Number of Overloads": calculate_total_number_of_overloads(overload_fixed_NP_flat),
        "Summed Overloads": calculate_summed_overloads(overload_fixed_NP_flat),
        "Number of Overloads for CNEC Lines": calculate_total_number_of_overloads_for_CNEC_lines(overload_fixed_NP_flat, CNEC_lines_list_flat),
        "Number of Overloads for Non-CNEC Lines": calculate_total_number_of_overloads_for_non_CNEC_lines(overload_fixed_NP_flat, CNEC_lines_list_flat),
    },
    "NP (Perfect)": {
        "Number of Overloads": calculate_total_number_of_overloads(overload_perfect_NP_flat),
        "Summed Overloads": calculate_summed_overloads(overload_perfect_NP_flat),
        "Number of Overloads for CNEC Lines": calculate_total_number_of_overloads_for_CNEC_lines(overload_perfect_NP_flat, CNEC_lines_list_flat),
        "Number of Overloads for Non-CNEC Lines": calculate_total_number_of_overloads_for_non_CNEC_lines(overload_perfect_NP_flat, CNEC_lines_list_flat),
    },
  #  "PTC (Prediction)": {
  #      "Number of Overloads": calculate_total_number_of_overloads(overload_fixed_PTC_flat),
  #      "Summed Overloads": calculate_summed_overloads(overload_fixed_PTC_flat),
  #      "Number of Overloads for CNEC Lines": calculate_total_number_of_overloads_for_CNEC_lines(overload_fixed_PTC_flat, CNEC_lines_list_flat),
  #      "Number of Overloads for Non-CNEC Lines": calculate_total_number_of_overloads_for_non_CNEC_lines(overload_fixed_PTC_flat, CNEC_lines_list_flat),
   # },
    "PTC (Prediction) 50 epochs": {
        "Number of Overloads": calculate_total_number_of_overloads(overload_fixed_PTC_flat_50_epochs),
        "Summed Overloads": calculate_summed_overloads(overload_fixed_PTC_flat_50_epochs),
        "Number of Overloads for CNEC Lines": calculate_total_number_of_overloads_for_CNEC_lines(overload_fixed_PTC_flat_50_epochs, CNEC_lines_list_flat),
        "Number of Overloads for Non-CNEC Lines": calculate_total_number_of_overloads_for_non_CNEC_lines(overload_fixed_PTC_flat_50_epochs, CNEC_lines_list_flat),
    },
    "PTC (Prediction) 300 epochs": {
        "Number of Overloads": calculate_total_number_of_overloads(overload_fixed_PTC_flat_300_epochs),
        "Summed Overloads": calculate_summed_overloads(overload_fixed_PTC_flat_300_epochs),
        "Number of Overloads for CNEC Lines": calculate_total_number_of_overloads_for_CNEC_lines(overload_fixed_PTC_flat_300_epochs, CNEC_lines_list_flat),
        "Number of Overloads for Non-CNEC Lines": calculate_total_number_of_overloads_for_non_CNEC_lines(overload_fixed_PTC_flat_300_epochs, CNEC_lines_list_flat),
    },
    "PTC (Perfect)": {
        "Number of Overloads": calculate_total_number_of_overloads(overload_perfect_PTC_flat),
        "Summed Overloads": calculate_summed_overloads(overload_perfect_PTC_flat),
        "Number of Overloads for CNEC Lines": calculate_total_number_of_overloads_for_CNEC_lines(overload_perfect_PTC_flat, CNEC_lines_list_flat),
        "Number of Overloads for Non-CNEC Lines": calculate_total_number_of_overloads_for_non_CNEC_lines(overload_perfect_PTC_flat, CNEC_lines_list_flat),
},
    "Base Case (Flat)": {
        "Number of Overloads": calculate_total_number_of_overloads(overload_basecase_flat),
        "Summed Overloads": calculate_summed_overloads(overload_basecase_flat),
        "Number of Overloads for CNEC Lines": calculate_total_number_of_overloads_for_CNEC_lines(overload_basecase_flat, CNEC_lines_list_flat),
        "Number of Overloads for Non-CNEC Lines": calculate_total_number_of_overloads_for_non_CNEC_lines(overload_basecase_flat, CNEC_lines_list_flat),
}
}
   

def pretty_print_overloads(data):
    print("\n" + "=" * 50)
    print("Overloads Summary")
    print("=" * 50)
    print(f"{'Scenario':<32}{'Number of Overloads':<20}{'Summed Overloads (MW)':<20}{'Number of Overloads for CNEC Lines':<20}{'Number of Overloads for Non-CNEC Lines':<20}")
    print("-" * 50)
    for key, value in data.items():
        print(f"{key:<32}{value['Number of Overloads']:<20}{np.round(value['Summed Overloads'], 2):<20}{value['Number of Overloads for CNEC Lines']:<20}{value['Number of Overloads for Non-CNEC Lines']:<20}")
    print("=" * 50 + "\n")

pretty_print_overloads(overload_data)
#%%
# Load FBME data
def load_parquet(file_path):
    return pd.read_parquet(os.path.abspath(os.path.join(script_dir, file_path))).reindex(index=ML_prediction_NP.index)

FBME_fixed_NP_flat = load_parquet(f"FBME_{Number_of_hours}_hours_flat_fixed_NP/FBME_flat.parquet")
FBME_perfect_NP_flat = load_parquet(f"FBME_{Number_of_hours}_hours_flat_perfect_NP/FBME_flat.parquet")
#FBME_fixed_PTC_flat = load_parquet(f"FBME_{Number_of_hours}_hours_flat_fixed_PTC/FBME_flat.parquet")
FBME_perfect_PTC_flat = load_parquet(f"FBME_{Number_of_hours}_hours_flat_perfect_PTC/FBME_flat.parquet")
FBME_basecase_flat = load_parquet(f"FBME_{Number_of_hours}_hours_flat_base_case/FBME_flat.parquet")
FBME_fixed_PTC_flat_50_epochs = load_parquet(f"FBME_{Number_of_hours}_hours_flat_fixed_PTC_50_epochs/FBME_flat.parquet")
FBME_fixed_PTC_flat_300_epochs = load_parquet(f"FBME_{Number_of_hours}_hours_flat_fixed_PTC_300_epochs/FBME_flat.parquet")

def calculate_total_FBME(FBME_df):
    return FBME_df.abs().mean().mean(), FBME_df.values.flatten().std()

FBME_data = {
    "NP (Prediction))": calculate_total_FBME(FBME_fixed_NP_flat),
    "NP (Perfect)": calculate_total_FBME(FBME_perfect_NP_flat),
 #   "PTC (Prediction)": calculate_total_FBME(FBME_fixed_PTC_flat),    
    "PTC (Prediction) 50 epochs": calculate_total_FBME(FBME_fixed_PTC_flat_50_epochs),
    "PTC (Prediction) 300 epochs": calculate_total_FBME(FBME_fixed_PTC_flat_300_epochs),
    "PTC (Perfect)": calculate_total_FBME(FBME_perfect_PTC_flat),
    "Base Case (Flat)": calculate_total_FBME(FBME_basecase_flat),
}

def pretty_print_fbme(data):
    print("\n" + "=" * 70)
    print("FBME Summary")
    print("=" * 70)
    print(f"{'Scenario':<32}{'Mean FBME (MW)':<20}{'FBME Std Dev (MW)':<20}")
    print("-" * 70)
    for key, value in data.items():
        print(f"{key:<32}{np.round(value[0], 2):<20}{np.round(value[1], 2):<20}")
    print("=" * 70 + "\n")

pretty_print_fbme(FBME_data)
#%%
# Load Zonal Price Data
zonal_prices_fixed_NP_flat = load_csv(f"../D-1_MC_fixed_NP_results_flat/{Number_of_hours}_hours/d_1_nodal_price.csv")
zonal_prices_perfect_NP_flat = load_csv(f"../D-1_MC_perfect_NP_results_flat/{Number_of_hours}_hours/d_1_nodal_price.csv")
#zonal_prices_fixed_PTC_flat = load_csv(f"../D-1_MC_fixed_PTC_results/{Number_of_hours}_hours/d_1_nodal_price.csv")
zonal_prices_perfect_PTC_flat = load_csv(f"../D-1_MC_perfect_PTC_results/{Number_of_hours}_hours/d_1_nodal_price.csv")


# Function to calculate mean and std of zonal prices
def calculate_mean_zonal_prices(zonal_prices_df):
    zonal_prices_mean = zonal_prices_df.mean(axis=0)
    zonal_prices_mean['Mean (All Zones)'] = zonal_prices_mean.mean()
    
    zonal_prices_std = zonal_prices_df.std(axis=0)
    zonal_prices_std['Std (All Zones)'] = zonal_prices_df.values.flatten().std()  # Overall std

    return zonal_prices_mean, zonal_prices_std

# Calculate means and stds
zonal_prices_data = {
    "CGMA fixed NP (Flat)": calculate_mean_zonal_prices(zonal_prices_fixed_NP_flat),
    "CGMA perfect NP (Flat)": calculate_mean_zonal_prices(zonal_prices_perfect_NP_flat),
 #   "CGMA fixed PTC (Flat)": calculate_mean_zonal_prices(zonal_prices_fixed_PTC_flat),
    "CGMA perfect PTC (Flat)": calculate_mean_zonal_prices(zonal_prices_perfect_PTC_flat),
}

# Pretty Print Function
def pretty_print_zonal_prices(title, means, stds):
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'-' * 50}")
    print(f"{'Zone':<15}{'Mean (EUR/MWh)':<20}{'Std Dev (EUR/MWh)':<20}")
    print(f"{'-' * 50}")
    
    # Iterate over zones
    for zone in means.index[:-1]:  # Exclude the overall mean for now
        print(f"{zone:<15}{np.round(means[zone], 2):<20}{np.round(stds[zone], 2):<20}")
    
    # Print the overall mean and std
    print(f"{'-' * 50}")
    print(f"{'Overall':<15}{np.round(means[-1], 2):<20}{np.round(stds[-1], 2):<20}")
    print(f"{'=' * 50}\n")

# Print Results
for scenario, (mean, std) in zonal_prices_data.items():
    pretty_print_zonal_prices(scenario, mean, std)

#%%
import tools
#load df_objective_per_hour
def load_csv(file_path):
    return pd.read_csv(os.path.abspath(os.path.join(script_dir, file_path)), index_col=0)

df_objective_per_hour_fixed_NP_flat = load_csv(f"../D-1_MC_fixed_NP_results_flat/{Number_of_hours}_hours/d_1_obj_per_hour.csv")
df_objective_per_hour_perfect_NP_flat = load_csv(f"../D-1_MC_perfect_NP_results_flat/{Number_of_hours}_hours/d_1_obj_per_hour.csv")
df_objective_per_hour_fixed_PTC_flat = load_csv(f"../D-1_MC_fixed_PTC_results/{Number_of_hours}_hours/d_1_obj_per_hour.csv")
df_objective_per_hour_perfect_PTC_flat = load_csv(f"../D-1_MC_perfect_PTC_results/{Number_of_hours}_hours/d_1_obj_per_hour.csv")    

# Function to calculate mean of the objective values
def calculate_mean_objective(df_objective_per_hour):
    return df_objective_per_hour.mean(axis=0)

# Creating a structured table with Base Case and CGMA as columns, Flat and Pmax Sub as rows
summary_table = pd.DataFrame({
    "CGMA fixed NP": [
        np.round(calculate_mean_objective(df_objective_per_hour_fixed_NP_flat).mean(), 0),
    ],

    "CGMA perfect NP": [
        np.round(calculate_mean_objective(df_objective_per_hour_perfect_NP_flat).mean(), 0),
    ],
    "CGMA fixed PTC": [
        np.round(calculate_mean_objective(df_objective_per_hour_fixed_PTC_flat).mean(), 0),
    ],
    "CGMA perfect PTC": [
        np.round(calculate_mean_objective(df_objective_per_hour_fixed_PTC_flat).mean(), 0),
    ]
}, index=["Flat"])

print(summary_table)



# %%
# read D-1 MC NPs 
df_NP_fixed_NP_D_1 = load_csv(f"../D-1_MC_fixed_NP_results_flat/{Number_of_hours}_hours/d_1_np.csv")
df_NP_perfect_NP_D_1 = load_csv(f"../D-1_MC_perfect_NP_results_flat/{Number_of_hours}_hours/d_1_np.csv")
df_NP_fixed_PTC_50_epochs_D_1 = load_csv(f"../D-1_MC_fixed_PTC_results_50_epochs/{Number_of_hours}_hours/d_1_np.csv")
df_NP_fixed_PTC_300_epochs_D_1 = load_csv(f"../D-1_MC_fixed_PTC_results_300_epochs/{Number_of_hours}_hours/d_1_np.csv")
df_NP_perfect_PTC_D_1 = load_csv(f"../D-1_MC_perfect_PTC_results/{Number_of_hours}_hours/d_1_np.csv")
df_NP_base_case_D_1 = load_csv(f"../D-1_MC_results_flat/{Number_of_hours}_hours/d_1_np.csv")
#df_NP_base_case_D_1_parquet from C:\Users\jlist\GitHub\Master_thesis_Part2\Paper_repo\D-1_MC_results_flat\4320_hours
df_NP_base_case_D_1_parquet =load_parquet('c:\\Users\\jlist\\GitHub\\Master_thesis_Part2\\Paper_repo\\D-1_MC_results_flat\\4320_hours\\Y_NP_FBMC.parquet')
#clip df_NP_base_case_D_1 by index of df_NP_base_case_D_1_parquet
df_NP_base_case_D_1 = df_NP_base_case_D_1.loc[df_NP_base_case_D_1.index.intersection(df_NP_base_case_D_1_parquet.index)]
#read D-2 CGM NPs
df_NP_fixed_NP_CGM_D_2 = load_csv(f"../D-2_base_case_fixed_NP_results_flat/{Number_of_hours}_hours/df_np.csv")
df_NP_perfect_NP_CGM_D_2 = load_csv(f"../D-2_CGM_perfect_NP_results_flat/{Number_of_hours}_hours/df_np.csv")
df_NP_fixed_PTC_CGM_D_2_50_epochs = load_csv(f"../D-2_base_case_fixed_PTC_results_50_epochs/{Number_of_hours}_hours/df_np.csv")
df_NP_fixed_PTC_CGM_D_2_300_epochs = load_csv(f"../D-2_base_case_fixed_PTC_results_300_epochs/{Number_of_hours}_hours/df_np.csv")
df_NP_perfect_PTC_CGM_D_2 = load_csv(f"../D-2_CGM_perfect_PTC_results/{Number_of_hours}_hours/df_np.csv")
#give df_NP_D_1 the same index as df_NP_CGM_D_2
df_NP_fixed_NP_D_1.index = df_NP_fixed_NP_CGM_D_2.index
df_NP_perfect_NP_D_1.index = df_NP_perfect_NP_CGM_D_2.index
df_NP_fixed_PTC_50_epochs_D_1.index = df_NP_fixed_PTC_CGM_D_2_50_epochs.index
df_NP_fixed_PTC_300_epochs_D_1.index = df_NP_fixed_PTC_CGM_D_2_300_epochs.index
df_NP_perfect_PTC_D_1.index = df_NP_perfect_PTC_CGM_D_2.index

diff_NP_fixed_NP = df_NP_fixed_NP_CGM_D_2 - df_NP_fixed_NP_D_1
diff_NP_fixed_PTC_50_epochs = df_NP_fixed_PTC_CGM_D_2_50_epochs - df_NP_fixed_PTC_50_epochs_D_1
diff_NP_fixed_PTC_300_epochs = df_NP_fixed_PTC_CGM_D_2_300_epochs - df_NP_fixed_PTC_300_epochs_D_1

comparison_NP_path = os.path.join(
    "..", "ML_results", f"{Number_of_hours}_hours", "comparison_NP.csv")
comparison_NP_path = os.path.abspath(os.path.join(script_dir, comparison_NP_path))

comparison_NP = pd.read_csv(comparison_NP_path, index_col='Timestep')
comparison_NP = comparison_NP.drop(columns='Unnamed: 0')

# Fixing index and column names
comparison_NP['CGMA Prediction Error'] = comparison_NP['Actual'] - comparison_NP['Predicted']

comparison_PTC_50_epochs_path = os.path.join(
    "..", "ML_results", f"{Number_of_hours}_hours", "comparison_PTC_50_epochs.csv") 
comparison_PTC_50_epochs_path = os.path.abspath(os.path.join(script_dir, comparison_PTC_50_epochs_path))
comparison_PTC_50_epochs = pd.read_csv(comparison_PTC_50_epochs_path, index_col='Timestep')
comparison_PTC_50_epochs = comparison_PTC_50_epochs.drop(columns='Unnamed: 0')
comparison_PTC_50_epochs['CGMA Prediction Error'] = comparison_PTC_50_epochs['Actual'] - comparison_PTC_50_epochs['Predicted']
comparison_PTC_300_epochs_path = os.path.join(
    "..", "ML_results", f"{Number_of_hours}_hours", "comparison_PTC_300_epochs.csv")
comparison_PTC_300_epochs_path = os.path.abspath(os.path.join(script_dir, comparison_PTC_300_epochs_path))
comparison_PTC_300_epochs = pd.read_csv(comparison_PTC_300_epochs_path, index_col='Timestep')
comparison_PTC_300_epochs = comparison_PTC_300_epochs.drop(columns='Unnamed: 0')
comparison_PTC_300_epochs['CGMA Prediction Error'] = comparison_PTC_300_epochs['Actual'] - comparison_PTC_300_epochs['Predicted']

# make a df called total_abs_CGMA_error_per_hour  with timestep as index and the absolute error per hour as column
total_abs_CGMA_error_per_hour = pd.DataFrame(index=comparison_PTC_300_epochs.index.unique())
total_abs_CGMA_error_per_hour['CGMA Prediction Error NP'] = comparison_NP['CGMA Prediction Error'].abs().groupby(comparison_NP.index).sum()
total_abs_CGMA_error_per_hour['CGMA Prediction Error PTC 50 epochs'] = comparison_PTC_50_epochs['CGMA Prediction Error'].abs().groupby(comparison_PTC_50_epochs.index).sum()
total_abs_CGMA_error_per_hour['CGMA Prediction Error PTC 300 epochs'] = comparison_PTC_300_epochs['CGMA Prediction Error'].abs().groupby(comparison_PTC_300_epochs.index).sum()

#load PTC_mapping
PTC_mapping_path = os.path.join(
    "..", "D-1_MC_results_flat", f"{Number_of_hours}_hours", "PTC_mapping.csv")
PTC_mapping_path = os.path.abspath(os.path.join(script_dir, PTC_mapping_path))
PTC_mapping = pd.read_csv(PTC_mapping_path)

# Ensure 'ZoneFrom' and 'ZoneTo' are strings (important for column matching)
PTC_mapping['ZoneFrom'] = PTC_mapping['ZoneFrom'].astype(str)
PTC_mapping['ZoneTo'] = PTC_mapping['ZoneTo'].astype(str)

# Initialize empty DataFrame with expected zone columns (e.g., '1', '2', '3')
zones = sorted(PTC_mapping['ZoneFrom'].unique())  # or use union with 'ZoneTo' if needed
comparison_PTC_to_NP_50_epochs = pd.DataFrame(0, index=comparison_PTC_50_epochs.index.unique(), columns=zones )

# Loop through each row and distribute flows
for row in comparison_PTC_50_epochs.itertuples():
    timestep = row.Index 
    PTC_line = row.Line
    predicted_value = row.Predicted

    NP_zone_from = PTC_mapping.loc[PTC_mapping['PTC'] == PTC_line, 'ZoneFrom'].values[0]
    NP_zone_to = PTC_mapping.loc[PTC_mapping['PTC'] == PTC_line, 'ZoneTo'].values[0]
    if NP_zone_to not in comparison_PTC_to_NP_50_epochs.columns:
        continue
    else:
        # Accumulate predicted flow (inject at 'from', withdraw at 'to')
        comparison_PTC_to_NP_50_epochs.at[timestep, NP_zone_from] += predicted_value
        comparison_PTC_to_NP_50_epochs.at[timestep, NP_zone_to] -= predicted_value

comparison_PTC_to_NP_300_epochs = pd.DataFrame(0, index=comparison_PTC_300_epochs.index.unique(), columns=zones)
# Loop through each row and distribute flows
for row in comparison_PTC_300_epochs.itertuples():
    timestep = row.Index 
    PTC_line = row.Line
    predicted_value = row.Predicted

    NP_zone_from = PTC_mapping.loc[PTC_mapping['PTC'] == PTC_line, 'ZoneFrom'].values[0]
    NP_zone_to = PTC_mapping.loc[PTC_mapping['PTC'] == PTC_line, 'ZoneTo'].values[0]
    if NP_zone_to not in comparison_PTC_to_NP_300_epochs.columns:
        continue
    else:
        # Accumulate predicted flow (inject at 'from', withdraw at 'to')
        comparison_PTC_to_NP_300_epochs.at[timestep, NP_zone_from] += predicted_value
        comparison_PTC_to_NP_300_epochs.at[timestep, NP_zone_to] -= predicted_value
#pivot comparison_NP to have timestep as index and zone as columns
#%%
true_NP_values = comparison_NP.reset_index().pivot(index='Timestep', columns='Line', values='Actual')
# Ensure the columns are strings for consistency
true_NP_values.columns = true_NP_values.columns.astype(str)

for row in comparison_PTC_to_NP_50_epochs.itertuples():
    timestep = row.Index 
    comparison_PTC_to_NP_50_epochs.at[timestep, 'Zone 1 NP error'] = comparison_PTC_to_NP_50_epochs.at[timestep, '1'] - true_NP_values.at[timestep, '1']
    comparison_PTC_to_NP_50_epochs.at[timestep, 'Zone 2 NP error'] = comparison_PTC_to_NP_50_epochs.at[timestep, '2'] - true_NP_values.at[timestep, '2']
    comparison_PTC_to_NP_50_epochs.at[timestep, 'Zone 3 NP error'] = comparison_PTC_to_NP_50_epochs.at[timestep, '3'] - true_NP_values.at[timestep, '3']
for row in comparison_PTC_to_NP_300_epochs.itertuples():
    timestep = row.Index 
    comparison_PTC_to_NP_300_epochs.at[timestep, 'Zone 1 NP error'] = comparison_PTC_to_NP_300_epochs.at[timestep, '1'] - true_NP_values.at[timestep, '1']
    comparison_PTC_to_NP_300_epochs.at[timestep, 'Zone 2 NP error'] = comparison_PTC_to_NP_300_epochs.at[timestep, '2'] - true_NP_values.at[timestep, '2']
    comparison_PTC_to_NP_300_epochs.at[timestep, 'Zone 3 NP error'] = comparison_PTC_to_NP_300_epochs.at[timestep, '3'] - true_NP_values.at[timestep, '3']

total_abs_CGMA_error_per_hour['CGMA Prediction Error PTC 50 epochs in NP'] = comparison_PTC_to_NP_50_epochs[['Zone 1 NP error', 'Zone 2 NP error', 'Zone 3 NP error']].abs().sum(axis=1)
total_abs_CGMA_error_per_hour['CGMA Prediction Error PTC 300 epochs in NP'] = comparison_PTC_to_NP_300_epochs[['Zone 1 NP error', 'Zone 2 NP error', 'Zone 3 NP error']].abs().sum(axis=1)
GSK = '_flat'
FBME_path_NP50 = os.path.join(
        "..", "Analysis", f"FBME_{Number_of_hours}_hours{GSK}_fixed_NP", f"FBME{GSK}.parquet")
FBME_path_NP50 = os.path.abspath(os.path.join(script_dir, FBME_path_NP50))
FBME_NP50 = pd.read_parquet(FBME_path_NP50)
FBME_NP50 = FBME_NP50.abs().mean(axis=1)

#%%
import matplotlib.pyplot as plt

# Ensure alignment of indexes
aligned_index = total_abs_CGMA_error_per_hour.index.intersection(diff_NP_fixed_PTC_50_epochs.index)


# Compute sum of absolute differences in NP between D-1 and D-2 for each timestep
sum_diff_NP_fixed_PTC_50_epochs = diff_NP_fixed_PTC_50_epochs.loc[aligned_index].abs().sum(axis=1)
sum_diff_NP_fixed_PTC_300_epochs = diff_NP_fixed_PTC_300_epochs.loc[aligned_index].abs().sum(axis=1)
sum_diff_NP_fixed_NP = diff_NP_fixed_NP.loc[aligned_index].abs().sum(axis=1)

# Extract CGMA prediction error for aligned timesteps
cgma_prediction_error_ptc_50 = total_abs_CGMA_error_per_hour.loc[aligned_index, 'CGMA Prediction Error PTC 50 epochs in NP']
cgma_prediction_error_ptc_300 = total_abs_CGMA_error_per_hour.loc[aligned_index, 'CGMA Prediction Error PTC 300 epochs in NP']
cgma_prediction_error_np = total_abs_CGMA_error_per_hour.loc[aligned_index, 'CGMA Prediction Error NP']
# Create scatter plot
plt.figure(figsize=(10, 6))
#add a y=x line
# Add y = x reference line
plt.plot([0, 5000],
         [0, 5000],
         color='red', linestyle='--', label='y = x')
#plt.scatter(sum_diff_NP_fixed_PTC_50_epochs, cgma_prediction_error_ptc_50, alpha=0.6, edgecolor='black', label='PTC 50 epochs', color='blue')
#plt.scatter(sum_diff_NP_fixed_PTC_300_epochs, cgma_prediction_error_ptc_300, alpha=0.6, edgecolor='black', label='PTC 300 epochs', color='orange')
plt.scatter(sum_diff_NP_fixed_NP, cgma_prediction_error_np, alpha=0.6, edgecolor='black', label='NP', color='green')
plt.legend()
plt.xlabel('Sum of |D1 - D2 NP| Differences [MW]')
plt.ylabel('CGMA Prediction Error PTC 50 epochs in NP [MW]')
plt.title('CGMA Prediction Error vs. D-1 to D-2 NP Differences (PTC 50 epochs)')
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

# Align indexes
aligned_index = total_abs_CGMA_error_per_hour.index.intersection(diff_NP_fixed_NP.index).intersection(FBM_NP50.index)

# Compute sum of absolute differences in NP between D-1 and D-2
sum_diff_NP_fixed_NP = diff_NP_fixed_NP.loc[aligned_index].abs().sum(axis=1)

# Extract CGMA prediction error for aligned timesteps
cgma_prediction_error_np = total_abs_CGMA_error_per_hour.loc[aligned_index, 'CGMA Prediction Error NP']

# FBME values for coloring
fbme_values = FBME_NP50.loc[aligned_index]

# Normalize the color scale
norm = colors.Normalize(vmin=fbme_values.min(), vmax=fbme_values.max())
cmap = cm.viridis  # You can choose any colormap you like




# Create scatter plot
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    sum_diff_NP_fixed_NP,
    cgma_prediction_error_np,
    c=fbme_values,
    cmap=cmap,
    norm=norm,
    alpha=1,
    edgecolor='black',
    linewidth=0.5,
)

# Add a color bar to represent FBME
cbar = plt.colorbar(scatter)
cbar.set_label('Mean Absolute FBME (NP50) [MW]')

# Add y=x reference line
max_val = max(sum_diff_NP_fixed_NP.max(), cgma_prediction_error_np.max())
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', label='y = x')

# Labels and title
plt.xlabel('Sum of |D-1 - D-2 NP| Differences [MW]')
plt.ylabel('CGMA Prediction Error NP [MW]')
plt.title('CGMA Prediction Error vs. D-1 to D-2 NP Differences colored by FBME (NP50)')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
