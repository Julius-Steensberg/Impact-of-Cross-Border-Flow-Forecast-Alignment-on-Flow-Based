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


zonal_prices_NP50 = load_csv(f"../D-1_MC_fixed_NP_results_flat/{Number_of_hours}_hours/d_1_nodal_price.csv")
zonal_prices_NPperf = load_csv(f"../D-1_MC_perfect_NP_results_flat/{Number_of_hours}_hours/d_1_nodal_price.csv")
zonal_prices_PTCperf = load_csv(f"../D-1_MC_perfect_PTC_results/{Number_of_hours}_hours/d_1_nodal_price.csv")
zonal_prices_PTC50 = load_csv(f"../D-1_MC_fixed_PTC_results_50_epochs/{Number_of_hours}_hours/d_1_nodal_price.csv")
zonal_prices_PTC300 = load_csv(f"../D-1_MC_fixed_PTC_results_300_epochs/{Number_of_hours}_hours/d_1_nodal_price.csv")

zonal_prices_baseline = load_csv(f"../D-1_MC_results_flat/{Number_of_hours}_hours/d_1_nodal_price.csv")
#only take the first 2160 rows in zonal_prices_baseline
zonal_prices_baseline = zonal_prices_baseline.iloc[:2160, :]

# Flatten data and compute correlation matrix
data_combined = pd.DataFrame({
    'NP50': zonal_prices_NP50.stack(),
    'NPperf': zonal_prices_NPperf.stack(),
    'PTCperf': zonal_prices_PTCperf.stack(),
    'PTC50': zonal_prices_PTC50.stack(),
    'PTC300': zonal_prices_PTC300.stack(),
    'Baseline': zonal_prices_baseline.stack()
})

correlation_matrix_PTC300 = zonal_prices_PTC300.corr()
correlation_matrix_PTC300
correlation_matrix_PTC50 = zonal_prices_PTC50.corr()
correlation_matrix_NP50 = zonal_prices_NP50.corr()
correlation_matrix_NPperf = zonal_prices_NPperf.corr()
correlation_matrix_PTCperf = zonal_prices_PTCperf.corr()
correlation_matrix_baseline = zonal_prices_baseline.corr()
#print all of them
print("Correlation Matrix PTC300:\n", correlation_matrix_PTC300)
print("Correlation Matrix PTC50:\n", correlation_matrix_PTC50)
print("Correlation Matrix NP50:\n", correlation_matrix_NP50)
print("Correlation Matrix NPperf:\n", correlation_matrix_NPperf)
print("Correlation Matrix PTCperf:\n", correlation_matrix_PTCperf)
print("Correlation Matrix Baseline:\n", correlation_matrix_baseline)


price_std_NP50 = zonal_prices_NP50.std(axis=1)  # Std dev per hour
mean_std_NP50 = price_std_NP50.mean()  # Average over time
price_std_NPperf = zonal_prices_NPperf.std(axis=1)  # Std dev per hour
mean_std_NPperf = price_std_NPperf.mean()  # Average over time
price_std_PTCperf = zonal_prices_PTCperf.std(axis=1)  # Std dev per hour
mean_std_PTCperf = price_std_PTCperf.mean()  # Average over time
price_std_PTC50 = zonal_prices_PTC50.std(axis=1)  # Std dev per hour
mean_std_PTC50 = price_std_PTC50.mean()  # Average over time
price_std_PTC300 = zonal_prices_PTC300.std(axis=1)  # Std dev per hour
mean_std_PTC300 = price_std_PTC300.mean()  # Average over time
price_std_baseline = zonal_prices_baseline.std(axis=1)  # Std dev per hour
mean_std_baseline = price_std_baseline.mean()  # Average over time
# Print standard deviations
print("Standard Deviation NP50:", mean_std_NP50)
print("Standard Deviation NPperf:", mean_std_NPperf)
print("Standard Deviation PTCperf:", mean_std_PTCperf)
print("Standard Deviation PTC50:", mean_std_PTC50)
print("Standard Deviation PTC300:", mean_std_PTC300)
print("Standard Deviation Baseline:", mean_std_baseline)

#%%
import itertools
dfs = {
    "NP50": zonal_prices_NP50,
    "NPperf": zonal_prices_NPperf,
    "CBperf": zonal_prices_PTCperf,
    "CB50": zonal_prices_PTC50,
    "CB300": zonal_prices_PTC300,
    "Baseline": zonal_prices_baseline,
}

# Define fairness metrics
def compute_fairness_metrics(df):
    std_dev = df.std(axis=1)
    price_range = df.max(axis=1) - df.min(axis=1)
    uniform_price_share = (df.nunique(axis=1) == 1).mean()

    def gini(array):
        sorted_arr = np.sort(array)
        n = len(array)
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * sorted_arr)) / (n * np.sum(sorted_arr))) if np.sum(sorted_arr) > 0 else 0

    gini_vals = df.apply(gini, axis=1)

    return {
        "Avg Std Dev": std_dev.mean(),
        "Avg Price Range": price_range.mean(),
        "Uniform Price Share": uniform_price_share,
        "Avg Gini": gini_vals.mean()
    }

# Compute metrics for each pricing scheme
fairness_results = {name: compute_fairness_metrics(df) for name, df in dfs.items()}
fairness_df = pd.DataFrame(fairness_results).T
#sort by uniform price share
fairness_df = np.round(fairness_df.sort_values(by="Uniform Price Share", ascending=False),3)

# Select only one pair for plotting: Zone 1 vs Zone 2
zone_pair = ("1", "2")

# Combine all scenarios into one DataFrame for comparison
combined_data = []

for name, df in dfs.items():
    if name == "Baseline" or name == "CB300":
        continue
    if all(zone in df.columns for zone in zone_pair):
        temp_df = df[[zone_pair[0], zone_pair[1]]].copy()
        temp_df['Scenario'] = name
        combined_data.append(temp_df)

combined_df = pd.concat(combined_data, ignore_index=True)

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_df, x=zone_pair[0], y=zone_pair[1], hue='Scenario', alpha=0.5)
plt.xlabel("Price Zone 1")
plt.ylabel("Price Zone 2")
plt.title("Scatter Plot of Zonal Prices: Zone 1 vs Zone 2 Across Scenarios")
plt.grid(True)
plt.tight_layout()
plt.show()
print("Fairness Metrics:"
      "\n", fairness_df)
#%%
# Corrected zone names
zone_pair = ['1', '2']
combined_data = []

# Rebuild combined dataframe using correct column names
for name, df in dfs.items():
   # if name == "Baseline" :
   #     continue
    if all(zone in df.columns for zone in zone_pair):
        temp_df = df[zone_pair].copy()
        temp_df['Scenario'] = name
        combined_data.append(temp_df)

combined_df = pd.concat(combined_data, ignore_index=True)

# Compute price differences
combined_df["Price_Diff"] = combined_df[zone_pair[0]] - combined_df[zone_pair[1]]


# Plot density of price differences
plt.figure(figsize=(10, 6))
sns.kdeplot(data=combined_df, x="Price_Diff", hue="Scenario", fill=True, common_norm=False, alpha=0.4)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.xlabel("Price Difference (Zone1 - Zone2) [€/MWh]")
plt.ylabel("Density")
plt.title("Density Plot of Zonal Price Differences (Zone1 vs Zone2) Across Scenarios")
plt.grid(True)
plt.tight_layout()
plt.show()


# %%
#load this one
#C:\Users\jlist\GitHub\Master_thesis_Part2\Paper_repo\D-1_MC_fixed_PTC_results_50_epochs\4320_hours/RAM.parquet
ram_data_PTC_50 = pd.read_parquet(os.path.abspath(os.path.join(script_dir, "../D-1_MC_fixed_PTC_results_50_epochs/4320_hours/RAM.parquet")))
ram_data_PTC_300 = pd.read_parquet(os.path.abspath(os.path.join(script_dir, "../D-1_MC_fixed_PTC_results_300_epochs/4320_hours/RAM.parquet")))
ram_data_NP_50 = pd.read_parquet(os.path.abspath(os.path.join(script_dir, "../D-1_MC_fixed_NP_results_flat/4320_hours/RAM.parquet")))
ram_data_NP_perf = pd.read_parquet(os.path.abspath(os.path.join(script_dir, "../D-1_MC_perfect_NP_results_flat/4320_hours/RAM.parquet")))
ram_data_PTC_perf = pd.read_parquet(os.path.abspath(os.path.join(script_dir, "../D-1_MC_perfect_PTC_results/4320_hours/RAM.parquet")))
ram_data_baseline = pd.read_parquet(os.path.abspath(os.path.join(script_dir, "../D-1_MC_results_flat/4320_hours/RAM.parquet")))
#for ram_data_baseline, only take the rows where time step is 2160 or greater
ram_data_baseline = ram_data_baseline[ram_data_baseline["Time Step"] >= 2160]
# Merge on Time Step and CNEC to align values
merged = pd.merge(
    ram_data_PTC_50,
    ram_data_PTC_perf,
    on=["Time Step", "CNEC"],
    suffixes=("_PTC50", "_PTC_perf")
)

# Calculate the differences
merged["RAM_Pos_Diff"] = merged["RAM_Pos_PTC50"] - merged["RAM_Pos_PTC_perf"]
merged["RAM_Neg_Diff"] = merged["RAM_Neg_PTC50"] - merged["RAM_Neg_PTC_perf"]

# Summary statistics
ram_pos_summary = merged["RAM_Pos_Diff"].describe()
ram_neg_summary = merged["RAM_Neg_Diff"].describe()

# Plotting 
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(merged["RAM_Pos_Diff"], kde=True, bins=50)
plt.title("Distribution of RAM_Pos Differences (PTC50 - PTCperf)")
plt.xlabel("RAM_Pos Difference (MW)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
sns.histplot(merged["RAM_Neg_Diff"], kde=True, bins=50)
plt.title("Distribution of RAM_Neg Differences (PTC50 - PTCperf)")
plt.xlabel("RAM_Neg Difference (MW)")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

ram_pos_summary, ram_neg_summary

#groupby CNEC and calculate the mean and std of the differences
ram_pos_grouped = merged.groupby("CNEC")["RAM_Pos_Diff"].agg(["mean", "std"]).reset_index()
ram_neg_grouped = merged.groupby("CNEC")["RAM_Neg_Diff"].agg(["mean", "std"]).reset_index()
#sort by mean
ram_pos_grouped = ram_pos_grouped.sort_values(by="mean", ascending=False)
ram_neg_grouped = ram_neg_grouped.sort_values(by="mean", ascending=False)


#%%
import matplotlib.pyplot as plt
import pandas as pd

# Assuming the required dataframes already exist in memory
number =10

# Group and get top 10 lowest RAM_Pos for both scenarios
lowest_avg_ram_PTC50 = ram_data_PTC_50.groupby("CNEC")["RAM_Pos"].mean().nsmallest(number)
lowest_avg_ram_PTCperf = ram_data_PTC_perf.groupby("CNEC")["RAM_Pos"].mean().nsmallest(number)
lowest_avg_ram_PTC300 = ram_data_PTC_300.groupby("CNEC")["RAM_Pos"].mean().nsmallest(number)
lowest_avg_ram_NP50 = ram_data_NP_50.groupby("CNEC")["RAM_Pos"].mean().nsmallest(number)
lowest_avg_ram_NPperf = ram_data_NP_perf.groupby("CNEC")["RAM_Pos"].mean().nsmallest(number)
# Combine into one DataFrame
combined_pos = pd.concat([
    lowest_avg_ram_PTC50.rename("PTC50_RAM_Pos"),
    lowest_avg_ram_PTCperf.rename("PTCperf_RAM_Pos"),
    lowest_avg_ram_PTC300.rename("PTC300_RAM_Pos"),
    lowest_avg_ram_NP50.rename("NP50_RAM_Pos"),
    lowest_avg_ram_NPperf.rename("NPperf_RAM_Pos")
], axis=1).dropna()
# Group and get top 10 highest RAM_Neg (most negative values) for both scenarios
lowest_neg_ram_PTC50 = ram_data_PTC_50.groupby("CNEC")["RAM_Neg"].mean().nlargest(number)
lowest_neg_ram_PTCperf = ram_data_PTC_perf.groupby("CNEC")["RAM_Neg"].mean().nlargest(number)
lowest_neg_ram_PTC300 = ram_data_PTC_300.groupby("CNEC")["RAM_Neg"].mean().nlargest(number)
lowest_neg_ram_NP50 = ram_data_NP_50.groupby("CNEC")["RAM_Neg"].mean().nlargest(number)
lowest_neg_ram_NPperf = ram_data_NP_perf.groupby("CNEC")["RAM_Neg"].mean().nlargest(number)

# Combine into one DataFrame
combined_neg = pd.concat([
    lowest_neg_ram_PTC50.rename("PTC50_ram_neg"),
    lowest_neg_ram_PTCperf.rename("PTCperf_ram_neg"),
    lowest_neg_ram_PTC300.rename("PTC300_ram_neg"),
    lowest_neg_ram_NP50.rename("NP50_ram_neg"),
    lowest_neg_ram_NPperf.rename("NPperf_ram_neg"),

], axis=1).dropna()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

combined_pos.plot(kind='bar', ax=axes[0])
axes[0].set_title("Top 10 CNECs with Lowest Avg RAM_Pos")
axes[0].set_ylabel("Avg RAM_Pos (MW)")
axes[0].grid(True)

combined_neg.plot(kind='bar', ax=axes[1])
axes[1].set_title("Top 10 CNECs with Lowest Avg RAM_Neg")
axes[1].set_ylabel("Avg RAM_Neg (MW)")
axes[1].grid(True)

plt.tight_layout()
plt.show()
dual_var_ram_pos_PTC300 = load_csv(f"../D-1_MC_fixed_PTC_results_300_epochs/{Number_of_hours}_hours/d_1_dual_variable_ram_pos.csv")
dual_var_ram_neg_PTC300 = load_csv(f"../D-1_MC_fixed_PTC_results_300_epochs/{Number_of_hours}_hours/d_1_dual_variable_ram_neg.csv")
dual_var_ram_pos_PTC50 = load_csv(f"../D-1_MC_fixed_PTC_results_50_epochs/{Number_of_hours}_hours/d_1_dual_variable_ram_pos.csv")
dual_var_ram_neg_PTC50 = load_csv(f"../D-1_MC_fixed_PTC_results_50_epochs/{Number_of_hours}_hours/d_1_dual_variable_ram_neg.csv")
dual_var_ram_pos_NP50 = load_csv(f"../D-1_MC_fixed_NP_results_flat/{Number_of_hours}_hours/d_1_dual_variable_ram_pos.csv")
dual_var_ram_neg_NP50 = load_csv(f"../D-1_MC_fixed_NP_results_flat/{Number_of_hours}_hours/d_1_dual_variable_ram_neg.csv")
dual_var_ram_pos_NPperf = load_csv(f"../D-1_MC_perfect_NP_results_flat/{Number_of_hours}_hours/d_1_dual_variable_ram_pos.csv")
dual_var_ram_neg_NPperf = load_csv(f"../D-1_MC_perfect_NP_results_flat/{Number_of_hours}_hours/d_1_dual_variable_ram_neg.csv")
dual_var_ram_pos_PTCperf = load_csv(f"../D-1_MC_perfect_PTC_results/{Number_of_hours}_hours/d_1_dual_variable_ram_pos.csv")
dual_var_ram_neg_PTCperf = load_csv(f"../D-1_MC_perfect_PTC_results/{Number_of_hours}_hours/d_1_dual_variable_ram_neg.csv")
#remove row if mean is 0
dual_var_ram_pos_PTC300_mean = dual_var_ram_pos_PTC300.mean(axis=0)

dual_var_ram_neg_PTCperf_mean = dual_var_ram_neg_PTCperf.mean(axis=0)

dual_var_ram_pos_PTCperf_mean = dual_var_ram_pos_PTCperf.mean(axis=0)

dual_var_ram_neg_PTC300_mean = dual_var_ram_neg_PTC300.mean(axis=0)

dual_var_ram_pos_NPperf_mean = dual_var_ram_pos_NPperf.mean(axis=0)
dual_var_ram_neg_NPperf_mean = dual_var_ram_neg_NPperf.mean(axis=0)
dual_var_ram_pos_PTC50_mean = dual_var_ram_pos_PTC50.mean(axis=0)
dual_var_ram_neg_PTC50_mean = dual_var_ram_neg_PTC50.mean(axis=0)
dual_var_ram_neg_NP50_mean = dual_var_ram_neg_NP50.mean(axis=0)
dual_var_ram_pos_NP50_mean = dual_var_ram_pos_NP50.mean(axis=0)


# Remove rows with all zeros

dual_var_ram_pos_PTC300_mean.index = dual_var_ram_pos_PTC300_mean.index.astype(int)
dual_var_ram_neg_PTCperf_mean.index = dual_var_ram_neg_PTCperf_mean.index.astype(int)
dual_var_ram_pos_PTCperf_mean.index = dual_var_ram_pos_PTCperf_mean.index.astype(int)
dual_var_ram_neg_PTC300_mean.index = dual_var_ram_neg_PTC300_mean.index.astype(int)
dual_var_ram_pos_NPperf_mean.index = dual_var_ram_pos_NPperf_mean.index.astype(int)
dual_var_ram_neg_NPperf_mean.index = dual_var_ram_neg_NPperf_mean.index.astype(int)
dual_var_ram_pos_PTC50_mean.index = dual_var_ram_pos_PTC50_mean.index.astype(int)
dual_var_ram_neg_PTC50_mean.index = dual_var_ram_neg_PTC50_mean.index.astype(int)
dual_var_ram_neg_NP50_mean.index = dual_var_ram_neg_NP50_mean.index.astype(int)
dual_var_ram_pos_NP50_mean.index = dual_var_ram_pos_NP50_mean.index.astype(int)
# Plotting RAM_Pos and Dual Variables with dual y-axes

dual_pos_df = pd.DataFrame({
    "PTCperf_dual": dual_var_ram_pos_PTCperf_mean,
    "PTC300_dual": dual_var_ram_pos_PTC300_mean,
    "NPperf_dual": dual_var_ram_pos_NPperf_mean,
    "PTC50_dual": dual_var_ram_pos_PTC50_mean,
    "NP50_dual": dual_var_ram_pos_NP50_mean
})
dual_neg_df = pd.DataFrame({
    "PTCperf_dual": dual_var_ram_neg_PTCperf_mean,
    "PTC300_dual": dual_var_ram_neg_PTC300_mean,
    "NPperf_dual": dual_var_ram_neg_NPperf_mean,
    "PTC50_dual": dual_var_ram_neg_PTC50_mean,
    "NP50_dual": dual_var_ram_neg_NP50_mean
})
combined_pos = combined_pos.join(dual_pos_df, how='inner')
combined_neg = combined_neg.join(dual_neg_df, how='inner')





fig, axes = plt.subplots(1, 2, figsize=(18, 7))

width = 0.35
x_pos = range(len(combined_pos.index))
x_neg = range(len(combined_neg.index))

# RAM_Pos
ax1 = axes[0]
ax1.bar(x_pos, combined_pos["PTCperf_RAM_Pos"], width=width, label="RAM_Pos PTCperf", color='lightblue')
ax1.bar([p + width for p in x_pos], combined_pos["PTC300_RAM_Pos"], width=width, label="RAM_Pos PTC300", color='lightgreen')
#ax1.bar([p + 2 * width for p in x_pos], combined_pos["NPperf_RAM_Pos"], width=width, label="RAM_Pos NPperf", color='lightyellow')
ax1.bar([p + 2 * width for p in x_pos], combined_pos["PTC50_RAM_Pos"], width=width, label="RAM_Pos PTC50", color='lightcoral')
ax1.set_xlabel("CNEC")
ax1.set_ylabel("RAM_Pos (MW)")
ax1.set_xticks([p + width / 2 for p in x_pos])
ax1.set_xticklabels(combined_pos.index)
ax1.set_title("RAM_Pos and Dual Variables")
ax1.grid(True)

ax1b = ax1.twinx()
ax1b.plot([p + width / 4 for p in x_pos], combined_pos["PTCperf_dual"], 'o--', color='blue', label="Dual PTCperf")
ax1b.plot([p + 3 * width / 4 for p in x_pos], combined_pos["PTC300_dual"], 'x--', color='green', label="Dual PTC300")
#ax1b.plot([p + 5 * width / 4 for p in x_pos], combined_pos["NPperf_dual"], 's--', color='orange', label="Dual NPperf")
ax1b.plot([p + 5 * width / 4 for p in x_pos], combined_pos["PTC50_dual"], 'd--', color='red', label="Dual PTC50")
ax1b.set_ylabel("Dual Variable (€/MW)")

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax1b.get_legend_handles_labels()
ax1b.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

# RAM_Neg
ax2 = axes[1]
ax2.bar(x_neg, combined_neg["PTCperf_ram_neg"], width=width, label="RAM_Neg PTCperf", color='lightcoral')
ax2.bar([p + width for p in x_neg], combined_neg["PTC300_ram_neg"], width=width, label="RAM_Neg PTC300", color='salmon')
#ax2.bar([p + 2 * width for p in x_neg], combined_neg["NPperf_ram_neg"], width=width, label="RAM_Neg NPperf", color='lightpink')
ax2.bar([p + 2 * width for p in x_neg], combined_neg["PTC50_ram_neg"], width=width, label="RAM_Neg PTC50", color='lightblue')
ax2.set_xlabel("CNEC")
ax2.set_ylabel("RAM_Neg (MW)")
ax2.set_xticks([p + width / 2 for p in x_neg])
ax2.set_xticklabels(combined_neg.index)
ax2.set_title("RAM_Neg and Dual Variables")
ax2.grid(True)

ax2b = ax2.twinx()
ax2b.plot([p + width / 4 for p in x_neg], combined_pos["PTCperf_dual"], 'o--', color='darkorange', label="Dual PTCperf")
ax2b.plot([p + 3 * width / 4 for p in x_neg], combined_pos["PTC300_dual"], 'x--', color='red', label="Dual PTC300")
#ax2b.plot([p + 5 * width / 4 for p in x_neg], combined_pos["NPperf_dual"], 's--', color='darkred', label="Dual NPperf")
ax2b.plot([p + 5 * width / 4 for p in x_neg], combined_pos["PTC50_dual"], 'd--', color='darkblue', label="Dual PTC50")
ax2b.set_ylabel("Dual Variable (€/MW)")

# Combine legends
lines_3, labels_3 = ax2.get_legend_handles_labels()
lines_4, labels_4 = ax2b.get_legend_handles_labels()

ax2b.legend(lines_3 + lines_4, labels_3 + labels_4, loc='lower right')

plt.tight_layout()
plt.show()


# %%
PTC50_color = 'lightblue'
PTC300_color = 'skyblue'
PTCperf_color = 'blue'
NPperf_color = 'red'
NP50_color = 'salmon'

dual_sorted_ptc300 = dual_var_ram_pos_PTC300_mean[dual_var_ram_pos_PTC300_mean != 0].sort_values(ascending=False)
ram_sorted_ptc300 = ram_data_PTC_300.groupby("CNEC")["RAM_Pos"].mean().loc[dual_sorted_ptc300.index]

dual_sorted_ptcperf = dual_var_ram_pos_PTCperf_mean[dual_var_ram_pos_PTCperf_mean != 0].sort_values(ascending=False)
ram_sorted_ptcperf = ram_data_PTC_perf.groupby("CNEC")["RAM_Pos"].mean().loc[dual_sorted_ptcperf.index]

dual_sorted_NPperf = dual_var_ram_pos_NPperf_mean[dual_var_ram_pos_NPperf_mean != 0].sort_values(ascending=False)
ram_sorted_NPperf = ram_data_NP_perf.groupby("CNEC")["RAM_Pos"].mean().loc[dual_sorted_NPperf.index]

dual_sorted_ptc50 = dual_var_ram_pos_PTC50_mean[dual_var_ram_pos_PTC50_mean != 0].sort_values(ascending=False)
ram_sorted_ptc50 = ram_data_PTC_50.groupby("CNEC")["RAM_Pos"].mean().loc[dual_sorted_ptc50.index]

dual_sorted_NP50 = dual_var_ram_pos_NP50_mean[dual_var_ram_pos_NP50_mean != 0].sort_values(ascending=False)
ram_sorted_NP50 = ram_data_NP_50.groupby("CNEC")["RAM_Pos"].mean().loc[dual_sorted_NP50.index]

# Align common CNECs for fair comparison
common_cnecs = dual_sorted_ptc300.index.intersection(dual_sorted_ptcperf.index)
#take the last four in common_cnecs
common_cnecs = common_cnecs[-4:]

dual_sorted_ptc300 = dual_sorted_ptc300.loc[common_cnecs]
ram_sorted_ptc300 = ram_sorted_ptc300.loc[common_cnecs]

dual_sorted_ptcperf = dual_sorted_ptcperf.loc[common_cnecs]
ram_sorted_ptcperf = ram_sorted_ptcperf.loc[common_cnecs]

dual_sorted_NPperf = dual_sorted_NPperf.loc[common_cnecs]
ram_sorted_NPperf = ram_sorted_NPperf.loc[common_cnecs]

dual_sorted_ptc50 = dual_sorted_ptc50.loc[common_cnecs]
ram_sorted_ptc50 = ram_sorted_ptc50.loc[common_cnecs]

dual_sorted_NP50 = dual_sorted_NP50.loc[common_cnecs]
ram_sorted_NP50 = ram_sorted_NP50.loc[common_cnecs]
# Plot
fig, ax1 = plt.subplots(figsize=(14, 7))
x = range(len(common_cnecs))

bar1 = ax1.bar([i - 0.4 for i in x], ram_sorted_ptc50.values, width=0.2, label='RAM PTC50', color= PTC50_color)
bar2 = ax1.bar([i - 0.2 for i in x], ram_sorted_ptc300.values, width=0.2, label='RAM PTC300', color= PTC300_color)
bar3 = ax1.bar([i - 0. for i in x], ram_sorted_ptcperf.values, width=0.2, label='RAM PTCperf', color= PTCperf_color)
bar4 = ax1.bar([i + 0.2 for i in x], ram_sorted_NPperf.values, width=0.2, label='RAM NPperf', color= NPperf_color)
bar5 = ax1.bar([i + 0.4 for i in x], ram_sorted_NP50.values, width=0.2, label='RAM NP50', color= NP50_color)

ax1.set_ylabel("RAM (MW)", fontsize=12)
ax1.set_xlabel("CNEC", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(common_cnecs, rotation=45)


ax2 = ax1.twinx()
line1 = ax2.plot(x, dual_sorted_ptc50.values, 'o--', color=PTC50_color, label='Dual PTC50')
line2 = ax2.plot(x, dual_sorted_ptc300.values, 's--', color=PTC300_color, label='Dual PTC300')
line3 = ax2.plot(x, dual_sorted_ptcperf.values, 'd--', color=   PTCperf_color, label='Dual PTCperf')
line4 = ax2.plot(x, dual_sorted_NPperf.values, 'x--', color= NPperf_color, label='Dual NPperf')
line5 = ax2.plot(x, dual_sorted_NP50.values, 'v--', color= NP50_color, label='Dual NP50')
ax2.set_xlabel("CNEC", fontsize=12)
ax2.set_ylabel("Dual Variable (€/MW)", fontsize=12)
ax2.grid(True)

# Combine legends
lines_labels, lines_text = ax2.get_legend_handles_labels()
bars_labels, bars_text = ax1.get_legend_handles_labels()
ax2.legend(bars_labels + lines_labels, bars_text + lines_text, loc='upper right')

plt.title("RAM vs Dual Variables (PTC300 vs PTCperf, Sorted by Dual PTC300)", fontsize=14)
plt.tight_layout()
plt.show()
# %%
# Sort by dual PTC300
dual_sorted_ptc300 = dual_var_ram_neg_PTC300_mean[dual_var_ram_neg_PTC300_mean != 0].sort_values(ascending=False)
ram_sorted_ptc300 = ram_data_PTC_300.groupby("CNEC")["RAM_Neg"].mean().loc[dual_sorted_ptc300.index]

dual_sorted_ptcperf = dual_var_ram_neg_PTCperf_mean[dual_var_ram_neg_PTCperf_mean != 0].sort_values(ascending=False)
ram_sorted_ptcperf = ram_data_PTC_perf.groupby("CNEC")["RAM_Neg"].mean().loc[dual_sorted_ptcperf.index]

dual_sorted_NPperf = dual_var_ram_neg_NPperf_mean[dual_var_ram_neg_NPperf_mean != 0].sort_values(ascending=False)
ram_sorted_NPperf = ram_data_NP_perf.groupby("CNEC")["RAM_Neg"].mean().loc[dual_sorted_NPperf.index]

dual_sorted_ptc50 = dual_var_ram_neg_PTC50_mean[dual_var_ram_neg_PTC50_mean != 0].sort_values(ascending=False)
ram_sorted_ptc50 = ram_data_PTC_50.groupby("CNEC")["RAM_Neg"].mean().loc[dual_sorted_ptc50.index]

dual_sorted_NP50 = dual_var_ram_neg_NP50_mean[dual_var_ram_neg_NP50_mean != 0].sort_values(ascending=False)
ram_sorted_NP50 = ram_data_NP_50.groupby("CNEC")["RAM_Neg"].mean().loc[dual_sorted_NP50.index]



# Align common CNECs for fair comparison
common_cnecs = dual_sorted_ptc300.index.intersection(dual_sorted_ptcperf.index)

dual_sorted_ptc300 = dual_sorted_ptc300.loc[common_cnecs]
ram_sorted_ptc300 = ram_sorted_ptc300.loc[common_cnecs]
dual_sorted_ptcperf = dual_sorted_ptcperf.loc[common_cnecs]
ram_sorted_ptcperf = ram_sorted_ptcperf.loc[common_cnecs]
dual_sorted_NPperf = dual_sorted_NPperf.loc[common_cnecs]
ram_sorted_NPperf = ram_sorted_NPperf.loc[common_cnecs]
dual_sorted_ptc50 = dual_sorted_ptc50.loc[common_cnecs]
ram_sorted_ptc50 = ram_sorted_ptc50.loc[common_cnecs]
dual_sorted_NP50 = dual_sorted_NP50.loc[common_cnecs]
ram_sorted_NP50 = ram_sorted_NP50.loc[common_cnecs]
# Plot
fig, ax1 = plt.subplots(figsize=(14, 7))
x = range(len(common_cnecs))

bar4 = ax1.bar([i - 0.4 for i in x], ram_sorted_ptc50.values, width=0.20, label='RAM_Neg PTC50', color= PTC50_color)
bar1 = ax1.bar([i - 0.2 for i in x], ram_sorted_ptc300.values, width=0.2, label='RAM_Neg PTC300', color= PTC300_color)
bar2 = ax1.bar([i - 0. for i in x], ram_sorted_ptcperf.values, width=0.2, label='RAM_Neg PTCperf', color= PTCperf_color)
bar3 = ax1.bar([i + 0.2 for i in x], ram_sorted_NPperf.values, width=0.2, label='RAM_Neg NPperf', color= NPperf_color)
bar5 = ax1.bar([i + 0.4 for i in x], ram_sorted_NP50.values, width=0.2, label='RAM_Neg NP50', color= NP50_color)
ax1.set_ylabel("RAM_Neg (MW)", fontsize=12)
ax1.set_xlabel("CNEC", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(common_cnecs, rotation=45)

ax2 = ax1.twinx()
line1 = ax2.plot(x, dual_sorted_ptc300.values, 'o--', color=PTC300_color, label='Dual PTC300')
line2 = ax2.plot(x, dual_sorted_ptcperf.values, 's--', color=PTCperf_color, label='Dual PTCperf')
line3 = ax2.plot(x, dual_sorted_NPperf.values, 'd--', color= NPperf_color, label='Dual NPperf')
line4 = ax2.plot(x, dual_sorted_ptc50.values, 'x--', color= PTC50_color, label='Dual PTC50')
line5 = ax2.plot(x, dual_sorted_NP50.values, 'v--', color= NP50_color, label='Dual NP50')
ax2.set_xlabel("CNEC", fontsize=12)
ax2.set_ylabel("Dual Variable (€/MW)", fontsize=12)
ax2.grid(True)

# Combine legends
lines_labels, lines_text = ax2.get_legend_handles_labels()
bars_labels, bars_text = ax1.get_legend_handles_labels()
ax2.legend(bars_labels + lines_labels, bars_text + lines_text, loc='upper right')

plt.title("RAM_Neg vs Dual Variables (PTC300 vs PTCperf, Sorted by Dual PTC300)", fontsize=14)
plt.tight_layout()
plt.show()
# %%
ram_pos_PTC_perf_std_on_CNEC_avg = ram_data_PTC_perf.groupby("CNEC")["RAM_Pos"].std().sort_values(ascending=False).mean()
ram_pos_PTC_300_std_on_CNEC_avg = ram_data_PTC_300.groupby("CNEC")["RAM_Pos"].std().sort_values(ascending=False).mean()
ram_pos_NP_perf_std_on_CNEC_avg = ram_data_NP_perf.groupby("CNEC")["RAM_Pos"].std().sort_values(ascending=False).mean()
ram_pos_PTC_50_std_on_CNEC_avg = ram_data_PTC_50.groupby("CNEC")["RAM_Pos"].std().sort_values(ascending=False).mean()

print("Average Standard Deviation of RAM_Pos on CNEC:")
print("PTC_perf:", ram_pos_PTC_perf_std_on_CNEC_avg)
print("PTC_300:", ram_pos_PTC_300_std_on_CNEC_avg)
print("NP_perf:", ram_pos_NP_perf_std_on_CNEC_avg)
print("PTC_50:", ram_pos_PTC_50_std_on_CNEC_avg)

# %%
#based on common CNECs make a table of the average standard deviation of RAM_Pos and RAM_Neg for each scenario and average value of thos RAMs

# Filter the RAM datasets to only include common CNECs with dual activity
dual_sorted_NPperf_pos = dual_var_ram_pos_NPperf_mean[dual_var_ram_pos_NPperf_mean != 0].sort_values(ascending=False)#.nlargest(1)
CNE_with_dual_activity_pos = dual_sorted_NPperf_pos.index
dual_sorted_NPperf_neg = dual_var_ram_neg_NPperf_mean[dual_var_ram_neg_NPperf_mean != 0].sort_values(ascending=False)#.nlargest(1)
CNE_with_dual_activity_neg = dual_sorted_NPperf_neg.index

def compute_ram_stats(ram_df, cnecs_pos, cnecs_neg, pos_col="RAM_Pos", neg_col="RAM_Neg"):
    df_filtered_pos = ram_df[ram_df["CNEC"].isin(cnecs_pos)]
    pos_std = df_filtered_pos.groupby("CNEC")[pos_col].std().median()
    pos_mean = df_filtered_pos.groupby("CNEC")[pos_col].mean().median()

    df_filtered_neg = ram_df[ram_df["CNEC"].isin(cnecs_neg)]
    neg_std = df_filtered_neg.groupby("CNEC")[neg_col].std().median()
    neg_mean = df_filtered_neg.groupby("CNEC")[neg_col].mean().median()
    return pos_mean, pos_std, neg_mean, neg_std

# Compute stats for each scenario
scenarios = {
    "PTCperf": ram_data_PTC_perf,
    "PTC300": ram_data_PTC_300,
    "PTC50": ram_data_PTC_50,
    "NPperf": ram_data_NP_perf
}
ram_stats = {}
for name, df in scenarios.items():
    pos_mean, pos_std, neg_mean, neg_std = compute_ram_stats(df, CNE_with_dual_activity_pos, CNE_with_dual_activity_neg)
    ram_stats[name] = {
        "Avg RAM_Pos": pos_mean,
        "Std Dev RAM_Pos": pos_std,
        "Avg RAM_Neg": neg_mean,
        "Std Dev RAM_Neg": neg_std
    }
# Create DataFrame from the stats
ram_stats_df = pd.DataFrame(ram_stats).T
# Round the DataFrame to 3 decimal places
#ram_stats_df = ram_stats_df.round(3)
# Display the DataFrame
print("Average Standard Deviation of RAM_Pos and RAM_Neg for each scenario:")
print(ram_stats_df)
# %%
