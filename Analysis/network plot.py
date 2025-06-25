#%% Import libraries and load data
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os
import numpy as np

Number_of_days = 30*6
Number_of_hours = 24 * Number_of_days
# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))

bus_data_path = os.path.join("..", "data", "df_bus_final.csv")
bus_data_path = os.path.abspath(os.path.join(script_dir, bus_data_path))

branch_data_path = os.path.join("..", "data", "df_branch_final.csv")
branch_data_path = os.path.abspath(os.path.join(script_dir, branch_data_path))

bus_data = pd.read_csv(bus_data_path)
branch_data = pd.read_csv(branch_data_path)
# Create the network graph
G = nx.MultiGraph()  # Use MultiGraph to allow multiple edges

# Add nodes with positions and zones
for _, row in bus_data.iterrows():
    G.add_node(row["BusID"], pos=(row["Longitude"], row["Latitude"]), zone=row["Zone"])

# Add edges with attributes
for _, row in branch_data.iterrows():
    G.add_edge(row["FromBus"], row["ToBus"], ID=int(row["BranchID"]), resistance=row["R"], reactance=row["X"])

# Get positions for the nodes
positions = nx.get_node_attributes(G, "pos")

# Map zones to colors
unique_zones= ['1','2','3', 'Import/Export_1', 'Import/Export_2',
       'Import/Export_3']
zone_color_map = {zone: color for zone, color in zip(unique_zones, plt.cm.tab20.colors)}  # Use a colormap

# Assign colors to nodes based on their zone
node_colors = [zone_color_map[G.nodes[node]["zone"]] for node in G.nodes]

ML_prediction_path = os.path.join(
    "..", "ML_results", f"{Number_of_hours}_hours", "predictions_NP.csv")
ML_prediction_path = os.path.abspath(os.path.join(script_dir, ML_prediction_path))

ML_prediction = pd.read_csv(ML_prediction_path, index_col=0)


# %%
from scipy.spatial import ConvexHull
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib


def plot_network(
    G, 
    bus_data, 
    positions, 
    show_branch_ids=False, 
    edge_colors=None, 
    filter_branch_ids=None,
    custom_title=None,
    cbar_label=None
):
    """
    Plots a network graph with zones, optional branch ID boxes, and customizable edge colors.
    
    Parameters:
        G (nx.MultiGraph): The network graph.
        bus_data (pd.DataFrame): DataFrame containing bus information with 'Zone', 'Longitude', and 'Latitude'.
        positions (dict): Dictionary of node positions.
        show_branch_ids (bool): Whether to show boxes with branch IDs between nodes.
        edge_colors (dict): Dictionary mapping BranchID to colors (e.g., {BranchID: "color"}). Default is black.
        filter_branch_ids (list): List of BranchIDs to include in the plot. If None, all branches are plotted.
    """
    plt.figure(figsize=(20, 16))
    ax = plt.gca()

    #apply a background color
    ax.set_facecolor('lightgrey')

    # Default edge color is black
    default_edge_color = "black"
    if edge_colors is not None:
        edge_colors['BranchID'] = edge_colors.index
    # Get unique zones and their points
    zone_colors = plt.cm.tab20.colors  # Use colormap for zones
    zone_groups = bus_data.groupby("Zone")

    for idx, (zone, group) in enumerate(zone_groups):
        # Get the points for the zone
        points = group[["Longitude", "Latitude"]].values
        if len(points) > 2:  # ConvexHull requires at least 3 points
            hull = ConvexHull(points)
            polygon_points = points[hull.vertices]
            # Create a polygon for the convex hull
            poly = Polygon(
                polygon_points,
                closed=True,
                facecolor='black',  # Fill color
                edgecolor=None,
                alpha=0.25,  # Transparency
                label=f"Zone {zone}",
            )
            ax.add_patch(poly)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos=positions, node_size=300, node_color=node_colors)
    nx.draw_networkx_labels(G, pos=positions, font_size=10)

    # Group branch IDs by node connections
    edge_labels = {}
    for u, v, key, edge_data in G.edges(data=True, keys=True):
        connection = tuple(sorted((u, v)))  # Sort nodes to avoid duplicates
        if connection not in edge_labels:
            edge_labels[connection] = []
        edge_labels[connection].append(edge_data["ID"])
    
    # Draw edges, applying curvature for multiple lines
    for u, v, key, edge_data in G.edges(data=True, keys=True):
        # Skip edge if it's not in the filter list (if filter is specified)
        branch_id = edge_data["ID"]
        if filter_branch_ids and branch_id not in filter_branch_ids:
            continue

        num_edges = len(G[u][v]) if G.is_multigraph() else 1
        rad = 0.2 * (key - 1) if num_edges > 1 else 0

        # Determine edge color based on the values in the edge_colors dataframe, which is branch_id as index and values
        #so values should be converted to a color map based on the min and max values and the color map
        if edge_colors is not None:
            min_value = edge_colors.iloc[:, 0].min()
            max_value = edge_colors.iloc[:, 0].max()


            # Normalize the value to be between 0 and 1
            norm = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)
            mapper = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.OrRd)
            color = mapper.to_rgba(edge_colors[edge_colors['BranchID'] == str(branch_id)].values[0][0])
        else:
            color = default_edge_color
        # Draw edge
        nx.draw_networkx_edges(
            G, pos=positions, edgelist=[(u, v)], connectionstyle=f"arc3,rad={rad}", edge_color=color, width=1.0
        )
    

    




    # Plot boxes with branch IDs at the midpoint between nodes
    if show_branch_ids:
        for (u, v), branch_ids in edge_labels.items():
            # Skip edge if it's not in the filter list (if filter is specified)
            if filter_branch_ids and not any(b in filter_branch_ids for b in branch_ids):
                continue

            x1, y1 = positions[u]
            x2, y2 = positions[v]
            midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
            # Add a box with the branch IDs
            plt.text(
                midpoint[0],
                midpoint[1],
                "\n".join(map(str, branch_ids)),  # Join branch IDs with a newline
                fontsize=6,
                color="black",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="grey", boxstyle="round", pad=0.3),
            )

    # Add legend for zones
    handles = [patches.Patch(color=zone_colors[idx % len(zone_colors)], label=f"Zone {zone}") for idx, zone in enumerate(zone_groups.groups.keys())]
    ax.legend(handles=handles, loc="lower right", title="Zones")

    if custom_title:
        plt.title(custom_title, fontsize=32)
    else:
        plt.title("Network Plot of Buses and Branches (Colored by Zone, MultiGraph)")
    #plt.xlabel("Longitude", fontsize=16)
    #plt.ylabel("Latitude", fontsize=16)
    if edge_colors is not None:
        #add color gradient legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.OrRd,norm =norm)
        sm._A = []
        cax = plt.axes([0.95, 0.2, 0.01, 0.5])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label, rotation=270, labelpad=10, fontsize=24)
        cbar.set_ticks([min_value, max_value])

        cbar.set_ticklabels([f'{min_value:.2f}', f'{max_value:.2f}'])
    
    plt.show()

#%%
plot_network(G, bus_data, positions, show_branch_ids=None,custom_title='Network in D-2 CGM', 
             filter_branch_ids=None)
# %%

##########################################################
### Plotting the network with only CNEC lines included ###
###            and the GSK is flat or pmax sub         ###
##########################################################
#Number_of_hours = 720*1
PTDF_Z_CNE_path_flat = os.path.join(
    "..", "D-1_MC_fixed_NP_results_flat", f"{Number_of_hours}_hours", "PTDF_Z_CNEC.csv")
PTDF_Z_CNE_path_flat = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path_flat))
PTDF_Z_CNE_flat = pd.read_csv(PTDF_Z_CNE_path_flat, index_col=0)
CNEC_lines_list_flat = (PTDF_Z_CNE_flat.index).to_list()



plot_network(G, bus_data, positions, show_branch_ids=None,custom_title='Network in D-1 MC with flat GSK',
             filter_branch_ids=CNEC_lines_list_flat)

#%%
#############################################
### Plotting the flow differences between ###
### CGM_D-2 and CGM_D-1 for fixed NP      ###
### with flat                             ###
#############################################

#load the flows
Number_of_days = 30*6
Number_of_hours = 24 * Number_of_days

line_f_CGM_D_2_fixed_NP_flat_path = os.path.join(
    "..", "D-2_base_case_fixed_NP_results_flat", f"{Number_of_hours}_hours", "df_line_f.csv")
line_f_CGM_D_2_fixed_NP_flat_path = os.path.abspath(os.path.join(script_dir, line_f_CGM_D_2_fixed_NP_flat_path))
line_f_CGM_D_2_fixed_NP_flat = pd.read_csv(line_f_CGM_D_2_fixed_NP_flat_path, index_col=0)
line_f_CGM_D_1_fixed_NP_flat_path = os.path.join(
    "..", "D-1_CGM_fixed_NP_results_flat", f"{Number_of_hours}_hours", "df_line_f.csv")
line_f_CGM_D_1_fixed_NP_flat_path = os.path.abspath(os.path.join(script_dir, line_f_CGM_D_1_fixed_NP_flat_path))
line_f_CGM_D_1_fixed_NP_flat = pd.read_csv(line_f_CGM_D_1_fixed_NP_flat_path, index_col=0)
line_f_diff_flat = line_f_CGM_D_1_fixed_NP_flat - line_f_CGM_D_2_fixed_NP_flat
line_f_diff_abs_flat = line_f_diff_flat.abs()
#only use the index from ML_prediction
line_f_diff_abs_flat = line_f_diff_abs_flat.reindex(index=ML_prediction.index)
line_f_diff_mean_per_line_abs_flat = line_f_diff_abs_flat.mean(axis=0)
# Ensure all values are numeric
line_f_diff_mean_per_line_abs_flat = pd.to_numeric(line_f_diff_mean_per_line_abs_flat, errors='coerce')
# Drop any NaN values if they exist after conversion
line_f_diff_mean_per_line_abs_flat = line_f_diff_mean_per_line_abs_flat.dropna()
line_f_diff_mean_per_line_abs_flat = line_f_diff_mean_per_line_abs_flat.to_frame(name='value')
line_f_diff_mean_per_line_abs_flat['BranchID'] = line_f_diff_mean_per_line_abs_flat.index.astype(str).str.strip()
plot_network(G, bus_data, positions, show_branch_ids=False,custom_title='Flow difference between D-1 CGM and D-2 CGM in fixed NP with flat GSK',
                edge_colors=line_f_diff_mean_per_line_abs_flat, cbar_label='Mean flow difference (MW)')

#%%
#############################################
### Plotting the flow differences between ###
### CGM_D-2 and CGM_D-1 for fixed PTC     ###
### with flat GSK                         ###
#############################################

#load the flows
Number_of_days = 30*6
Number_of_hours = 24 * Number_of_days

line_f_CGM_D_2_fixed_PTC_flat_path = os.path.join(
    "..", "D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours", "df_line_f.csv")
line_f_CGM_D_2_fixed_PTC_flat_path = os.path.abspath(os.path.join(script_dir, line_f_CGM_D_2_fixed_PTC_flat_path))
line_f_CGM_D_2_fixed_PTC_flat = pd.read_csv(line_f_CGM_D_2_fixed_PTC_flat_path, index_col=0)
line_f_CGM_D_1_fixed_PTC_flat_path = os.path.join(
    "..", "D-1_CGM_fixed_PTC_results_flat_50_epochs", f"{Number_of_hours}_hours", "df_line_f.csv")
line_f_CGM_D_1_fixed_PTC_flat_path = os.path.abspath(os.path.join(script_dir, line_f_CGM_D_1_fixed_PTC_flat_path))
line_f_CGM_D_1_fixed_PTC_flat = pd.read_csv(line_f_CGM_D_1_fixed_PTC_flat_path, index_col=0)
line_f_diff_flat_PTC = line_f_CGM_D_1_fixed_PTC_flat - line_f_CGM_D_2_fixed_PTC_flat
line_f_diff_abs_flat_PTC = line_f_diff_flat_PTC.abs()
#only use the index from ML_prediction
line_f_diff_abs_flat_PTC = line_f_diff_abs_flat_PTC.reindex(index=ML_prediction.index)
line_f_diff_mean_per_line_abs_flat_PTC = line_f_diff_abs_flat_PTC.mean(axis=0)
# Ensure all values are numeric
line_f_diff_mean_per_line_abs_flat_PTC = pd.to_numeric(line_f_diff_mean_per_line_abs_flat_PTC, errors='coerce')
# Drop any NaN values if they exist after conversion
line_f_diff_mean_per_line_abs_flat_PTC = line_f_diff_mean_per_line_abs_flat_PTC.dropna()
line_f_diff_mean_per_line_abs_flat_PTC = line_f_diff_mean_per_line_abs_flat_PTC.to_frame(name='value')
line_f_diff_mean_per_line_abs_flat_PTC['BranchID'] = line_f_diff_mean_per_line_abs_flat_PTC.index.astype(str).str.strip()
plot_network(G, bus_data, positions, show_branch_ids=False,custom_title='Flow difference between D-1 CGM and D-2 CGM in fixed PTC with flat GSK',
                edge_colors=line_f_diff_mean_per_line_abs_flat_PTC, cbar_label='Mean flow difference (MW)')
#%%
###############################################
### Plotting the overloads on lines in the  ###
### fixed NP with flat GSK                  ###
###############################################

overload_fixed_NP_flat_path = os.path.join(
    "..", "D-1_CGM_fixed_NP_results_flat", f"{Number_of_hours}_hours", "overload_df.csv")
overload_fixed_NP_flat_path = os.path.abspath(os.path.join(script_dir, overload_fixed_NP_flat_path))
overload_fixed_NP_flat = pd.read_csv(overload_fixed_NP_flat_path, index_col=0)
number_of_overloads_per_line_fixed_NP_flat = pd.DataFrame(index=overload_fixed_NP_flat.columns, columns=['Number of overloads'])
for l in overload_fixed_NP_flat.columns:
    number_of_overloads_per_line_fixed_NP_flat.loc[l, 'Number of overloads'] = overload_fixed_NP_flat[l].count()
#give nuber of overloads per line index the heading 'BranchID'
number_of_overloads_per_line_fixed_NP_flat.index.name = 'BranchID'
plot_network(G, bus_data, positions, show_branch_ids=False,custom_title='System colored by overloads on lines in fixed NP with flat GSK',
             edge_colors=number_of_overloads_per_line_fixed_NP_flat, filter_branch_ids=None,
             cbar_label='Number of overloads per line')
#%%
###############################################
### Plotting the overloads on lines in the  ###
### fixed PTC with flat GSK                 ###
###############################################

overload_fixed_PTC_flat_path = os.path.join(
    "..", "D-1_CGM_fixed_PTC_results_flat_50_epochs", f"{Number_of_hours}_hours", "overload_df.csv")
overload_fixed_PTC_flat_path = os.path.abspath(os.path.join(script_dir, overload_fixed_PTC_flat_path))
overload_fixed_PTC_flat = pd.read_csv(overload_fixed_PTC_flat_path, index_col=0)
number_of_overloads_per_line_fixed_PTC_flat = pd.DataFrame(index=overload_fixed_PTC_flat.columns, columns=['Number of overloads'])
for l in overload_fixed_PTC_flat.columns:
    number_of_overloads_per_line_fixed_PTC_flat.loc[l, 'Number of overloads'] = overload_fixed_PTC_flat[l].count()
#give nuber of overloads per line index the heading 'BranchID'
number_of_overloads_per_line_fixed_PTC_flat.index.name = 'BranchID'
plot_network(G, bus_data, positions, show_branch_ids=False,custom_title='System colored by overloads on lines in fixed PTC with flat GSK',
                edge_colors=number_of_overloads_per_line_fixed_PTC_flat, filter_branch_ids=None,
                cbar_label='Number of overloads per line')



#%%
#############################################
### Plotting the FBME on lines in the     ###
### fixed NP with flat GSK                ###
#############################################

GSK = 'flat'
FBME_fixed_NP_flat_path = os.path.join(
    "..", "Analysis", f"FBME_{Number_of_hours}_hours_{GSK}_fixed_NP", f"FBME_{GSK}.parquet")
FBME_fixed_NP_flat_path = os.path.abspath(os.path.join(script_dir, FBME_fixed_NP_flat_path))
FBME_fixed_NP_flat = pd.read_parquet(FBME_fixed_NP_flat_path)
FBME_fixed_NP_flat_abs = FBME_fixed_NP_flat.abs()
FBME_mean_per_line_abs_flat = FBME_fixed_NP_flat_abs.mean(axis=0)
FBME_mean_per_line_abs_flat = pd.DataFrame(FBME_mean_per_line_abs_flat, columns=['FBME_sum_per_line_abs'])
#from number_of_overloads_per_line_df, assign the same number of rows to FBME_sum_per_line_abs_flat, but use values from FBME_sum_per_line_abs_flat
FBME_mean_per_line_abs_flat = FBME_mean_per_line_abs_flat.reindex(index=number_of_overloads_per_line_fixed_NP_flat.index)
#assign zero to nan values
FBME_mean_per_line_abs_flat = FBME_mean_per_line_abs_flat.fillna(0)
FBME_mean_per_line_abs_flat['BranchID'] = FBME_mean_per_line_abs_flat.index.astype(str).str.strip()
plot_network(G, bus_data, positions, show_branch_ids=False,custom_title='System colored by mean FBME on lines in fixed NP with flat GSK',
                edge_colors=FBME_mean_per_line_abs_flat, filter_branch_ids=CNEC_lines_list_flat,
                cbar_label='FBME mean per line (MW)')



#%%
#############################################
### Plotting the FBME on lines in the     ###
### fixed PTC with flat GSK               ###
#############################################

FBME_fixed_PTC_flat_path = os.path.join(
    "..", "Analysis", f"FBME_{Number_of_hours}_hours_flat_fixed_PTC_50_epochs", f"FBME_flat.parquet")
FBME_fixed_PTC_flat_path = os.path.abspath(os.path.join(script_dir, FBME_fixed_PTC_flat_path))
FBME_fixed_PTC_flat = pd.read_parquet(FBME_fixed_PTC_flat_path)
FBME_fixed_PTC_flat_abs = FBME_fixed_PTC_flat.abs()
FBME_mean_per_line_abs_flat_base_case = FBME_fixed_PTC_flat_abs.mean(axis=0)
FBME_mean_per_line_abs_flat_base_case = pd.DataFrame(FBME_mean_per_line_abs_flat_base_case, columns=['FBME_sum_per_line_abs'])
#from number_of_overloads_per_line_df, assign the same number of rows to FBME_sum_per_line_abs_flat_base_case, but use values from FBME_sum_per_line_abs_flat_base_case
FBME_mean_per_line_abs_flat_base_case = FBME_mean_per_line_abs_flat_base_case.reindex(index=number_of_overloads_per_line_fixed_PTC_flat.index)
#assign zero to nan values
FBME_mean_per_line_abs_flat_base_case = FBME_mean_per_line_abs_flat_base_case.fillna(0)
FBME_mean_per_line_abs_flat_base_case['BranchID'] = FBME_mean_per_line_abs_flat_base_case.index.astype(str).str.strip()
plot_network(G, bus_data, positions, show_branch_ids=False,custom_title='System colored by mean FBME on lines in base case with flat GSK',
                edge_colors=FBME_mean_per_line_abs_flat_base_case, filter_branch_ids=CNEC_lines_list_flat,
                cbar_label='FBME mean per line (MW)')
#%%
#############################################
### Plotting the FBME on lines in the     ###
### perfect PTC with flat GSK             ###
#############################################

FBME_perfect_PTC_flat_path = os.path.join(
    "..", "Analysis", f"FBME_{Number_of_hours}_hours_flat_perfect_PTC", f"FBME_flat.parquet")
FBME_perfect_PTC_flat_path = os.path.abspath(os.path.join(script_dir, FBME_perfect_PTC_flat_path))
FBME_perfect_PTC_flat = pd.read_parquet(FBME_perfect_PTC_flat_path)
FBME_perfect_PTC_flat_abs = FBME_perfect_PTC_flat.abs()
FBME_mean_per_line_abs_flat_perfect_case = FBME_perfect_PTC_flat_abs.mean(axis=0)
FBME_mean_per_line_abs_flat_perfect_case = pd.DataFrame(FBME_mean_per_line_abs_flat_perfect_case, columns=['FBME_sum_per_line_abs'])
#from number_of_overloads_per_line_df, assign the same number of rows to FBME_sum_per_line_abs_flat_perfect_case, but use values from FBME_sum_per_line_abs_flat_perfect_case
FBME_mean_per_line_abs_flat_perfect_case = FBME_mean_per_line_abs_flat_perfect_case.reindex(index=number_of_overloads_per_line_fixed_PTC_flat.index)
#assign zero to nan values
FBME_mean_per_line_abs_flat_perfect_case = FBME_mean_per_line_abs_flat_perfect_case.fillna(0)
FBME_mean_per_line_abs_flat_perfect_case['BranchID'] = FBME_mean_per_line_abs_flat_perfect_case.index.astype(str).str.strip()
plot_network(G, bus_data, positions, show_branch_ids=False,custom_title='System colored by mean FBME on lines in perfect PTC with flat GSK',
                edge_colors=FBME_mean_per_line_abs_flat_perfect_case, filter_branch_ids=CNEC_lines_list_flat,
                cbar_label='FBME mean per line (MW)')

#%%
import seaborn as sns
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
    ram_data_baseline,
    on=["Time Step", "CNEC"],
    suffixes=("_PTC50", "_baseline")
)

# Calculate the differences
merged["RAM_Pos_Diff"] = merged["RAM_Pos_PTC50"] - merged["RAM_Pos_baseline"]
merged["RAM_Neg_Diff"] = merged["RAM_Neg_PTC50"] - merged["RAM_Neg_baseline"]

# Summary statistics
ram_pos_summary = merged["RAM_Pos_Diff"].describe()
ram_neg_summary = merged["RAM_Neg_Diff"].describe()

# Plotting distributions
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
#plot network for ram_pos_grouped
#make CNEC a column, and make index from 1 to 271, CNEC does not match with an index then set zero
#FBME_mean_per_line_abs_flat = FBME_fixed_NP_flat_abs.mean(axis=0)
#FBME_mean_per_line_abs_flat = pd.DataFrame(FBME_mean_per_line_abs_flat, columns=['FBME_sum_per_line_abs'])
#from number_of_overloads_per_line_df, assign the same number of rows to FBME_sum_per_line_abs_flat, but use values from FBME_sum_per_line_abs_flat
#FBME_mean_per_line_abs_flat = FBME_mean_per_line_abs_flat.reindex(index=number_of_overloads_per_line_fixed_NP_flat.index)
#assign zero to nan values
#FBME_mean_per_line_abs_flat = FBME_mean_per_line_abs_flat.fillna(0)
#FBME_mean_per_line_abs_flat['BranchID'] = FBME_mean_per_line_abs_flat.index.astype(str).str.strip()`

#ram_pos_grouped_for_plot_df = pd.DataFrame(ram_pos_grouped, columns=['mean'])
#ram_pos_grouped_for_plot_df = pd.DataFrame(ram_pos_grouped_for_plot_df, index=ram_pos_grouped_for_plot_df)

# Align df_ram_diff with df_overloads index, fill missing with 0
aligned_ram_diff = pd.DataFrame(index=number_of_overloads_per_line_fixed_NP_flat.index)
ram_pos_grouped = ram_pos_grouped.set_index('CNEC')
#make sure the index is a string
ram_pos_grouped.index = ram_pos_grouped.index.astype(str).str.strip()
aligned_ram_diff["RAM_Diff_Mean"] = ram_pos_grouped["mean"].reindex(number_of_overloads_per_line_fixed_NP_flat.index).fillna(0)


plot_network(G, bus_data, positions, show_branch_ids=False, custom_title='Network colored by RAM_Pos differences (PTC50 - PTCperf)',
                edge_colors=aligned_ram_diff, filter_branch_ids=CNEC_lines_list_flat,
                cbar_label='RAM_Pos difference (MW)')



#%%
ram_pos_grouped_for_plot_df = ram_pos_grouped_for_plot_df.reindex(index=number_of_overloads_per_line_fixed_NP_flat.index)

#%%


plot_network(G, bus_data, positions, show_branch_ids=False, custom_title='Network colored by RAM_Pos differences (PTC50 - PTCperf)',
                edge_colors=ram_pos_grouped_for_plot, filter_branch_ids=None,
                cbar_label='RAM_Pos difference (MW)')