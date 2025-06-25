

#%%

######################
###    PACKAGES    ###
######################
import os
import pandas as pd
import numpy as np


#############################
###    LOAD INPUT DATA    ###
#############################

# Set the current working path
current_path = os.getcwd()
data_path = os.path.join(current_path, 'data')

# Import data
df_bus_load = pd.read_csv(os.path.join(data_path, "df_bus_load_added_abroad_final.csv"))
df_bus = pd.read_csv(os.path.join(data_path, "df_bus_final.csv"))
df_branch = pd.read_csv(os.path.join(data_path, "df_branch_final.csv"))
df_plants = pd.read_csv(os.path.join(data_path, "df_gen_final.csv"), sep=";")
# Uncomment this line if you want to load the high renewable scenario
# df_plants = pd.read_csv(os.path.join(data_path, "df_gen_final_high_RES.csv"))
incidence = pd.read_csv(os.path.join(data_path, "matrix_A_final.csv"))
susceptance = pd.read_csv(os.path.join(data_path, "matrix_Bd_final.csv"))

# Load Renewable capacity factors
df_pv = pd.read_excel(os.path.join(data_path, "data_renew_2015.xlsx"), sheet_name="pv", header=0)
df_pv.columns = df_pv.columns.astype(str)
df_wind = pd.read_excel(os.path.join(data_path, "data_renew_2015.xlsx"), sheet_name="onshore", header=0)
df_wind.columns = df_wind.columns.astype(str)
df_wind_off = pd.read_excel(os.path.join(data_path, "data_renew_2015.xlsx"), sheet_name="offshore", header=0)
df_wind_off.columns = df_wind_off.columns.astype(str)




#%%

#########################
###    CREATE SETS    ###
#########################

# General sets
T = list(range(1, len(df_bus_load) + 1))  # Time steps

R = ["PV", "Wind", "Wind Offshore"]  # Renewable types

# Filter out plant IDs where the Type column is not in R
P = df_plants.loc[~df_plants["Type"].isin(R), "GenID"].tolist()

# Extract unique bus IDs
N = df_bus["BusID"].tolist()

# Extract unique branch IDs
L = df_branch["BranchID"].tolist()

# Extract sorted unique zones
Z = sorted(df_bus["Zone"].unique())

# Flow-based sets
Z_FBMC = Z[:len(Z)-3]  # First (length(Z) - 3) elements of Z
Z_not_in_FBMC = Z[len(Z)-3:]  # Last 3 elements of Z

# Nodes in Z_FBMC
N_FBMC = df_bus.loc[df_bus["Zone"].isin(Z_FBMC), "BusID"].tolist()

# Nodes not in Z_FBMC
N_not_in_FBMC = df_bus.loc[~df_bus["Zone"].isin(Z_FBMC), "BusID"].tolist()

# Print the sets for verification
print("Printing Sets and lists")
print("T (time steps):", T[:10])  # Print only the first 10 for brevity
print("R (renewables):", R)
print("P (non-renewable plants):", P[:10])  # Print only the first 10 for brevity
print("N (bus IDs):", N[:10])  # Print only the first 10 for brevity
print("L (branch IDs):", L[:10])  # Print only the first 10 for brevity
print("Z (zones):", Z)
print("Z_FBMC:", Z_FBMC)
print("Z_not_in_FBMC:", Z_not_in_FBMC)
print("N_FBMC (nodes in Z_FBMC):", N_FBMC[:10])  # Print only the first 10 for brevity
print("N_not_in_FBMC (nodes not in Z_FBMC):", N_not_in_FBMC[:10])  # Print only the first 10 for brevity

# This function is used to assign units to zones,
# in case new zone configurations are used in df_bus
def replaced_zones():
    # Create an empty list to store new zones
    zone_p_new = []
    
    # Loop through each OnBus value in df_plants
    for i in df_plants["OnBus"]:
        # Find the corresponding Zone in df_bus where BusID matches the current OnBus value
        matching_zone = df_bus.loc[df_bus["BusID"] == i, "Zone"].values
        
        # Append the first matching Zone to the list (if it exists)
        if len(matching_zone) > 0:
            zone_p_new.append(matching_zone[0])
        else:
            zone_p_new.append(None)  # Add None if no matching zone is found
    
    return zone_p_new


# Assign the result of the function to a new column 'Zone' in df_plants
df_plants["Zone"] = replaced_zones()

# Adjust capacities in the DataFrames
df_branch["Pmax"] = 0.5 * df_branch["Pmax"]
df_bus["ZoneRes"] = df_bus["Zone"]


# Redispatch: Filter plants based on type and zone
P_RD = df_plants.loc[
    (df_plants["Type"].isin(["Hard Coal", "Gas/CCGT"])) & 
    (df_plants["Zone"].isin(Z_FBMC)),
    "GenID"
].tolist()

# Check the resulting list of redispatchable plant IDs
print("P_RD (redispatchable plants):", P_RD[:10])  # Print the first 10 for verification



#########################################
###    CREATE MAPPING DICTIONARIES    ###
#########################################


# Mapping: Nodes in each zone
n_in_z = {z: [n for n in N if df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0] == z] for z in Z}

# Mapping: Plants at each node
p_at_n = {n: [p for p in P if df_plants.loc[df_plants["GenID"] == p, "OnBus"].iloc[0] == n] for n in N}

# Mapping: Redispatchable plants at each node
p_rd_at_n = {n: [p for p in P_RD if df_plants.loc[df_plants["GenID"] == p, "OnBus"].iloc[0] == n] for n in N}

# Mapping: Plants in each zone
p_in_z = {z: [p for p in P if df_plants.loc[df_plants["GenID"] == p, "Zone"].iloc[0] == z] for z in Z}

# Mapping: Zone-to-zone connections
z_to_z = {
    z: [
        zz for zz in Z
        if zz in df_bus.loc[
            (
                (df_bus["BusID"].isin(df_branch.loc[df_branch["FromBus"].isin(n_in_z[z]), "ToBus"])) |
                (df_bus["BusID"].isin(df_branch.loc[df_branch["ToBus"].isin(n_in_z[z]), "FromBus"]))
            ),
            "Zone"
        ].unique()
        and zz != z
    ]
    for z in Z
}

# Print the mappings for verification
print("n_in_z:", {k: v[:5] for k, v in n_in_z.items()})  # Print first 5 nodes per zone for brevity
print("p_at_n:", {k: v[:5] for k, v in p_at_n.items()})  # Print first 5 plants per node for brevity
print("p_rd_at_n:", {k: v[:5] for k, v in p_rd_at_n.items()})  # Print first 5 redispatchable plants per node for brevity
print("p_in_z:", {k: v[:5] for k, v in p_in_z.items()})  # Print first 5 plants per zone for brevity
print("z_to_z:", {k: v for k, v in z_to_z.items()})  # Print zone-to-zone connections


##############################
###    DEFINE FUNCTIONS    ###
##############################

# Function to get marginal cost for a plant
def get_mc(p):
    # Find the cost for the given plant ID
    cost = df_plants.loc[df_plants["GenID"] == p, "Costs"].iloc[0]
    return cost

# Function to find the maximum marginal cost for redispatchable plants
def find_maximum_mc():
    max_temp = 0
    for p in P_RD:  # P_RD is the list of redispatchable plants
        mc_temp = get_mc(p)
        if mc_temp > max_temp:
            max_temp = mc_temp
    return max_temp

#print("Marginal cost of plant 13:", get_mc(P_RD[0]))
#print("Maximum marginal cost among redispatchable plants:", find_maximum_mc())

# Function to get demand for a given time and node
def get_dem(t, n):
    # Access the demand for time step `t` and node `n` from df_bus_load
    return df_bus_load.loc[t, str(n)]

# Function to create the renewable energy resource table
# AVAILABLE RENEWABLE PRODUCTION
def create_res_table():
    
    # Initialize a 3D numpy array for the resource table
    res_temp = np.zeros((len(T), len(N), len(R)), dtype=float)
    
    # Loop through all nodes (n) and renewable types (r)
    for n in N:
        for r in R:
            # Get the zone for the current node
            zone_temp = df_bus.loc[df_bus["BusID"] == n, "ZoneRes"].iloc[0]
            
            # Calculate the total capacity of plants of type `r` at node `n`
            cap_temp = df_plants.loc[
                (df_plants["Type"] == r) & (df_plants["OnBus"] == n), "Pmax"
            ].sum()
            
            # Get the share for the current zone and renewable type
            if r == "PV":
                share_temp = df_pv[zone_temp]
            elif r == "Wind":
                share_temp = df_wind[zone_temp]
            else:  # "Wind Offshore"
                share_temp = df_wind_off[zone_temp]
            
            # Update the resource table
            res_temp[:, N.index(n), R.index(r)] = 1.5*cap_temp * share_temp.values

    return res_temp


# Create the resource table
res_table = create_res_table()

#print("Demand at time 0 and node 1:", get_dem(0, N[0]))
#print("Resource table shape:", res_table.shape)
#print(f"Wind capacity at node 110: {res_table[0, N.index(110), R.index('Wind')]}")

# Function to get renewable generation at a given time and node
def get_renew(t, n):
    # Sum up the renewable contributions for all types (R) at time `t` and node `n`
    t_index = T.index(t)  # Find the index for the time step
    n_index = N.index(n)  # Find the index for the node
    return sum(res_table[t_index, n_index, R.index(r)] for r in R)
#renewable_per_hour = np.zeros((len(T), len(N)), dtype=float)
#for t in T:
  #  for n in N:
 #       renewable_per_hour[t-1, N.index(n)] = get_renew(t, n)
#renewable_per_hour = renewable_per_hour.sum(axis=1)
#to csv
#pd.DataFrame(renewable_per_hour).to_csv('renewable_per_hour.csv')
 
# use n_in_z to get renewable generation per zone per hour
def get_renew_zone(t, z):
    # Find the nodes in the zone
    nodes_in_zone = n_in_z[z]
    
    # Sum up the renewable contributions for all nodes in the zone
    return sum(get_renew(t, n) for n in nodes_in_zone)

# Example usage
#print("Renewable generation at time", T[0], "and node", N[0], ":", get_renew(T[0], N[0]))
#print("Renewable generation at time", T[0], "and zone", Z[0], ":", get_renew_zone(T[0], Z[0]))
#print("Renewable generation at time", T[0], "and zone", Z[1], ":", get_renew_zone(T[0], Z[1]))
#print("Renewable generation at time", T[0], "and zone", Z[2], ":", get_renew_zone(T[0], Z[2]))

#renewable_per_hour_per_zone = np.zeros((len(T), len(Z)), dtype=float)
#for t in T:
#    for z in Z:
#        renewable_per_hour_per_zone[t-1, Z.index(z)] = get_renew_zone(t, z)

# #to csv
#pd.DataFrame(renewable_per_hour_per_zone, columns=Z).to_csv('renewable_per_hour_per_zone.csv')

# Function to get the maximum generation capacity of a conventional plant
def get_gen_up(p):
    # Find the capacity (Pmax) of the plant with GenID `p`
    return df_plants.loc[df_plants["GenID"] == p, "Pmax"].iloc[0]

# Function to get the capacity of a line
def get_line_cap(l):
    # Find the capacity (Pmax) of the line with BranchID `l`
    return df_branch.loc[df_branch["BranchID"] == l, "Pmax"].iloc[0]

#print("Renewable generation at time", T[115], "and node", N[0], ":", get_renew(T[115], N[0]))
#print("Maximum generation capacity of plant", P[0], ":", get_gen_up(P[0]))
#print("Line capacity of line", L[0], ":", get_line_cap(L[0]))


# Function to find cross-border lines
def find_cross_border_lines():
    cb_lines_temp = []

    # Loop through all lines in L
    for l in L:
        # Get the "FromBus" and "ToBus" for the current line
        from_bus = df_branch.loc[df_branch["BranchID"] == l, "FromBus"].iloc[0]
        to_bus = df_branch.loc[df_branch["BranchID"] == l, "ToBus"].iloc[0]
        
        # Get the zones for the "FromBus" and "ToBus"
        from_zone_temp = df_bus.loc[df_bus["BusID"] == from_bus, "Zone"].iloc[0]
        to_zone_temp = df_bus.loc[df_bus["BusID"] == to_bus, "Zone"].iloc[0]
        
        # Check if both zones are in Z_FBMC and are different
        if from_zone_temp in Z_FBMC and to_zone_temp in Z_FBMC and from_zone_temp != to_zone_temp:
            cb_lines_temp.append(l)  # Add the line to the cross-border list
    
    return cb_lines_temp

# Example usage
cross_border_lines = find_cross_border_lines()
print("Cross-border lines:", cross_border_lines)


#########################################
###    CREATE SUSCEPTANCE MATRICES    ###
#########################################

# Create line and node susceptance matrices
line_sus_mat = np.dot(susceptance.values, incidence.values)
node_sus_mat = np.dot(incidence.values.T, np.dot(susceptance.values, incidence.values))

# Function to get line susceptance
def get_line_sus(l, n):
    # Find the indices of line l and node n
    l_index = L.index(l)  # L is the list of line IDs
    n_index = N.index(n)  # N is the list of node IDs
    return line_sus_mat[l_index, n_index]

# Function to get node susceptance
def get_node_sus(n, nn):
    # Find the indices of nodes n and nn
    n_index = N.index(n)  # N is the list of node IDs
    nn_index = N.index(nn)
    return node_sus_mat[n_index, nn_index]

# Construct H_mat as a dictionary
H_mat = {
    (l, n): get_line_sus(l, n)
    for l in L
    for n in N
}

# Construct B_mat as a dictionary
B_mat = {
    (n, nn): get_node_sus(n, nn)
    for n in N
    for nn in N
}

# Print a subset of the dictionaries for verification
#print("H_mat sample:", {k: H_mat[k] for k in list(H_mat)[:5]})  # Print first 5 entries
#print("B_mat sample:", {k: B_mat[k] for k in list(B_mat)[:5]})  # Print first 5 entries

# Parameters
MWBase = 380**2  # Base value for scaling
slack_node = 50  # Define the slack node
slack_position = N.index(slack_node)  # Find the index of the slack node in N

# Convert susceptance and incidence DataFrames to numpy arrays
line_sus_mat = np.matmul(susceptance.values / MWBase, incidence.values)
node_sus_mat = np.matmul(
    np.matmul(incidence.values.T, susceptance.values / MWBase), incidence.values
)

# Remove the slack position column from the line susceptance matrix
line_sus_mat_ = np.delete(line_sus_mat, slack_position, axis=1)

# Remove the slack position row and column from the node susceptance matrix
node_sus_mat_ = np.delete(np.delete(node_sus_mat, slack_position, axis=0), slack_position, axis=1)


########################################
###    CREATE NODAL PTDF MATRICES    ###
########################################

# Compute the reduced PTDF
PTDF = np.matmul(line_sus_mat_, np.linalg.inv(node_sus_mat_))

# Add a zero column for the slack node
zero_column = np.zeros((len(L), 1))  # L is the list of line IDs
PTDF = np.hstack((PTDF[:, :slack_position], zero_column, PTDF[:, slack_position:]))

# Keep only columns corresponding to N_FBMC
N_FBMC_indices = [N.index(n) for n in N_FBMC]  # Find indices of N_FBMC nodes in N
PTDF = PTDF[:, N_FBMC_indices]


#################################
###    CREATE GSK MATRICES    ###
#################################

# Function to build the flat Generation Shift Key (GSK)
def get_gsk_flat():
    # Initialize the GSK matrix with zeros
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    # Loop through each node in N_FBMC
    for n in N_FBMC:
        # Find the zone associated with the node
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]

        # Calculate the GSK value (1 divided by the number of nodes in the zone)
        gsk_value_temp = 1 / len(df_bus[df_bus["Zone"] == zone_temp])

        # Find the indices of the node (n) and the zone (zone_temp)
        n_index = N_FBMC.index(n)  # Index of the node in N_FBMC
        z_index = Z_FBMC.index(zone_temp)  # Index of the zone in Z_FBMC

        # Assign the GSK value to the corresponding position in the matrix
        gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

# Build the flat GSK
gsk_flat = get_gsk_flat()

# Print the results
#print("Flat GSK matrix:\n", gsk_flat)
#print("Sum of GSK values for each zone (columns):\n", np.sum(gsk_flat, axis=0))


# Function to build the flat unit Generation Shift Key (GSK)
def get_gsk_flat_unit():
    # Initialize the GSK matrix with zeros
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    # Loop through each node in N_FBMC
    for n in N_FBMC:
        # Find the zone associated with the node
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]

        # Find all conventional nodes in the same zone
        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()

        # Check if the current node is a conventional node
        if n in conv_nodes_in_zone:
            # Calculate the GSK value (1 divided by the number of conventional nodes in the zone)
            gsk_value_temp = 1 / len(conv_nodes_in_zone)

            # Find the indices of the node (n) and the zone (zone_temp)
            n_index = N_FBMC.index(n)  # Index of the node in N_FBMC
            z_index = Z_FBMC.index(zone_temp)  # Index of the zone in Z_FBMC

            # Assign the GSK value to the corresponding position in the matrix
            gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

# Build the flat unit GSK
gsk_flat_unit = get_gsk_flat_unit()

# Print the results
#print("Flat Unit GSK matrix:\n", gsk_flat_unit)
#print("Sum of GSK values for each zone (columns):\n", np.sum(gsk_flat_unit, axis=0))


# Function to build the Pmax-based Generation Shift Key (GSK)
def get_gsk_pmax():
    # Initialize the GSK matrix with zeros
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    # Loop through each node in N_FBMC
    for n in N_FBMC:
        # Find the zone associated with the node
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]

        # Find all conventional nodes in the same zone
        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P)),
            "OnBus"
        ].unique()

        if n in conv_nodes_in_zone:
            # Calculate the total Pmax for conventional nodes in the zone
            conv_pmax_in_zone = df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) & (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()

            # Calculate the Pmax for the current node
            conv_pmax_at_node = df_plants.loc[
                (df_plants["OnBus"] == n) & (df_plants["GenID"].isin(P)),
                "Pmax"
            ].sum()

            # Compute the GSK value
            gsk_value_temp = conv_pmax_at_node / conv_pmax_in_zone

            # Find the indices of the node (n) and the zone (zone_temp)
            n_index = N_FBMC.index(n)  # Index of the node in N_FBMC
            z_index = Z_FBMC.index(zone_temp)  # Index of the zone in Z_FBMC

            # Assign the GSK value to the corresponding position in the matrix
            gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

# Build the Pmax-based GSK
gsk_pmax = get_gsk_pmax()

# Print the results
#print("Pmax-based GSK matrix:\n", gsk_pmax)
#print("Sum of GSK values for each zone (columns):\n",  np.sum(gsk_pmax, axis=0))


# Function to build the Pmax-based subset Generation Shift Key (GSK)
def get_gsk_pmax_sub():
    # Filter for generators of type "Hard Coal" or "Gas/CCGT" in Z_FBMC
    P_sub = df_plants.loc[
        (df_plants["Type"].isin(["Hard Coal", "Gas/CCGT"])) & 
        (df_plants["Zone"].isin(Z_FBMC)),
        "GenID"
    ].tolist()

    # Initialize the GSK matrix with zeros
    gsk_temp = np.zeros((len(N_FBMC), len(Z_FBMC)), dtype=float)

    # Loop through each node in N_FBMC
    for n in N_FBMC:
        # Find the zone associated with the node
        zone_temp = df_bus.loc[df_bus["BusID"] == n, "Zone"].iloc[0]

        # Find all conventional nodes in the same zone for the subset of generators
        conv_nodes_in_zone = df_plants.loc[
            (df_plants["Zone"] == zone_temp) & (df_plants["GenID"].isin(P_sub)),
            "OnBus"
        ].unique()

        if n in conv_nodes_in_zone:
            # Calculate the total Pmax for conventional nodes in the zone
            conv_pmax_in_zone = df_plants.loc[
                (df_plants["OnBus"].isin(conv_nodes_in_zone)) & (df_plants["GenID"].isin(P_sub)),
                "Pmax"
            ].sum()

            # Calculate the Pmax for the current node
            conv_pmax_at_node = df_plants.loc[
                (df_plants["OnBus"] == n) & (df_plants["GenID"].isin(P_sub)),
                "Pmax"
            ].sum()

            # Compute the GSK value
            gsk_value_temp = conv_pmax_at_node / conv_pmax_in_zone

            # Find the indices of the node (n) and the zone (zone_temp)
            n_index = N_FBMC.index(n)  # Index of the node in N_FBMC
            z_index = Z_FBMC.index(zone_temp)  # Index of the zone in Z_FBMC

            # Assign the GSK value to the corresponding position in the matrix
            gsk_temp[n_index, z_index] = gsk_value_temp

    return gsk_temp

# Build the Pmax-based subset GSK
gsk_pmax_sub = get_gsk_pmax_sub()

# Print the results
#print("All GSKs built, with column sums:")
#print("1) GSK flat:",  np.round(np.sum(gsk_flat, axis=0), decimals=2))
#print("2) GSK flat unit:", np.round(np.sum(gsk_flat_unit, axis=0), decimals=2))
#print("3) GSK pmax:", np.round(np.sum(gsk_pmax, axis=0), decimals=2))
#print("4) GSK pmax sub:", np.round(np.sum(gsk_pmax_sub, axis=0), decimals=2))


###########################
###    CNE SELECTION    ###
###########################

#### YOUR INPUT ####
cne_alpha = 0.05
gsk_cne = gsk_pmax_sub
gsk_mc = gsk_pmax_sub
frm = 0.05
include_cb_lines = True

# Build zonal PTDFs
PTDF_Z = np.matmul(PTDF, gsk_cne)  # Multiply nodal PTDF with GSK to get zonal PTDFs

# Initialize the z2z_temp matrix
z2z_temp = np.zeros((len(L), int(len(Z_FBMC) * (len(Z_FBMC) - 1) / 2)))

# Fill the z2z_temp matrix
counter = 0  # Python uses 0-based indexing
for z in range(len(Z_FBMC) - 1):  # Loop over zones (z)
    for zz in range(z + 1, len(Z_FBMC)):  # Loop over subsequent zones (zz)
        #print(f"Zone 1: {z+1}, Zone 2: {zz+1}, Counter: {counter+1}")  # Print zones and counter (1-based for clarity)
        
        # Compute the difference between zonal PTDFs for the two zones
        z2z_temp[:, counter] = PTDF_Z[:, z] - PTDF_Z[:, zz]
        
        # Increment the counter
        counter += 1
        

# Compute the absolute values of z2z_temp
z2z_temp_abs = np.abs(z2z_temp)

# Find the maximum absolute value in each row (line)
maximum_abs_z2z = np.max(z2z_temp_abs, axis=1)  # axis=1 corresponds to rows

# Identify Critical Network Elements and Contingencies (CNEC)
CNEC = [L[i] for i, x in enumerate(maximum_abs_z2z) if x >= cne_alpha]

# If cross-border lines should be included
if include_cb_lines:
    # Find cross-border lines
    cross_border_lines = find_cross_border_lines()
    
    # Add cross-border lines not already in CNEC
    for line in cross_border_lines:
        if line not in CNEC:
            CNEC.append(line)
    
    # Ensure CNEC contains only unique entries
    CNEC = list(set(CNEC))

# Ensure CNEC follows the order in L
CNEC = [l for l in L if l in CNEC]

# Extract rows corresponding to CNEC from PTDF and PTDF_Z
PTDF_CNEC = PTDF[[L.index(l) for l in CNEC], :]  # Rows from PTDF corresponding to CNEC
PTDF_Z_CNEC = PTDF_Z[[L.index(l) for l in CNEC], :]  # Rows from PTDF_Z corresponding to CNEC

# Print the number of CNEs selected and the alpha value
print(f"CNE selection: {len(CNEC)} CNEs selected at alpha={cne_alpha}")
print("Critical Network Elements and Contingencies (CNEC):", CNEC)

# %%
