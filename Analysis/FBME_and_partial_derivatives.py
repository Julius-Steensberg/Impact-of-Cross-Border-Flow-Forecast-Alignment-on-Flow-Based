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


GSK_list = ['_flat'] # select either '_flat' or '_pmax_sub'


#%% base case:
NP_list = ['']
for GSK in GSK_list:
    for NP in NP_list:
        D1_CGM_NP_path = os.path.join(
            "..", f"D-1_CGM_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))

        D2_CGM_NP_path = os.path.join(
            "..", f"D-2_base_case_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
        D1_CGM_Flow_path = os.path.join(
            "..", f"D-1_CGM_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D1_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D1_CGM_Flow_path))
        D2_CGM_Flow_path = os.path.join(
            "..", f"D-2_base_case_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D2_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D2_CGM_Flow_path))
        PTDF_Z_CNE_path = os.path.join(
            "..", f"D-1_MC_results{GSK}", f"{Number_of_hours}_hours{NP}", "PTDF_Z_CNEC.csv")
        PTDF_Z_CNE_path = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path))
        PTDF_Z_CNE = pd.read_csv(PTDF_Z_CNE_path, index_col=0)
        D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
        D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
        D1_CGM_Flow = pd.read_csv(D1_CGM_Flow_path, index_col=0)
        D2_CGM_Flow = pd.read_csv(D2_CGM_Flow_path, index_col=0)
        CNEC_lines_list = (PTDF_Z_CNE.index).to_list()
        CNEC_lines_list = [str(i) for i in CNEC_lines_list]
        D1_CGM_Flow_CNEC = D1_CGM_Flow[CNEC_lines_list]
        D2_CGM_Flow_CNEC = D2_CGM_Flow[CNEC_lines_list]
        FBME_df = pd.DataFrame(index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_df = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values)
        FBME_df = pd.DataFrame(FBME_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_underestimate_pos_df = np.abs(D1_CGM_Flow_CNEC.values) - np.abs(D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values))
        FBME_underestimate_pos_df = pd.DataFrame(FBME_underestimate_pos_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_Flow_difference = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values
        D1_D2_Flow_difference = pd.DataFrame(D1_D2_Flow_difference, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_NP_difference = D1_CGM_NP.values - D2_CGM_NP.values
        D1_D2_NP_difference = pd.DataFrame(D1_D2_NP_difference, index=D1_CGM_Flow_CNEC.index, columns=['NP1','NP2','NP3'])
        adjustment = np.matmul(D1_D2_NP_difference, PTDF_Z_CNE.T.values)
        adjustment = pd.DataFrame(adjustment, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_outer_folder = os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours{GSK}_base_case"))
        # create folder
        if not os.path.exists(FBME_outer_folder):
            os.makedirs(FBME_outer_folder)

        FBME_decomposition_folder = os.path.abspath(os.path.join(FBME_outer_folder, "FBME_decomposition"))
        # create folder
        if not os.path.exists(FBME_decomposition_folder):
            os.makedirs(FBME_decomposition_folder)
        #to csv in //FBME folder
        FBME_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME{GSK}{NP}.parquet")))
        FBME_underestimate_pos_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME_underestimate_pos{GSK}{NP}.parquet")))
        D1_D2_Flow_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_Flow_difference{GSK}{NP}.parquet")))
        D1_D2_NP_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_NP_difference{GSK}{NP}.parquet")))
        adjustment.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"adjustment{GSK}{NP}.parquet")))



#%% CGMA pipeline:

NP_list = ['']
for GSK in GSK_list:
    for NP in NP_list:
        D1_CGM_NP_path = os.path.join(
            "..", f"D-1_CGM_fixed_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))

        D2_CGM_NP_path = os.path.join(
            "..", f"D-2_base_case_fixed_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))

        D1_CGM_Flow_path = os.path.join(
            "..", f"D-1_CGM_fixed_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        
        D1_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D1_CGM_Flow_path))

        D2_CGM_Flow_path = os.path.join(
            "..", f"D-2_base_case_fixed_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D2_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D2_CGM_Flow_path))

        PTDF_Z_CNE_path = os.path.join(
            "..", f"D-1_MC_fixed_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "PTDF_Z_CNEC.csv")
        PTDF_Z_CNE_path = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path))

        PTDF_Z_CNE = pd.read_csv(PTDF_Z_CNE_path, index_col=0)
        D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
        D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
        D1_CGM_Flow = pd.read_csv(D1_CGM_Flow_path, index_col=0)
        D2_CGM_Flow = pd.read_csv(D2_CGM_Flow_path, index_col=0)

        CNEC_lines_list = (PTDF_Z_CNE.index).to_list()
        CNEC_lines_list = [str(i) for i in CNEC_lines_list]

        D1_CGM_Flow_CNEC = D1_CGM_Flow[CNEC_lines_list]
        D2_CGM_Flow_CNEC = D2_CGM_Flow[CNEC_lines_list]

        FBME_df = pd.DataFrame(index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_df = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values)
        FBME_df = pd.DataFrame(FBME_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_underestimate_pos_df = np.abs(D1_CGM_Flow_CNEC.values) - np.abs(D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values))
        FBME_underestimate_pos_df = pd.DataFrame(FBME_underestimate_pos_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        D1_D2_Flow_difference = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values
        D1_D2_Flow_difference = pd.DataFrame(D1_D2_Flow_difference, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_NP_difference = D1_CGM_NP.values - D2_CGM_NP.values
        D1_D2_NP_difference = pd.DataFrame(D1_D2_NP_difference, index=D1_CGM_Flow_CNEC.index, columns=['NP1','NP2','NP3'])
        adjustment = np.matmul(D1_D2_NP_difference, PTDF_Z_CNE.T.values)
        adjustment = pd.DataFrame(adjustment, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_outer_folder = os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours{GSK}_fixed_NP"))
        # create folder
        if not os.path.exists(FBME_outer_folder):
            os.makedirs(FBME_outer_folder)

        FBME_decomposition_folder = os.path.abspath(os.path.join(FBME_outer_folder, "FBME_decomposition"))
        # create folder
        if not os.path.exists(FBME_decomposition_folder):
            os.makedirs(FBME_decomposition_folder)

        #to csv in //FBME folder
        FBME_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME{GSK}{NP}.parquet")))
        FBME_underestimate_pos_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME_underestimate_pos{GSK}{NP}.parquet")))
        D1_D2_Flow_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_Flow_difference{GSK}{NP}.parquet")))
        D1_D2_NP_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_NP_difference{GSK}{NP}.parquet")))
        adjustment.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"adjustment{GSK}{NP}.parquet")))



#%% Perfect CGMA NP:

NP_list = ['']
for GSK in GSK_list:
    for NP in NP_list:
        D1_CGM_NP_path = os.path.join(
            "..", f"D-1_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))

        D2_CGM_NP_path = os.path.join(
            "..", f"D-2_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))

        D1_CGM_Flow_path = os.path.join(
            "..", f"D-1_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        
        D1_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D1_CGM_Flow_path))

        D2_CGM_Flow_path = os.path.join(
            "..", f"D-2_CGM_perfect_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D2_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D2_CGM_Flow_path))

        PTDF_Z_CNE_path = os.path.join(
            "..", f"D-1_MC_perfect_NP_results{GSK}", f"{Number_of_hours}_hours{NP}", "PTDF_Z_CNEC.csv")
        PTDF_Z_CNE_path = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path))

        PTDF_Z_CNE = pd.read_csv(PTDF_Z_CNE_path, index_col=0)
        D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
        D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
        D1_CGM_Flow = pd.read_csv(D1_CGM_Flow_path, index_col=0)
        D2_CGM_Flow = pd.read_csv(D2_CGM_Flow_path, index_col=0)

        CNEC_lines_list = (PTDF_Z_CNE.index).to_list()
        CNEC_lines_list = [str(i) for i in CNEC_lines_list]

        D1_CGM_Flow_CNEC = D1_CGM_Flow[CNEC_lines_list]
        D2_CGM_Flow_CNEC = D2_CGM_Flow[CNEC_lines_list]

        FBME_df = pd.DataFrame(index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_df = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values)
        FBME_df = pd.DataFrame(FBME_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_underestimate_pos_df = np.abs(D1_CGM_Flow_CNEC.values) - np.abs(D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values))
        FBME_underestimate_pos_df = pd.DataFrame(FBME_underestimate_pos_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        D1_D2_Flow_difference = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values
        D1_D2_Flow_difference = pd.DataFrame(D1_D2_Flow_difference, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_NP_difference = D1_CGM_NP.values - D2_CGM_NP.values
        D1_D2_NP_difference = pd.DataFrame(D1_D2_NP_difference, index=D1_CGM_Flow_CNEC.index, columns=['NP1','NP2','NP3'])
        adjustment = np.matmul(D1_D2_NP_difference, PTDF_Z_CNE.T.values)
        adjustment = pd.DataFrame(adjustment, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_outer_folder = os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours{GSK}_perfect_NP"))
        # create folder
        if not os.path.exists(FBME_outer_folder):
            os.makedirs(FBME_outer_folder)

        FBME_decomposition_folder = os.path.abspath(os.path.join(FBME_outer_folder, "FBME_decomposition"))
        # create folder
        if not os.path.exists(FBME_decomposition_folder):
            os.makedirs(FBME_decomposition_folder)

        #to csv in //FBME folder
        FBME_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME{GSK}{NP}.parquet")))
        FBME_underestimate_pos_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME_underestimate_pos{GSK}{NP}.parquet")))
        D1_D2_Flow_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_Flow_difference{GSK}{NP}.parquet")))
        D1_D2_NP_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_NP_difference{GSK}{NP}.parquet")))
        adjustment.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"adjustment{GSK}{NP}.parquet")))

#%% PTC 50 epochs

NP_list = ['']
for GSK in GSK_list:
    for NP in NP_list:

        D1_CGM_NP_path = os.path.join(
            "..", f"D-1_CGM_fixed_PTC_results{GSK}_50_epochs", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))
        D2_CGM_NP_path = os.path.join(
            "..", f"D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
        D1_CGM_Flow_path = os.path.join(
            "..", f"D-1_CGM_fixed_PTC_results{GSK}_50_epochs", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D1_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D1_CGM_Flow_path))
        D2_CGM_Flow_path = os.path.join(
            "..", f"D-2_base_case_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D2_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D2_CGM_Flow_path))
        PTDF_Z_CNE_path = os.path.join(
            "..", f"D-1_MC_fixed_PTC_results_50_epochs", f"{Number_of_hours}_hours{NP}", "PTDF_Z_CNEC.csv")
        PTDF_Z_CNE_path = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path))
        PTDF_Z_CNE = pd.read_csv(PTDF_Z_CNE_path, index_col=0)
        D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
        D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
        D1_CGM_Flow = pd.read_csv(D1_CGM_Flow_path, index_col=0)
        D2_CGM_Flow = pd.read_csv(D2_CGM_Flow_path, index_col=0)
        CNEC_lines_list = (PTDF_Z_CNE.index).to_list()
        CNEC_lines_list = [str(i) for i in CNEC_lines_list]
        D1_CGM_Flow_CNEC = D1_CGM_Flow[CNEC_lines_list]
        D2_CGM_Flow_CNEC = D2_CGM_Flow[CNEC_lines_list]
        FBME_df = pd.DataFrame(index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_df = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values)
        FBME_df = pd.DataFrame(FBME_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_underestimate_pos_df = np.abs(D1_CGM_Flow_CNEC.values) - np.abs(D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values))
        FBME_underestimate_pos_df = pd.DataFrame(FBME_underestimate_pos_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_Flow_difference = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values
        D1_D2_Flow_difference = pd.DataFrame(D1_D2_Flow_difference, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_NP_difference = D1_CGM_NP.values - D2_CGM_NP.values

        D1_D2_NP_difference = pd.DataFrame(D1_D2_NP_difference, index=D1_CGM_Flow_CNEC.index, columns=['NP1','NP2','NP3'])
        adjustment = np.matmul(D1_D2_NP_difference, PTDF_Z_CNE.T.values)
        adjustment = pd.DataFrame(adjustment, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_outer_folder = os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours{GSK}_fixed_PTC_50_epochs"))
        # create folder
        if not os.path.exists(FBME_outer_folder):
            os.makedirs(FBME_outer_folder)
        FBME_decomposition_folder = os.path.abspath(os.path.join(FBME_outer_folder, "FBME_decomposition"))
        # create folder
        if not os.path.exists(FBME_decomposition_folder):
            os.makedirs(FBME_decomposition_folder)
        #to csv in //FBME folder
        FBME_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME{GSK}{NP}.parquet")))
        FBME_underestimate_pos_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME_underestimate_pos{GSK}{NP}.parquet")))
        D1_D2_Flow_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_Flow_difference{GSK}{NP}.parquet")))
        D1_D2_NP_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_NP_difference{GSK}{NP}.parquet")))

#%% PTC 300 epochs

NP_list = ['']
for GSK in GSK_list:
    for NP in NP_list:

        D1_CGM_NP_path = os.path.join(
            "..", f"D-1_CGM_fixed_PTC_results{GSK}_300_epochs", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))
        D2_CGM_NP_path = os.path.join(
            "..", f"D-2_base_case_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
        D1_CGM_Flow_path = os.path.join(
            "..", f"D-1_CGM_fixed_PTC_results{GSK}_300_epochs", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D1_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D1_CGM_Flow_path))
        D2_CGM_Flow_path = os.path.join(
            "..", f"D-2_base_case_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D2_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D2_CGM_Flow_path))
        PTDF_Z_CNE_path = os.path.join(
            "..", f"D-1_MC_fixed_PTC_results_300_epochs", f"{Number_of_hours}_hours{NP}", "PTDF_Z_CNEC.csv")
        PTDF_Z_CNE_path = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path))
        PTDF_Z_CNE = pd.read_csv(PTDF_Z_CNE_path, index_col=0)
        D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
        D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
        D1_CGM_Flow = pd.read_csv(D1_CGM_Flow_path, index_col=0)
        D2_CGM_Flow = pd.read_csv(D2_CGM_Flow_path, index_col=0)
        CNEC_lines_list = (PTDF_Z_CNE.index).to_list()
        CNEC_lines_list = [str(i) for i in CNEC_lines_list]
        D1_CGM_Flow_CNEC = D1_CGM_Flow[CNEC_lines_list]
        D2_CGM_Flow_CNEC = D2_CGM_Flow[CNEC_lines_list]
        FBME_df = pd.DataFrame(index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_df = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values)
        FBME_df = pd.DataFrame(FBME_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_underestimate_pos_df = np.abs(D1_CGM_Flow_CNEC.values) - np.abs(D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values))
        FBME_underestimate_pos_df = pd.DataFrame(FBME_underestimate_pos_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_Flow_difference = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values
        D1_D2_Flow_difference = pd.DataFrame(D1_D2_Flow_difference, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_NP_difference = D1_CGM_NP.values - D2_CGM_NP.values

        D1_D2_NP_difference = pd.DataFrame(D1_D2_NP_difference, index=D1_CGM_Flow_CNEC.index, columns=['NP1','NP2','NP3'])
        adjustment = np.matmul(D1_D2_NP_difference, PTDF_Z_CNE.T.values)
        adjustment = pd.DataFrame(adjustment, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_outer_folder = os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours{GSK}_fixed_PTC_300_epochs"))
        # create folder
        if not os.path.exists(FBME_outer_folder):
            os.makedirs(FBME_outer_folder)
        FBME_decomposition_folder = os.path.abspath(os.path.join(FBME_outer_folder, "FBME_decomposition"))
        # create folder
        if not os.path.exists(FBME_decomposition_folder):
            os.makedirs(FBME_decomposition_folder)
        #to csv in //FBME folder
        FBME_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME{GSK}{NP}.parquet")))
        FBME_underestimate_pos_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME_underestimate_pos{GSK}{NP}.parquet")))
        D1_D2_Flow_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_Flow_difference{GSK}{NP}.parquet")))
        D1_D2_NP_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_NP_difference{GSK}{NP}.parquet")))



#%% Perfect CGMA PTC:

NP_list = ['']
for NP in NP_list:
    for GSK in GSK_list:
        D1_CGM_NP_path = os.path.join(
            "..", f"D-1_CGM_perfect_PTC_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))
        D2_CGM_NP_path = os.path.join(
            "..", f"D-2_CGM_perfect_PTC_results", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))
        D1_CGM_Flow_path = os.path.join(
            "..", f"D-1_CGM_perfect_PTC_results{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D1_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D1_CGM_Flow_path))
        D2_CGM_Flow_path = os.path.join(
            "..", f"D-2_CGM_perfect_PTC_results", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D2_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D2_CGM_Flow_path))
        PTDF_Z_CNE_path = os.path.join(
            "..", f"D-1_MC_perfect_PTC_results", f"{Number_of_hours}_hours{NP}", "PTDF_Z_CNEC.csv")
        PTDF_Z_CNE_path = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path))
        PTDF_Z_CNE = pd.read_csv(PTDF_Z_CNE_path, index_col=0)
        D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
        D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
        D1_CGM_Flow = pd.read_csv(D1_CGM_Flow_path, index_col=0)
        D2_CGM_Flow = pd.read_csv(D2_CGM_Flow_path, index_col=0)
        CNEC_lines_list = (PTDF_Z_CNE.index).to_list()
        CNEC_lines_list = [str(i) for i in CNEC_lines_list]
        D1_CGM_Flow_CNEC = D1_CGM_Flow[CNEC_lines_list]
        D2_CGM_Flow_CNEC = D2_CGM_Flow[CNEC_lines_list]
        FBME_df = pd.DataFrame(index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_df = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values)
        FBME_df = pd.DataFrame(FBME_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_underestimate_pos_df = np.abs(D1_CGM_Flow_CNEC.values) - np.abs(D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values))
        FBME_underestimate_pos_df = pd.DataFrame(FBME_underestimate_pos_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_Flow_difference = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values
        D1_D2_Flow_difference = pd.DataFrame(D1_D2_Flow_difference, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_NP_difference = D1_CGM_NP.values - D2_CGM_NP.values
        D1_D2_NP_difference = pd.DataFrame(D1_D2_NP_difference, index=D1_CGM_Flow_CNEC.index, columns=['NP1','NP2','NP3'])
        adjustment = np.matmul(D1_D2_NP_difference, PTDF_Z_CNE.T.values)
        adjustment = pd.DataFrame(adjustment, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        FBME_outer_folder = os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours{GSK}_perfect_PTC"))

        # create folder
        if not os.path.exists(FBME_outer_folder):
            os.makedirs(FBME_outer_folder)
        FBME_decomposition_folder = os.path.abspath(os.path.join(FBME_outer_folder, "FBME_decomposition"))
        # create folder
        if not os.path.exists(FBME_decomposition_folder):
            os.makedirs(FBME_decomposition_folder)
        #to csv in //FBME folder
        FBME_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME{GSK}{NP}.parquet")))
        FBME_underestimate_pos_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME_underestimate_pos{GSK}{NP}.parquet")))
        D1_D2_Flow_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_Flow_difference{GSK}{NP}.parquet")))
        D1_D2_NP_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_NP_difference{GSK}{NP}.parquet")))




#%%


#################################################
###   CALCULATE FBME FOR PARTIAL DERIVATIVES  ###
#################################################
NP_list = ['_NP1','_NP2','_NP3']
for GSK in GSK_list:
    for NP in NP_list:
        D1_CGM_NP_path = os.path.join(
            "..", f"D-1_CGM_PD{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D1_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D1_CGM_NP_path))

        D2_CGM_NP_path = os.path.join(
            "..", f"D-2_PD{GSK}", f"{Number_of_hours}_hours{NP}", "df_np.csv")
        D2_CGM_NP_path = os.path.abspath(os.path.join(script_dir, D2_CGM_NP_path))

        D1_CGM_Flow_path = os.path.join(
            "..", f"D-1_CGM_PD{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        
        D1_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D1_CGM_Flow_path))

        D2_CGM_Flow_path = os.path.join(
            "..", f"D-2_PD{GSK}", f"{Number_of_hours}_hours{NP}", "df_line_f.csv")
        D2_CGM_Flow_path = os.path.abspath(os.path.join(script_dir, D2_CGM_Flow_path))

        PTDF_Z_CNE_path = os.path.join(
            "..", f"D-1_MC_fixed_NP_results_flat", f"{Number_of_hours}_hours", "PTDF_Z_CNEC.csv")        
        PTDF_Z_CNE_path = os.path.abspath(os.path.join(script_dir, PTDF_Z_CNE_path))

        PTDF_Z_CNE = pd.read_csv(PTDF_Z_CNE_path, index_col=0)
        D1_CGM_NP = pd.read_csv(D1_CGM_NP_path, index_col=0)
        D2_CGM_NP = pd.read_csv(D2_CGM_NP_path, index_col=0)
        D1_CGM_Flow = pd.read_csv(D1_CGM_Flow_path, index_col=0)
        D2_CGM_Flow = pd.read_csv(D2_CGM_Flow_path, index_col=0)

        CNEC_lines_list = (PTDF_Z_CNE.index).to_list()
        CNEC_lines_list = [str(i) for i in CNEC_lines_list]

        D1_CGM_Flow_CNEC = D1_CGM_Flow[CNEC_lines_list]
        D2_CGM_Flow_CNEC = D2_CGM_Flow[CNEC_lines_list]

        FBME_df = pd.DataFrame(index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_df = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values)
        FBME_df = pd.DataFrame(FBME_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_underestimate_pos_df = np.abs(D1_CGM_Flow_CNEC.values) - np.abs(D2_CGM_Flow_CNEC.values - np.matmul(D1_CGM_NP.values - D2_CGM_NP.values, PTDF_Z_CNE.T.values))
        FBME_underestimate_pos_df = pd.DataFrame(FBME_underestimate_pos_df, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        D1_D2_Flow_difference = D1_CGM_Flow_CNEC.values - D2_CGM_Flow_CNEC.values
        D1_D2_Flow_difference = pd.DataFrame(D1_D2_Flow_difference, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)
        D1_D2_NP_difference = D1_CGM_NP.values - D2_CGM_NP.values
        D1_D2_NP_difference = pd.DataFrame(D1_D2_NP_difference, index=D1_CGM_Flow_CNEC.index, columns=['NP1','NP2','NP3'])
        adjustment = np.matmul(D1_D2_NP_difference, PTDF_Z_CNE.T.values)
        adjustment = pd.DataFrame(adjustment, index=D1_CGM_Flow_CNEC.index, columns=CNEC_lines_list)

        FBME_outer_folder = os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours_PD{GSK}"))
        # create folder
        if not os.path.exists(FBME_outer_folder):
            os.makedirs(FBME_outer_folder)

        FBME_decomposition_folder = os.path.abspath(os.path.join(FBME_outer_folder, "FBME_decomposition"))
        # create folder
        if not os.path.exists(FBME_decomposition_folder):
            os.makedirs(FBME_decomposition_folder)

        #to csv in //FBME folder
        FBME_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME{GSK}{NP}.parquet")))
        FBME_underestimate_pos_df.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_outer_folder, f"FBME_underestimate_pos{GSK}{NP}.parquet")))
        D1_D2_Flow_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_Flow_difference{GSK}{NP}.parquet")))
        D1_D2_NP_difference.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"D1_D2_NP_difference{GSK}{NP}.parquet")))
        adjustment.to_parquet(os.path.abspath(os.path.join(script_dir, FBME_decomposition_folder, f"adjustment{GSK}{NP}.parquet")))

#%%
# read the 16 parquet files
try:
    FBME_flat = pd.read_parquet(os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours_flat_fixed_NP/FBME_flat.parquet")))
    FBME_flat_NP1 = pd.read_parquet(os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours_PD_flat/FBME_flat_NP1.parquet")))
    FBME_flat_NP2 = pd.read_parquet(os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours_PD_flat/FBME_flat_NP2.parquet")))
    FBME_flat_NP3 = pd.read_parquet(os.path.abspath(os.path.join(script_dir, f"FBME_{Number_of_hours}_hours_PD_flat/FBME_flat_NP3.parquet")))   
    print('flat files found')
except:
    print('flat files not found')

try:
    FBME_slack_pos_path = os.path.join(
        "..", "D-2_base_case_fixed_NP_results_flat", f"{Number_of_hours}_hours", "df_NP_slack_pos.csv")
    FBME_slack_pos_path = os.path.abspath(os.path.join(script_dir, FBME_slack_pos_path))
    FBME_slack_pos = pd.read_csv(FBME_slack_pos_path, index_col=0)
    FBME_slack_neg_path = os.path.join(
        "..", "D-2_base_case_fixed_NP_results_flat", f"{Number_of_hours}_hours", "df_NP_slack_neg.csv")
    FBME_slack_neg_path = os.path.abspath(os.path.join(script_dir, FBME_slack_neg_path))
    FBME_slack_neg = pd.read_csv(FBME_slack_neg_path, index_col=0)
    print('slack files found')
except:
    print('slack files not found')

############################
###   GET SLACK VALUES   ###
############################

slack_df = FBME_slack_pos - FBME_slack_neg

# find indices in slack_df '1' where there is a value > 0 or < 0
slack_df_with_values_Z1 = slack_df[slack_df['1'] != 0]
slack_df_with_values_Z2 = slack_df[slack_df['2'] != 0]
slack_df_with_values_Z3 = slack_df[slack_df['3'] != 0]

slack_df_with_0_values_Z1 = slack_df[slack_df['1'] == 0]['1']
slack_df_with_0_values_Z2 = slack_df[slack_df['2'] == 0]['2']
slack_df_with_0_values_Z3 = slack_df[slack_df['3'] == 0]['3']




##############################################
###   CALCULATE MEAN PARTIAL DERIVATIVES   ###
##############################################

uncertainty = 0.001

# flat
PD_flat_NP1 = (FBME_flat_NP1.abs() - FBME_flat.abs())/uncertainty
PD_flat_NP2 = (FBME_flat_NP2.abs() - FBME_flat.abs())/uncertainty
PD_flat_NP3 = (FBME_flat_NP3.abs() - FBME_flat.abs())/uncertainty

# in PD_flat_NP1 only extract indexes equal to index in slack_df_with_0_values_Z1 
PD_flat_NP1 = PD_flat_NP1.loc[slack_df_with_0_values_Z1.index]
PD_flat_NP2 = PD_flat_NP2.loc[slack_df_with_0_values_Z2.index]
PD_flat_NP3 = PD_flat_NP3.loc[slack_df_with_0_values_Z3.index]



PD_flat_NP1_med = PD_flat_NP1.median()
PD_flat_NP2_med = PD_flat_NP2.median()
PD_flat_NP3_med = PD_flat_NP3.median()

# export as csv
#PD_flat_NP1_med.to_csv(os.path.abspath(os.path.join(script_dir, f"PD_flat_NP1_med.csv")))
#PD_flat_NP2_med.to_csv(os.path.abspath(os.path.join(script_dir, f"PD_flat_NP2_med.csv")))
#PD_flat_NP3_med.to_csv(os.path.abspath(os.path.join(script_dir, f"PD_flat_NP3_med.csv")))

PD_flat_NP1_med_mean = PD_flat_NP1.median().mean()
PD_flat_NP2_med_mean = PD_flat_NP2.median().mean()
PD_flat_NP3_med_mean = PD_flat_NP3.median().mean()

PD_flat_NP1_mean_mean = PD_flat_NP1.mean().mean()
PD_flat_NP2_mean_mean = PD_flat_NP2.mean().mean()
PD_flat_NP3_mean_mean = PD_flat_NP3.mean().mean()

print('flat median mean:')
print(f"Median mean PD for NP1: {PD_flat_NP1_med_mean}")
print(f"Median mean PD for NP2: {PD_flat_NP2_med_mean}")
print(f"Median mean PD for NP3: {PD_flat_NP3_med_mean}")
print('')
print('flat mean mean:')
print(f"Mean mean PD for NP1: {PD_flat_NP1_mean_mean}")
print(f"Mean mean PD for NP2: {PD_flat_NP2_mean_mean}")
print(f"Mean mean PD for NP3: {PD_flat_NP3_mean_mean}")


sns.histplot(PD_flat_NP1.median(), bins=30, kde=True)
sns.histplot(PD_flat_NP2.median(), bins=30, kde=True)
sns.histplot(PD_flat_NP3.median(), bins=30, kde=True)
PD_flat_NP1_med_sorted = PD_flat_NP1_med.sort_values(ascending=False)
PD_flat_NP2_med_sorted = PD_flat_NP2_med.sort_values(ascending=False)
PD_flat_NP3_med_sorted = PD_flat_NP3_med.sort_values(ascending=False)

def plot_median_partial_derivatives(data, title):
    plt.figure(figsize=(15, 5))
    ax = sns.barplot(x=data.index, y=data.values)
    ax.set_xticks([])
    plt.xlabel('CNE', fontsize=14)
    plt.ylabel('Median Partial Derivatives', fontsize=14)
    plt.title(title, fontsize=16)
    plt.show()

# Call the function separately for each dataset
plot_median_partial_derivatives(PD_flat_NP1_med_sorted, 'GSK: Flat \n Median Partial Derivatives for each CNE for a CGMA error in Zone 1 NP')
plot_median_partial_derivatives(PD_flat_NP2_med_sorted, 'GSK: Flat \n Median Partial Derivatives for each CNE for a CGMA error in Zone 2 NP')
plot_median_partial_derivatives(PD_flat_NP3_med_sorted, 'GSK: Flat \n Median Partial Derivatives for each CNE for a CGMA error in Zone 3 NP')

# %%
def plot_median_partial_derivatives_subplots(data_lists, titles, main_title):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharey=True)
    
    for i, (data, title) in enumerate(zip(data_lists, titles)):
        ax = axes[i]
        sns.barplot(x=data.index, y=data.values, ax=ax, palette='viridis')
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=14)
        ax.set_xlabel('CNE', fontsize=14)
        ax.set_ylabel('Median Partial Derivatives [MW]', fontsize=14, labelpad=10)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    # Add a main title
    if main_title:
        fig.suptitle(main_title[0], fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(script_dir, 'median_partial_derivatives_subplots.png'), dpi=300)
    plt.show()

# Call the function
plot_median_partial_derivatives_subplots(
    [PD_flat_NP1_med_sorted, PD_flat_NP2_med_sorted, PD_flat_NP3_med_sorted],
    ['Increased CGMA error in Zone 1 NP',
     'Increased CGMA error in Zone 2 NP',
     'Increased CGMA error in Zone 3 NP'],
    ['Median Partial Derivatives for each CNE']
)