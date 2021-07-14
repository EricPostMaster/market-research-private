# Streamlit
import streamlit as st

# Data processing
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression

# Clustering
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy import stats

# Download button
import base64
import uuid
import re
import plotly.express as px
import time
from io import BytesIO

# App title, favicon
st.set_page_config(
	page_title="Market Research Cluster/Classifier",
	page_icon=":bulb:",
	layout="centered"
)


#########################################################################################
# This is the workhorse function
#########################################################################################

def cluster_solver(df):

    global writer
    global output
    global op

    # Import Data
    df = df

    # Add very small random number to Rating
    df['target']=df['Rating'].apply(lambda x: x+random.random()/1000)


    #########################################################################################
    # Regressions for each UID

    # Unique IDs
    ids = df.UID.unique()

    # Run linear regressions for each UID
    op = pd.DataFrame
    intercept = []
    coefficients=[]
    UID = []
    for p in ids:
        df_i = df[df.UID == p]              # Create dataframe for current user id
        X = df_i.filter(regex='^[a-zA-Z][0-9]')  # df input variables only
        y = df_i['target']                  # Series of target variable
        reg = LinearRegression().fit(X, y)  # Fit linear regression
        reg.score(X, y)                     # Score regression model
        unique_id=df_i['UID'].unique()      # Saves current user id
        const = reg.intercept_              # Save intercept of the regression model
        coef = reg.coef_                    # Coefficients of regression model
        UID.append(unique_id)               # Append current user id
        intercept.append(const)             # Append current intercept
        coefficients.append(coef)           # Append current regression coefficients

    # Convert newly created lists into dataframes
    intercep_new = pd.DataFrame(intercept)
    coefficients_new = pd.DataFrame(coefficients)
    UID_new = pd.DataFrame(UID)

    # Get columns names
    colNames = df.drop(['Rating', 'target',], axis=1).columns
    colNames = colNames.insert(1, 'Const')

    # Concatenate the new dataframes and add column names
    op = pd.concat([UID_new,intercep_new, coefficients_new], axis=1)
    op.columns = colNames

    # Save only regression coefficients for clustering
    scores = op.drop(['UID','Const'], axis=1)

    op_base_column_count = op.shape[1]


    #########################################################################################
    # Clustering, classification, outputs created

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # Create dataframe for storing all cluster/variable combo averages and stdevs
    cls_averages_all = pd.DataFrame()

    # All maps for all cluster solutions
    all_maps = []

    # Column of last cluster solution
    last_var = op.shape[1]

    # Create Pearson distance function
    def pearson_dist(x, y):
        r = stats.pearsonr(x, y)[0]
        return (1 - r) / 2

    # Holds only final cluster solutions created in loop below
    cluster_solutions = {}

    # Maximum number of clusters
    max_clusters = 6

    for n in range(2, max_clusters+1):

        # change your df to numpy arr
        sample = scores.to_numpy()
        
        # define a custom metric
        metric = distance_metric(type_metric.USER_DEFINED, func=pearson_dist)
        
        # carry out a km++ init
        initial_centers = kmeans_plusplus_initializer(sample, n, random_state=123).initialize()
        
        # execute kmeans
        kmeans_instance = kmeans(sample, initial_centers, metric=metric)
        
        # run cluster analysis
        kmeans_instance.process()
        
        # get clusters
        clusters = kmeans_instance.get_clusters()
        
        # Empty dataframe to take in cluster assignments for each loop iteration
        df_clusters = pd.DataFrame()

        for i in range(len(clusters)):
            df_scores = scores.iloc[clusters[i],:]
            df_scores[f'Optimal {n} cluster solution'] = i+1
            df_clusters = pd.concat([df_clusters, df_scores])
            df_clusters.sort_index(inplace=True)
        
        cluster_solutions[f'Optimal {n} cluster solution'] = df_clusters.iloc[:, -1]

    all_cluster_solutions = pd.DataFrame.from_dict(cluster_solutions)

    op = op.merge(all_cluster_solutions, left_index=True, right_index=True)

        
    # #**********************************************************************#
    # # This part takes the averages and standard deviation of each cluster

    last_var = op.shape[1]-max_clusters+1
    last_cluster = op.shape[1]

    for i in range(last_var, last_cluster):
    
        df_cl_cons = op.iloc[:,np.r_[1:last_var,i]]  # Same as df_cl but with constant

        if n == max_clusters:

            cls_avg_list = []

            # Take the mean of every variable for each cluster
            for k in range(1, df_cl_cons.iloc[:,-1].max()+1):
                cls_mean = df_cl_cons[df_cl_cons.iloc[:,-1] == k].iloc[:,0:-1].mean()
                cls_mean = cls_mean.append(pd.Series({"Count":df_cl_cons[df_cl_cons.iloc[:,-1] == k].iloc[:,0:-1].shape[0]}))
                cls_avg_list.append(cls_mean)
                cls_std = df_cl_cons[df_cl_cons.iloc[:,-1] == k].iloc[:,0:-1].std()
                cls_avg_list.append(cls_std)
                # NaN means there is either only 1 observation in that cluster or none.

            # Convert to dataframe and transpose
            cls_averages = pd.DataFrame(cls_avg_list)
            cls_averages = cls_averages.T

            # Create helpful column names (Cluster # of total_#)
            col_names = []

            for col in range(1, k+1):
                new_name1 = f"Avg cluster {col}/{k}"
                col_names.append(new_name1)
                new_name2 = f"Std cluster {col}/{k}"
                col_names.append(new_name2)            

            # Rename columns
            cls_averages.columns = col_names

            cls_averages_all = pd.concat([cls_averages_all, cls_averages], axis=1)

    op.to_excel(writer, index=False, sheet_name="All Regressions, Clusters")


    #**********************************************************************#
    # Add averages for all observations to cls_averages_all before exporting
    all_obs = []

    # Variable means for all observations
    all_obs_mean = list(op.filter(regex='^[a-zA-Z][0-9]').mean().values)
    all_obs_mean.insert(0,op['Const'].mean())
    all_obs_mean.insert(0,len(op))
    all_obs.append(all_obs_mean)

    # Variable standard deviations for all observations
    all_obs_std = list(op.filter(regex='^[a-zA-Z][0-9]').std().values)
    all_obs_std.insert(0,op['Const'].std())
    all_obs_std.insert(0,"")
    all_obs.append(all_obs_std)


    # Save as dataframe and append to all cls_averages_all dataframe
    all_obs_cols = list(op.filter(regex='^[a-zA-Z][0-9]').columns)
    all_obs_cols.insert(0, "Const")
    all_obs_cols.insert(0, "Count")
    all_obs_df = pd.DataFrame(all_obs, columns=all_obs_cols)
    all_obs_df = all_obs_df.T
    all_obs_cols = ['All obs avg', 'All obs stdev']
    all_obs_df.columns = all_obs_cols
    cls_averages_all = pd.concat([cls_averages_all, all_obs_df], axis=1)

    cls_averages_all.to_excel(writer, sheet_name="Cluster Avgs and StDevs")

########################################################################################

# This is the Scenarios analysis that looks at each Category of variable and isolates its impact
# Each variable is exported to its own sheet in the final Excel workbook

    # Unique letters in the Categories
    category_letters = list(df.filter(regex='^[a-zA-Z][0-9]').columns.str[0].unique())
    category_letters_numbers = {key: None for key in category_letters}

    for letter in category_letters:

        # List of all variables starting with the current Category letter
        letter_list = list(df.loc[:, df.columns.str.startswith(letter)].columns)
        number_list = []

        # Create a list of the numbers associated with each Category letter
        for i in letter_list:
            number_list.append(i[1])
        
        # Add the maximum number of the Category as a value to the Category letter key
        category_letters_numbers[letter] = int(max(number_list))

    all_regressions = []

    for category in category_letters_numbers:
        
        all_cat_coefs = {}

        # Initial run for A0 (e.g., show only samples when A was omitted)
        var_of_interest = category+'0'

        vars_of_interest = df.columns.str[0] == category
        vars_of_interest = list(df.columns[vars_of_interest])

        # Include in df_var if all Categories are 0
        df_var = df[(df[vars_of_interest] == 0).all(axis=1)]

        # Drop Category columns
        df_var = df_var.drop(vars_of_interest, axis=1)

        X = df_var.filter(regex='^[a-zA-Z][0-9]')
        y = df_var['target']    

        # Fit and score model
        reg = LinearRegression().fit(X, y)
        reg.score(X, y)
        const = reg.intercept_
        coef = list(reg.coef_)
        coef.insert(0, const)

        all_cat_coefs[var_of_interest] = coef
        all_columns = list(X.columns.insert(0,'const'))

        # Now do the same regression as above, but isolate each individual variable in the category (e.g., A1, A2, etc.)
        for number in range(1,category_letters_numbers[category]+1):
            # Isolate the variable of interest
            var_of_interest = category+str(number)

            # Create dataset with variable of interest == 1
            df_var = df[df[var_of_interest] == 1]
            
            # Filter out the variables from the Category letter
            keep_columns = df_var.columns.str[0] != category

            df_var = df_var[df_var.columns[keep_columns]]
            
            # Create predictor and target datasets (and remove UID and Rating)
            X = df_var.filter(regex='^[a-zA-Z][0-9]')
            y = df_var['target']

            # Fit and score model
            reg = LinearRegression().fit(X, y)
            reg.score(X, y)
            const = reg.intercept_
            coef = list(reg.coef_)
            coef.insert(0, const)

            # All coefficients and vars of interest needed (data, columns)
            all_cat_coefs[var_of_interest] = coef

            # Columns (index)
            all_columns = list(X.columns.insert(0,'const'))
            

        # Put the pieces together
        category_df_components = [category, all_cat_coefs, all_columns]

        # Add to master list to put into individual dataframes
        all_regressions.append(category_df_components)

    # Put each element of all_regressions into its own dataframe and save to the Excel document

    for i in all_regressions:
        df_category = pd.DataFrame(data=i[1], index=i[2])
        df_category.to_excel(writer, index=True, sheet_name=f"Category {i[0]} Scenarios")

        
########################################################################################

# This is the end of the function. Just don't break it, and life is good :)

    return [writer, op]


#########################################################################################
# Download link
#########################################################################################

def get_table_download_link(table):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)
    # button_id = "download_button"

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(204, 204, 204);
                color: rgb(38, 39, 48);
                padding: 0.55em 0.68em;
                position: relative;
                text-decoration: none;
                border-radius: 8px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                background-color: rgb(184, 184, 184);
                color: rgb(20, 21, 25);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(154, 154, 154);
                color: rgb(20, 21, 25);
                }}
        </style> """


    writer.save()
    xlsx_data = output.getvalue()
    b64 = base64.b64encode(xlsx_data)
    dl_link = custom_css + f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.xlsx" id="{button_id}">Download Excel file</a>' # decode b'abc' => abc
    return dl_link






#########################################################################################
# Streamlit app begins
#########################################################################################

st.title("Cluster and Classification Mapping Utility")

st.subheader("How it works:")
st.write("""Upload your research participant dataset, and this tool will compute
optimal cluster solutions and mappings for key variable combinations""")
data_file = st.file_uploader("Upload CSV",type=['csv'])

# Once a file is uploaded, everything starts
if data_file is not None:

	# Save uploaded CSV to dataframe
	df = pd.read_csv(data_file)

	# Reassign value of df attribute to the new dataframe df
	r = cluster_solver(df)
	
	# Progress bar animation
	my_bar = st.progress(0)

	for percent_complete in range(100):
		time.sleep(0.01)
		my_bar.progress(percent_complete + 1)
	
	# Display and download clustered data 
	st.header("Clustered Data & Mappings")
	st.write("""Cluster assignments have been added to the original data.
	Click the button below to download cluster solutions and mappings.""")
	st.dataframe(op)

	st.write("To download the dataset, enter the desired filename and click below.")

	# The download filename and button are in these two columns
	col1, col2 = st.beta_columns([2,1])

	with col1:
		filename = st.text_input('Enter output filename:', 'clustered-data')

	with col2:
		st.write(" ")
		st.write(" ")
		st.write(" ")
		download_button_str = get_table_download_link(writer)
		st.markdown(download_button_str, unsafe_allow_html=True)