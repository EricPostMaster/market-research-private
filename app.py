# Streamlit
#!pip install streamlit
import streamlit as st
import streamlit.components.v1 as stc

# Data processing
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import LinearRegression
import xlsxwriter

# Clustering
from sklearn.cluster import KMeans
# conda install -c conda-forge scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from operator import itemgetter
from scipy.spatial import distance_matrix
from scipy.spatial import distance
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import type_metric, distance_metric
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from scipy import stats

# Classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import itertools

# Download button
import base64
import os
import json
import pickle
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
            df = scores.iloc[clusters[i],:]
            df[f'Optimal {n} cluster solution'] = i+1
            df_clusters = pd.concat([df_clusters, df])
            df_clusters.sort_index(inplace=True)
        
        cluster_solutions[f'Optimal {n} cluster solution'] = df_clusters.iloc[:, -1]

    all_cluster_solutions = pd.DataFrame.from_dict(cluster_solutions)

    op.merge(all_cluster_solutions, left_index=True, right_index=True)
        
        # sw = []
        
        # # Create clustering objects
        # cls1 = KMeans(n_clusters=n, random_state=0)
        # cls2 = KMedoids(n_clusters=n, random_state=0)
        # cls3 = AgglomerativeClustering(n_clusters=n,
        #                             affinity='euclidean',
        #                             linkage='ward')
        #     # Agglomerative clustering: if linkage=ward, affinity must be Euclidean
        # cls_algs = [['kMeans', cls1],
        #             ['kMedoids', cls2],
        #             ['Hierarchical', cls3]]
        
        # # Fit and score clustering solutions for i clusters w/ each algorithm
        # for cls in cls_algs:
            
        #     # Fit the model to the factor analysis scores
        #     cls[1].fit(scores)
            
        #     # List of assigned clusters
        #     clusters = cls[1].fit_predict(scores)
            
        #     # Silhouette scores for each solution
        #     silhouette_avg = silhouette_score(scores,clusters)
            
        #     # Store solution info
        #     algorithm = cls[0]
        #     n_stats = [algorithm, n, silhouette_avg, clusters]
        #     sw.append(n_stats)

        # # Reorder cluster lists by descending silhouette scores.
        # # Clusters in first element should be assigned to training data.
        # sw = sorted(sw, key=itemgetter(2), reverse=True)
        # op[f'Optimal {sw[0][1]} cluster solution ({sw[0][0]})'] = sw[0][3] + 1


        
    # #**********************************************************************#
    # # This is where the classification stuff begins

    for i in range(op_base_column_count+1, op_base_column_count+max_clusters-1):
        
        df_cl = op.iloc[:,np.r_[2:op_base_column_count,i]]  # i is the current cluster solution
        df_cl_const = op.iloc[:,np.r_[1:op_base_column_count,i]]  # i is the current cluster solution

    #     #**********************************************************************#

    #     # Split data into 70% training, 30% validation
    #     train, valid = train_test_split(df_cl, test_size=0.30, random_state=123)

    #     # X is unlabeled training data, y is true training labels 
    #     X, y = train.iloc[:,0:-1], train.iloc[:,-1]

    #     X_valid, y_valid = valid.iloc[:,0:-1], valid.iloc[:,-1]

    #     #**********************************************************************#

    #     # Get variable importances

    #     clf1 = RandomForestClassifier(random_state=0)
    #     clf2 = GradientBoostingClassifier(random_state=0)

    #     classifiers = [['rf', clf1], ['gbt', clf2]]

    #     for classifier in classifiers:    
    #         # Fit classifier to training data
    #         classifier[1].fit(X,y)    

    #     # Create variable importance dataframe
    #     num_vars = list(range(1,len(clf1.feature_importances_)+1))
    #     importance = pd.DataFrame({'variable': num_vars,
    #                             'rf': clf1.feature_importances_,
    #                             'gbt': clf2.feature_importances_,})

    #     # Average variable importance of rf and gbt models
    #     importance['avg'] = (importance['rf']+importance['gbt'])/2

    #     # Put avg importances on a scale from 0 to 1 to make it easier to visualize
    #     importance['Relative Importance'] = np.interp(importance['avg'],
    #                                                 (importance['avg'].min(),
    #                                                 importance['avg'].max()),
    #                                                 (0, 1))

    #     # View top 10 variables when RF and GBT models are averaged
    #     top_10_avg = importance.sort_values(by='avg', ascending=False)[['avg','Relative Importance']].head(10)

    #     # Add variable rank column to dataframe
    #     importance_rank = num_vars
    #     importance = importance.sort_values(by='Relative Importance', ascending=False)
    #     importance['rank'] = importance_rank
    #     importance.reset_index(inplace=True)

    #     # Save index of top 5 variables (not the variable number!)
    #     top_5 = importance[importance['rank'] <= 5]['index']

        #**********************************************************************#
        # Average and Standard Deviations for each cluster/variable combination
        # For cluster 1 of 2, calculate the average and stdev for each variable
        # For cluster 2 of 2, calculate the average and stdev for each variable
        # Etc.
        
        if n == max_clusters:
        
            cls_avg_list = []

            # Take the mean of every variable for each cluster
            for k in range(1, df_cl.iloc[:,-1].max()+1):
                cls_mean = pd.Series({"Count":df_cl[df_cl.iloc[:,-1] == k].iloc[:,0:-1].shape[0]})
                cls_mean = cls_mean.append(pd.Series({"Const":op[op.iloc[:,-1] == k].loc[:,'Const'].mean()}))
                cls_mean = cls_mean.append(df_cl[df_cl.iloc[:,-1] == k].iloc[:,0:-1].mean())
                cls_avg_list.append(cls_mean)
                cls_std = pd.Series({"Const":op[op.iloc[:,-1] == k].loc[:,'Const'].std()})
                cls_std = cls_std.append(df_cl[df_cl.iloc[:,-1] == k].iloc[:,0:-1].std())
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
        
        
    #     #**********************************************************************#
    #     # Convert data to binary, train classifiers, score validation, create maps

    #     # Convert X, X_valid, and df_cl predictors to all 1 and -1
    #     X = (X.mask(df > 0, other=1, inplace=False)
    #         .mask(df <= 0, other=-1, inplace=False))
    #     X_valid = (X_valid.mask(df > 0, other=1, inplace=False)
    #             .mask(df <= 0, other=-1, inplace=False))
    #     all_data_masked = (df_cl.iloc[:,0:-1].mask(df > 0, other=1, inplace=False)
    #                     .mask(df <= 0, other=-1, inplace=False))

    #     map_collection = []

    #     # Retrain on the 2-5 most important variables
    #     for j in range(2,6):

    #         clf_scores = []

    #         clf1 = RandomForestClassifier(random_state=0)
    #         clf2 = GradientBoostingClassifier(random_state=0)
    #         clf3 = SVC(random_state=0)
    #         clf4 = KNeighborsClassifier()

    #         classifiers = [['rf', clf1], ['gbt', clf2], ['svc', clf3], ['knn', clf4]]
            
    #         # Fit each classifier to the current variable/cluster combination
    #         for classifier in classifiers:

    #             # Fit classifier to training data
    #             classifier[1].fit(X.iloc[:,np.r_[top_5[0:j]]],y)

    #             # Store classifier-specific results [algorithm object, classifier name, scores]
    #             results = [classifier[1],
    #                     classifier[0],
    #                     classifier[1].score(X_valid.iloc[:,np.r_[top_5[0:j]]],y_valid)]

    #             # Overall classifier results
    #             clf_scores.append(results)

    #         # Sort classifier accuracy in descending order
    #         clf_scores = sorted(clf_scores, key=itemgetter(2), reverse=True)
    #         # clf_scores[0][0] is the best model
            
    #         # Fit the best model on all data
    #         best_model = clf_scores[0][0].fit(all_data_masked.iloc[:,np.r_[top_5[0:j]]], df_cl.iloc[:,-1])

    #         #******************************************************************#
    #         # Create mappings
            
    #         # Creates grid of dimension j
    #         grid = pd.DataFrame(list(itertools.product([-1,1], repeat=j)))
            
    #         grid.columns = all_data_masked.iloc[:,np.r_[top_5[0:j]]].columns

    #         # This is the best model predicting the grid
    #         preds = best_model.predict(grid)            

    #         # Add to grid dataframe
    #         grid['Predicted Cluster'] = preds

    #         # Change grid to mapping to fit into the rest of the code
    #         mapping = grid

    #         # Save current mapping to map collection for this cluster solution
    #         map_collection.append(mapping)

    #         # Write each dataframe to a different worksheet.
    #         mapping.to_excel(writer, index=False, sheet_name=f"{df_cl.columns[-1][8:17]}s, {j} vars, {round(clf_scores[0][2]*100)}% Acc.")

    #     all_maps.append(map_collection)

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