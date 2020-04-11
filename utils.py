import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Coordinate change operations
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


# Given predictions and test labels, create the confusion matrix
def confusion_matrix_dataframe(predictions, test_labels):
    predictions = np.nan_to_num(predictions)
    matrix = metrics.confusion_matrix(np.argmax(test_labels.to_numpy(), axis = 1), 
                                  np.argmax(predictions, axis = 1))
    matrix_df = pd.DataFrame(matrix)
    matrix_df.columns = ['AGRICULTURE', 'INDUSTRIAL', 'OFFICE', 'OTHER', 'PUBLIC','RESIDENTIAL', 'RETAIL']
    matrix_df.rename(index={0 : 'AGRICULTURE',
                        1 : 'INDUSTRIAL',
                        2 : 'OFFICE',
                        3 : 'OTHER',
                        4 : 'PUBLIC',
                        5 : 'RESIDENTIAL',
                        6 : 'RETAIL'}, inplace=True)
    return matrix_df


# Extracts rgbnir channels from DataFrame
def extract_channels(df, drop_tails = False, return_class_for_each = False, return_class_df = True):
    channels = ["Q_R","Q_G","Q_B","Q_NIR"]
    channel_dfs = []
    for c in channels:
        col_filter = df.columns.str.contains(c)
        if drop_tails:
            col_filter = col_filter * np.invert(df.columns.str.endswith("_0"))
        if return_class_for_each:
            index = list(df.columns).index("CLASE")
            col_filter[index] = True
        channel_dfs.append(df[df.columns[col_filter]])
    if return_class_df:
        channel_dfs.append(df["CLASE"])
    return channel_dfs
    
# Merge for cnn1
def merge_for_Cnn1(df_r, df_g, df_b, df_nir ):
    merge = []
    for a,b,c,d in zip(df_r.values,df_g.values,df_b.values,df_nir.values):
        row = np.vstack((a,b,c,d)).T
        merge.append(row)
    merge = np.array(merge)
    return merge


# Normalize
def normalizer(df):
    return (df - df.mean())/df.std()


#Standarizes dataset by sets of columns
def process_data(data):
    
    # Remove the nulls
    data = data.dropna()
    # replace values in CADASTRALQUALITYID
    replacement  = {"A": 12, "B" : 11, "C" : 10, "1": 9, "2" : 8, "3" : 7, "4" : 6, "5" : 5,"6" : 4,"7" : 3,"8" : 2,"9" : 1}
    data = data.replace(replacement)
    #Scale by coordinates
    data = standard_scale_several_columns(data, ["X","Y"])
    #Scale by rgbnir
    data = standard_scale_several_columns(data, data.columns[data.columns.str.contains("Q_")])
    #Scale by geom_properties
    data = standard_scale_several_columns(data, ['GEOM_R1',  'GEOM_R2',  'GEOM_R3',  'GEOM_R4'])
    
    for col in ["AREA", "CADASTRALQUALITYID", "MAXBUILDINGFLOOR", "CONTRUCTIONYEAR"]:
        data = standard_scale_several_columns(data,[col])

    #One hot encode CLASE
    enc = OneHotEncoder(handle_unknown='ignore', sparse = False)
    df_cls = data.CLASE.values.reshape(-1, 1)
    df_cls_encoded = pd.DataFrame(enc.fit_transform(df_cls), columns = enc.categories_[0])
    df_cls_encoded.index = data.index

    #Append and concatenate
    data_processed = pd.merge(data, df_cls_encoded, left_index=True, right_index=True)
    return data_processed


# Prepare data for network
def prepare_data_for_network(data, 
                             columns_dense = ['X', 'Y','AREA', 'GEOM_R1', 'GEOM_R2', 'GEOM_R3', 'GEOM_R4', 
                                              'CONTRUCTIONYEAR', 'MAXBUILDINGFLOOR', 'CADASTRALQUALITYID'],
                             columns_cnn = ['AGRICULTURE', 'INDUSTRIAL', 'OFFICE', 'OTHER', 'PUBLIC','RESIDENTIAL', 'RETAIL'],
                             process_data_bool = False):
    if process_data_bool:
        data = process_data(data)

    data_dense = data[columns_dense]
    labels = data[columns_cnn]

    df_r, df_g, df_b, df_nir = extract_channels(data, True, False, False)
    data_rgbnir = merge_for_Cnn1(df_r, df_g, df_b, df_nir)
    return data_dense, data_rgbnir, labels


# Plots the means of the rgbnir channels of the data by terrain type
def plot_all_channels(data):
    terrain_types = data.index.unique()
    size = len(terrain_types)    
    fig, ax  = plt.subplots(size,4,figsize=(35, 5*size))
    fig.suptitle("Mean of channels ")
    for i in range(size):
        terrain = terrain_types[i]
        r, g, b, nir = extract_channels(data, False, False, False)
        ax[i][0].set_title("R channel-" + terrain)
        ax[i][1].set_title("G channel-" + terrain)
        ax[i][2].set_title("B channel-" + terrain)
        ax[i][3].set_title("NIR channel-" + terrain)
        ax[i][0].plot(r.filter(like = terrain, axis = 0).iloc[0], color = "red")
        ax[i][1].plot(g.filter(like = terrain, axis = 0).iloc[0], color = "green")
        ax[i][2].plot(b.filter(like = terrain, axis = 0).iloc[0], color = "blue")
        ax[i][3].plot(nir.filter(like = terrain, axis = 0).iloc[0], color = "brown")

        
# Plots the means of the rgbnir channels of the data given a specific terrain type
def plot_channels(terrain, data):
    r, g, b, nir = extract_channels(data)
    fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2,figsize=(20,10))
    fig.suptitle("Mean of channels for " + terrain)
    ax1.set_title("R channel")
    ax2.set_title("G channel")
    ax3.set_title("B channel")
    ax4.set_title("NIR channel")
    ax1.plot(r.filter(like = terrain, axis = 0).iloc[0], color = "red")
    ax2.plot(g.filter(like = terrain, axis = 0).iloc[0], color = "green")
    ax3.plot(b.filter(like = terrain, axis = 0).iloc[0], color = "blue")
    ax4.plot(nir.filter(like = terrain, axis = 0).iloc[0], color = "brown")


# Plots the correlation matrix of the given dataframe
def plot_corr(data, title = None):
    fig, ax = plt.subplots(figsize=(14,14))
    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
    )
    if title is not None:
        ax.set_title(title,fontdict = {'fontsize':30}) 
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );

    
# Pol to cart
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
    
# Standarize some columns of a dataframe taking all the values within the columns together 
def standard_scale_several_columns(data, columns):
    df = data[columns]
    df = (df - df.mean())/df.std()
    data[columns] = df
    return data
