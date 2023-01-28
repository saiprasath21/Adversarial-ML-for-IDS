import pandas as pd
import numpy as np
import sklearn
from sklearn import tree
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
# cd /home/sai/Desktop/ICS_IDS/My_code/src

class DataAnalysis():
    
    def __init__(self, df, file_name):
        self.pdf = df
        self.df = np.array(df)
        self.r = np.shape(df)[0]
        self.c = np.shape(df)[1]
        self.cnames = df.columns
        self.file_name = file_name
        self.data = "/home/sai/Desktop/ICS_IDS/My_code/data"
        self.src = "/home/sai/Desktop/ICS_IDS/My_code/src"
        self.results = "/home/sai/Desktop/ICS_IDS/My_code/results"
        #self.basic_info()

    # function that performs the task of Collections.Counter
    def basic_info(self):
        print("Shape of the data = {}".format(np.shape(self.pdf)))

        print("Binary Attack Classification")
        self.counter('Binary Result')

        # print("Categorized Attack Classification")
        # self.counter('Categorized Result')

        # print("Specific Attack Classification")
        # self.counter('Specific Result')

    def generate_datasets(self):
        self.fill_missingvalues('impute')
        self.fill_missingvalues('zero_indicator')

    def counter(self,feature):
        y = self.pdf[feature]
        counter = {}
        for i in y:

            if i in counter.keys():
                counter[i] += 1
            else:
                counter[i] = 1
        
        for i in sorted(counter):
            print(i, counter[i])

    def fill_missingvalues(self, type):

        if(type=='impute'):

            base_features = self.df[0,:].copy()
            safe_copy = self.df.copy()
            index = -1
            for i in range(1, self.r):
                if(index<0):
                    flag=1
                    for k in base_features:
                        if(k == "?"):
                            flag=0
                    if(flag==1):
                        index=i-1

                for j in range(self.c):
                    if(self.df[i,j]=='?'):
                        self.df[i,j]=base_features[j]
                    else:
                        base_features[j] = self.df[i,j]
            
            if(index>0):
                base_features = self.df[index,:].copy()
                for i in range(index-1,-1,-1):
                    for j in range(self.c):
                        if(self.df[i,j]=="?"):
                            self.df[i,j]=base_features[j]
                        else:
                            base_features[j] = self.df[i,j]

            f_name = "impute_" + self.file_name
            directory = self.data + "/" +f_name

            df_copy = pd.DataFrame(self.df)
            df_copy.columns = self.cnames
            df_copy.to_csv(directory, index=False)
            self.df = safe_copy
            print("Successfully created Impute Dataset")

        elif(type=="zero_indicator"):
            max_qm = 0
            max_qm_names = []
            max_qm_index = []
            safe_copy = self.df.copy()
            for i in range(self.r):
                count = 0
                names = []
                index = []
                for j in range(self.c):
                    if(self.df[i,j]=='?'):
                        count += 1
                        names.append(self.cnames[j])
                        index.append(j)
                if(count > max_qm):
                    max_qm = count
                    max_qm_names = names
                    max_qm_index = index
            
            print(max_qm_index, max_qm_names)

            zero_indicator = np.zeros((self.r, max_qm))
            for i in range(self.r):
                k = 0
                for j in max_qm_index:
                    if(self.df[i,j]=='?'):
                        self.df[i,j] = 0
                        zero_indicator[i, k] = 1
                    k += 1
                
            self.df = np.concatenate([self.df, zero_indicator], axis=1)

            
            indicator_features = list(self.cnames)
            for i in max_qm_names:
                s = "indicator_" + i
                indicator_features.append(s)
            df_copy = pd.DataFrame(self.df)
            df_copy.columns = indicator_features
            f_name = "zero_indicator_" + self.file_name
            directory = self.data + "/" +f_name
            df_copy.to_csv(directory, index=False)
            self.df = safe_copy
            print("Successfully created Zero Indicator Dataset")

# Reading the file in ARFF format
#file_name = "processed_full_data_3_notime.csv"
file_name = "processed_data.csv"
directory = "../data/" + file_name
df = pd.read_csv(directory)
da = DataAnalysis(df, file_name)

# Use this command to intially generate datasets with 2 filling tehcniques for missing values 
# da.generate_datasets()


# analysing the missing values present in the data. Note that these functions take the pandas dataframe as input.

def find_missing_index(df):
    qm_index = []
    for i in range(0,4):
        missing = []
        for j in df.columns:
            if(df.loc[i,j]=="?"):
                missing.append(j)
        qm_index.append(missing)
    return qm_index

def check_pattern(df):
    count=0
    prev_count=count
    pattern_mismatch=0
    qm_index=find_missing_index(df)
    for i in range(0, np.shape(df)[0],4):
        for j in range(i, i+4):
            for k in qm_index[j-i]:
                if(df.loc[j,k] == "?"):
                    count += 1

        if(prev_count!=count):
            pattern_mismatch += 1
            prev_count = count
    
    if(not pattern_mismatch):
        print("All packets follows the same order")
    else:
        print("No such pattern exist")
    return pattern_mismatch

def process_dataset(da):
    pro_df = da.df.copy()
    
    pro_df = np.array(pro_df)
    # only the continuous and binary features are in this list
    feature_order = [2, 3, 4, 5, 6, 7, 8, 13, 14, 16, 0, 10, 11, 12, 15]

    pro_df[:,0:1] = preprocessing.label_binarize(pro_df[:,0:1], classes=[4])
    continuous_var = np.concatenate([pro_df[:, 2:9],pro_df[:,13:15], pro_df[:,16:17]], axis=1)
    discrete_var_binary = np.concatenate([pro_df[:, 0:1], pro_df[:,10:13],pro_df[:,15:16]],axis=1)
    discrete_var_multi = np.concatenate([pro_df[:, 1:2], pro_df[:,9:10]],axis=1)
    #only categorical features
    categorical = [1, 9]

    # Processing all the features individually
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(continuous_var)
    continuous_var = scaler.transform(continuous_var)

    # One-hot encode all the categorical features
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder()
    print(discrete_var_multi)
    discrete_var_multi = encoder.fit_transform(discrete_var_multi).toarray()
    print(discrete_var_binary)
    print(encoder.categories_)

    feature_names = []
    for i in feature_order:
        feature_names.append(da.cnames[i])

    for i in range(0,len(encoder.categories_)):
        base_name = da.cnames[categorical[i]]
        for j in encoder.categories_[i]:
            s = base_name + "_" + str(j)
            feature_names.append(s)


    # Combining them into a single dataframe
    labels_binary = pro_df[:, 17:18]
    labels_cat = pro_df[:, 18:19]
    feature_names.append('Binary Result')
    feature_names.append('Cat Result')
    final_df = np.concatenate([continuous_var, discrete_var_binary, discrete_var_multi, labels_binary, labels_cat], axis=1)
    final_df = pd.DataFrame(final_df)
    print(final_df, feature_names, len(feature_names))
    final_df.columns = feature_names
    directory = da.data + "/" + "processed_full_data_final.csv"
    final_df.to_csv(directory, index=False)

# Used to create the processed dataset
# process_dataset(da)

# Basic Data Analysis

def dataset_distribution_analysis(df):
    f = open('processed_data_analysis.txt','w')
    #print(df.groupby(['Function','Categorized Result']).size())
    #print(df.groupby(['Slave address','Cat Result']).size())
    #print(df.groupby(['System Mode', 'Binary Result']).size())
    f.write(df.nunique().to_string() + "\n")
    f.write(df.groupby(['Binary Result', 'CRC Rate' ]).size().to_string() + "\n")
    f.write(df.groupby(['Binary Result', 'Setpoint' ]).size().to_string()+ "\n")
    f.write(df.groupby(['Binary Result', 'Cycle time']).size().to_string()+ "\n")
    f.write(df.groupby(['Binary Result', 'Reset']).size().to_string()+ "\n")
    f.write(df.groupby(['Binary Result', 'Length']).size().to_string()+ "\n")
    f.write(df.groupby(['Binary Result', 'Gain']).size().to_string()+ "\n")

#dataset_distribution_analysis(da.pdf)

'''

duplicated_values = da.pdf.duplicated()
print(duplicated_values.sum())
print(da.pdf[duplicated_values])

==> No duplicate values

sns.heatmap(da.pdf.corr(), cmap='RdYlGn')
plt.show()

==> No major correlation between the features
'''
