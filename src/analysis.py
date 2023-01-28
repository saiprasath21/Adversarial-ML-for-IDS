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
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from collections import Counter
# cd /home/sai/Desktop/ICS_IDS/My_code/src

from data_preprocessing import DataAnalysis

file_name = "impute_data.csv"
directory = "../data/" + file_name
data = pd.read_csv(directory)

top_10 = ['CRC Rate', 'Reset', 'Setpoint', 'Cycle time', 'Deadband', 'Gain', 'Length', 'Rate', 'Function_16.0', 'Function_3.0', 'Binary Result', 'Cat Result']
attack_bin = ['Normal', 'Attack']
attacks_cat = ['Normal', 'NMRI', 'CMRI','MSCI', 'MPCI', 'MDCI', 'DOS', 'Recon']




def find_interval(da):
    '''
    
    Finding the intervals where the anomalous packets occur and analysing them

    '''
    result = da.pdf['Binary Result']
    result = np.array(result)
    interval = []
    flag=0
    count=0
    middle_0=0
    index = 0
    fc = 0
    interval_data = []
    for i in result:

        if(i==0 and flag==0):
            count=0
            flag=0
        
        elif(i==1 and flag==0):
            flag=1
            count=1
            middle_0=0
        
        elif(i==1 and flag==1):
            count += 1
            middle_0=0
            interval_data.append(da.pdf.loc[index,:])
        
        elif(i==0 and flag==1):
            middle_0 += 1
            count += 1
            interval_data.append(da.pdf.loc[index,:])
            if(middle_0>3):
                flag=0
                if(count>100):
                    interval.append(count)
                count=0
        
        index += 1

    print(Counter(interval), len(interval), sum(interval), len(result), fc)
    interval_data = pd.DataFrame(interval_data)
    print(interval_data)

def make_dataset(data, time=0, address=0, label=1, mix=0):
    
    '''
    This section was used to analyse the use of time parameter. After analysis the time parameter will be dropped along with 
    the slave address 
    '''

    file_name = ''

    if(time==0):
        data = data.drop(['Time'], axis=1)
        file_name += 'notime'

    elif(time==1):
        file_name += 'yestime'

    elif(time==2): # replacing with the time difference
        file_name += 'timedifference'
        time = data['Time']
        time_diff = np.zeros((len(time),1))
        for i in range(1,len(time)):
            time_diff[i] = time[i]-time[i-1]
        data['Time']=time_diff

    file_name += '---'

    if(address==0):
        data = data.drop(['Slave address'], axis=1)
        file_name += 'noaddress'
    elif(address==1):
        file_name += 'yesaddress'

    file_name += '---'   
    
    df = np.array(data)
    X = df[:, :-2]
    y = []
    l = []
    if(label==0):
        y = df[:, -2]
        l = attack_bin
        file_name += 'binary'
    elif(label==1):
        y = df[:, -1]
        l = attacks_cat
        file_name += 'cat'

    file_name += '---'

    if(mix==0):
        X, y = sklearn.utils.shuffle(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        file_name += 'random'

    elif(mix==1):
        size = int(len(X)*0.8)
        X_train = X[0:size,:]
        X_test = X[size:,:]
        y_train = y[0:size]
        y_test = y[size:]
        file_name += 'direct'
    
    
    df = DataAnalysis(data, file_name)
    return [df, X_train, X_test, y_train, y_test, l, file_name]


def verify(y_true, y_pred, l):
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=5, target_names=l))


def random_forest(info, cross_validation_score=False, importance=True, write=True):

    df, X_train, X_test, y_train, y_test, l, file_name = info

    clf = RandomForestClassifier(n_estimators=30, max_depth=45)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)


    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=5, target_names=l))

    imp = []
    if(importance==True):
        imp = [[df.cnames[i],round(100*clf.feature_importances_[i],2)] for i in range(len(clf.feature_importances_))]
        imp.sort(key=lambda x: x[1])
        imp = np.array(imp)
        imp = imp[-10:, :]
        print(imp)


    if(cross_validation_score==True):
        scores = cross_val_score(clf, X, y, cv=10)
        print("%0.4f accuracy with a standard deviation of %0.4f" % (scores.mean(), scores.std()))
    
    if(write==True):
        directory = df.results + "/rf/" + file_name + ".txt" 
        f = open(directory,'w')
        f.write(file_name  + "\n \n")
        f.write("Accuray training data = "+str(accuracy_score(y_train, y_train_pred))+"\n")
        f.write("Accuray testing data = "+str(accuracy_score(y_test, y_pred))+"\n \n")
        
        cm = confusion_matrix(y_test, y_pred)
        arr = np.array(cm)
        content = str(arr)
        f.write(content)

        f.write("\n \n")

        f.write(classification_report(y_test, y_pred, digits=5)+"\n")
        if(importance==True):
            imp = list(imp)
            for i in imp:
                f.write(" ".join(i))
                f.write("\n")
        f.close()


#Generating results for the random forest classifier
'''
for time in {0,1,2}:
    for address in {0,1}:
        for label in {0,1}:
            for mix in {0,1}:
                print(time, address, label, mix)
                random_forest(make_dataset(data, time, address, label, mix))
'''


crc_mean = np.mean(np.array(data['CRC Rate']))
crc_std = np.std(np.array(data['CRC Rate']))
print(crc_mean, crc_std)