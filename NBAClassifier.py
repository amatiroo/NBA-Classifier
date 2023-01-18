import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing



#Reading the csv file
datanba =pd.read_csv('nba2021.csv')


#Making the position data numerical so that the model can understand
datanba['Pos'].replace({
    'PF':0,
    'PG':1,
    'C':2,
    'SG':3,
    'SF':4,
    
},inplace=True)

datanba.shape


cols = ['Player','Age','FG','FGA','2P','3PA','2PA','FT','G','FTA','Tm','3P','TOV','PF','STL','GS','MP'] # these are the features that will not be included
datanba=datanba.drop(cols,axis=1) # dropping the not so important feature columns
datanba.fillna(0, inplace=True) # fill na as 0
y = datanba.iloc[:,0] #setting the target data i.e Position as y
datanba.pop('Pos') 
#Normalize the feature columns
d = preprocessing.normalize(datanba, axis=0)
df = pd.DataFrame(d, columns=datanba.columns)
df.head()
datanba = df

X = datanba  # feature columns as X


print("\nThe below is using Random state just to show i have reached this accuracy\n")
#splitting the train test data
train_feature, test_feature, train_class, test_class = train_test_split(X, y, test_size=0.25, random_state=12)

#Buliding the model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_feature, train_class)
print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))


#Confusion Matrix
prediction = knn.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['Actual'], colnames=['Predicted'], margins=True))


#Cross Validation
print("10 fold cross validation results")
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
print("Average score of cross-Validation ")
print(scores.mean())

print("\nThe below is without using Random state , the accuracy keeps changing everytime\n")
train_feature, test_feature, train_class, test_class = train_test_split(X, y, test_size=0.25)


knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(train_feature, train_class)
print("Test set accuracy: {:.2f}".format(knn.score(test_feature, test_class)))



prediction = knn.predict(test_feature)
print("Confusion matrix:")
print(pd.crosstab(test_class, prediction, rownames=['Actual'], colnames=['Predicted'], margins=True))


print("10 fold cross validation results")
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print(scores)
print("Average score of cross-Validation ")
print(scores.mean())