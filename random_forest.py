import matplotlib.pylab as plt
import numpy as np
import os
import sklearn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import  KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import itertools
from utils import plot_confusion_matrix
from sklearn.decomposition import PCA

# Set the number of decimal houses to show when printing
np.set_printoptions(precision = 2)


# Path to CSV feature file  - 'Neuroimaging', 'Clinical',   'Neuroimaging' + 'Clinical'
data_path = "./procan_ml_input_04.csv"

# Loading data
missing_values = ["NaN"]
df = pd.read_csv(data_path, header = None,na_values = missing_values)
data = df.apply(pd.to_numeric, errors='coerce').values

# Splitting features and labels into to arrays
feats = data[:,:-1]
labels = data[:,-1].astype(int)


#Obtain mean of columns as you need disregarding NaN values
col_mean = np.nanmean(feats, axis=0)


# Remove samples with more than  25 features
to_remove = ~((np.isnan(feats).sum(axis = 1)) > 25 )
feats = feats[to_remove]
labels = labels[to_remove]

#Find indicies that you need to replace
inds = np.where(np.isnan(feats))

#Place column means in the indices. Align the arrays using take
feats[inds] = np.take(col_mean, inds[1])

print("Number of samples:", feats.shape[0])
print("Number of features per sample:", feats.shape[1])
print("Number of classes:", np.unique(labels).size)
print("Classes:", np.unique(labels))
for ii in np.unique(labels):
    print("Number of samples class %d: %d" %(ii,(labels == ii).sum()))

# Remove features which are the same for all samples
indexes = np.where(feats.max(axis = 0) == 0)
print(indexes)

feats = np.delete(feats,indexes,axis = 1)



# Let's look into principal components
# Min-max normalization
feats_norm = (feats - feats.min(axis = 0))/(feats.max(axis = 0) - feats.min(axis = 0))
pca = PCA(n_components=5)
pca.fit(feats_norm)
print("Explained variange by each PCA component: \n", pca.explained_variance_ratio_)  
feats_2d = pca.transform(feats_norm)

plt.figure(dpi = 300)
plt.scatter(feats_2d[labels==0,0],feats_2d[labels==0,1],marker="x",c = "y")
plt.scatter(feats_2d[labels==1,0],feats_2d[labels==1,1],marker="o",c = "g")
plt.scatter(feats_2d[labels==2,0],feats_2d[labels==2,1],marker="^",c = "b")
plt.scatter(feats_2d[labels==3,0],feats_2d[labels==3,1],marker="d",c = "r")
plt.xlabel("PCA 1 - explained variance - %f" %pca.explained_variance_ratio_[0])
plt.ylabel("PCA 2 - explained variance - %f" %pca.explained_variance_ratio_[1])
plt.grid()
plt.show()


# Cross-validation parameters
nfolds = labels.size # Leave-one-out cross-validation
nfolds2 = 4 # Inner number of folds in the cross-validation 

# Compute confusion matrix
class_names = ['0','1','2','3'] # Name for the labels: control, risk 01, risk 02

# Variable to store the confusion matrix
cm = np.zeros((4,4),dtype = int)

# Random Forest
feature_importance = np.zeros((nfolds,feats.shape[1]))


# Grid Search parameters
# Number of trees in random forest
n_estimators = [80]#10,20,40,80]
# Number of features to consider at every split
max_features = ['auto']#, 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 16)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}


# K-fold cross-validation
skf = KFold(n_splits = nfolds,shuffle=True)
counter = 0
counter2 = 0
accu = np.zeros(nfolds)
yp = np.zeros(feats.shape[0])
yt = np.zeros(feats.shape[0])

for train_index, test_index in skf.split(feats):
    # Split train and test sets 
    X_train, X_test = feats[train_index], feats[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # Grid search
    clf = GridSearchCV(RandomForestClassifier(), random_grid, cv=nfolds2,
                       scoring='accuracy')
    clf.fit(X_train, y_train)

    # Best model in the grid search
    clf = clf.best_estimator_


    # Prediction and label storage
    y_pred = clf.predict(X_test)
    yp[counter:counter+test_index.size] = y_pred
    yt[counter:counter+test_index.size] = y_test
    accu[counter2] = accuracy_score(y_test,y_pred)
    counter2+=1
    counter+=test_index.size

cm = confusion_matrix(yp,yt)
print("Average accuracy: %f +/- %f" %(accu.mean(),accu.std()))

# Plot normalized confusion matrix
plt.figure(dpi =300)
plot_confusion_matrix(cm, classes=class_names, normalize=True,
                     title='Normalized confusion matrix')
plt.show() 


# Plot non-normalized confusion matrix
plt.figure(dpi =300)
plot_confusion_matrix(cm, classes=class_names, normalize=False,
                      title='Normalized confusion matrix')
plt.show()

# ## Features Importance
clf = GridSearchCV(RandomForestClassifier(),random_grid,cv=nfolds2,scoring='accuracy')
clf.fit(feats, labels)
clf = clf.best_estimator_

fi = clf.feature_importances_

#Plotting 10 most important features
indices = np.argsort(fi)[::-1]
plt.figure()
plt.title("Feature importances")
plt.plot(fi[indices][0:10])
plt.xticks(range(10), indices[:10])
plt.xlim([-1, 10])
plt.show()
print(indices[:10].sum())
