import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from pydotplus import graph_from_dot_data
import webbrowser
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore")
import os
os.environ["PATH"] += os.pathsep + r'C:\Users\yueya\Anaconda3\Library\bin\graphviz/'

#importing data
tt = pd.read_csv(r'data.csv', sep=',',header=0)
#%%-----------------------------------------------------------------------

#data preprocessing
# look at first few rows
tt.head()

# replace missing characters as NaN
tt.replace('?', np.NaN, inplace=True)

# check the structure of mushroom_data
tt.info()

# check the null values in each column
print(tt.isnull().sum())

# check the summary of the mushroom_data
tt.describe(include='all')

# replace categorical mushroom_data with the most frequent value in that column
tt = tt.apply(lambda x: x.fillna(x.value_counts().index[0]))

# drop Cabin, ticket and name of passenger
tt.drop(['Cabin','Name', 'Ticket','Embarked'], axis=1, inplace=True)

# again check the null values in each column
print(tt.isnull().sum())

#Encoder features
X_data = pd.get_dummies(tt.iloc[:, 1:])
X = X_data.values
Y = tt.values[:, 0]
Y = Y.astype('int')
#%%-----------------------------------------------------------------------

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)

# data visualization
ntrain=tt.shape[0]
import seaborn as sns
train=tt[:ntrain]
sns.pairplot(train,vars=['Age','Sex','Pclass',],hue='Survived',)
plt.title("data visualization")
plt.show()

# data standardize
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

#%%-----------------------------------------------------------------------

# Decision Tree

# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)

# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
#prediction
y_pred_gini = clf_gini.predict(X_test)

y_pred_entropy = clf_entropy.predict(X_test)

# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')

# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')

# confusion matrix for gini model
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = tt.Survived.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.title("Decision Tree - Gini")
plt.tight_layout()
plt.show()

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = tt.Survived.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.title("Decision Tree - Entropy")
plt.tight_layout()
plt.show()

# display decision tree
dot_data = tree.export_graphviz(clf_gini, filled=True, rounded=True, class_names='survived', feature_names=tt.iloc[:, 0:].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_gini.pdf")
webbrowser.open_new(r'decision_tree_gini.pdf')

dot_data = tree.export_graphviz(clf_entropy, filled=True, rounded=True, class_names='survived', feature_names=tt.iloc[:, 0:].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy.pdf")
webbrowser.open_new(r'decision_tree_entropy.pdf')