import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report,accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score, f1_score, log_loss
from sklearn.metrics import classification_report,confusion_matrix, precision_recall_fscore_support 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn import grid_search, svm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore') 
df = pd.read_csv('mycsvdata.csv')
df.head()
df.columns
result=df.dtypes
print(result)
#Data Mapping
df.drop('id',axis=1,inplace = True)     
#Data Mapping
df['classification'] = df['classification'].map({'ckd':1,'\tckd':1,'notckd':0,'\tnotckd':0})
df['htn'] = df['htn'].map({'yes':1,'no':0})
df['dm'] = df['dm'].map({'yes':1,'no':0,'\tyes':1,'\tno':0})
df['cad'] = df['cad'].map({'yes':1,'\tno':0,'no':0})
df['appet'] = df['appet'].map({'good':1,'poor':0})
df['ane'] = df['ane'].map({'yes':1,'no':0})
df['pe'] = df['pe'].map({'yes':1,'no':0})
df['ba'] = df['ba'].map({'present':1,'notpresent':0})
df['pcc'] = df['pcc'].map({'present':1,'notpresent':0})
df['pc'] = df['pc'].map({'abnormal':1,'normal':0})
df['rbc'] = df['rbc'].map({'abnormal':1,'normal':0})
print(len(df.columns))
sns.heatmap(df.isnull(), cbar=False)
df['classification'].value_counts()
plt.figure(figsize = (19,19))
sns.heatmap(df.corr(), annot = True, cmap = 'coolwarm') # looking for strong correlations with "class" row
#df.drop(['pcv','wbcc','rbcc'], axis=1, inplace=True)
df.fillna(round(df.mean()),inplace=True)
sns.heatmap(df.isnull(), cbar=False)
df.plot(kind='density', subplots=True, layout=(100,100), sharex=False,sharey=False)
plt.show()
df.hist()
plt.show()
# Select upper triangle of correlation matrix
corr_matrix = df.corr().abs()
upper = np.triu(np.ones_like(corr_matrix,dtype=bool))
k=corr_matrix.mask(upper)
# Find index of feature columns with correlation greater than 0.4
to_drop = [column for column in k.columns if any(k[column] > 0.9)]
# Drop features 
dropped=df.drop(df[to_drop], axis=1)
print(to_drop)
print(len(dropped.columns))

"""X = df.drop("classification", axis=1)
y = df["classification"]"""
X = dropped.iloc[:, :-1].values # attributes to determine dependent variable / Class
y = dropped.iloc[:, -1].values# dependent variable / Class
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=0.20, random_state=43)

#pca
sc = StandardScaler() 
  
X1_train = sc.fit_transform(X1_train) 
X1_test = sc.transform(X1_test) 

pca = PCA() 
  
X1_train = pca.fit_transform(X1_train) 
X1_test = pca.transform(X1_test) 

x = np.random.randint(low = 0, high = 15, size=100) 

plt.figure()
plt.hist(x)
plt.show()

#explained_variance = pca.explained_variance_ratio_ 

#end-pca
classifiers = [
    SGDClassifier(),
    KNeighborsClassifier(5),
    SVC(C=.1, degree=1, kernel='poly', probability=True,gamma='scale'),
    NuSVC(nu=.1, degree=1, kernel='poly', probability=True,gamma='scale'),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    LogisticRegression(max_iter=1100,solver='lbfgs')]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", 'Log Loss']
log = pd.DataFrame(columns=log_cols)
row_index=0
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__    
    print("_"*30)
    print(name)
    try:
        print('****Results****')
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))
#         print("accuracy_score: \n",accuracy_score(y_test, train_predictions))
        #print("precision_score: \n",precision_score(y_test, train_predictions))
        print("f1_score: \n",f1_score(y_test, train_predictions))
#         print("classification_report: \n",classification_report(y_test, train_predictions))
        print("confusion_matrix: \n",confusion_matrix(y_test, train_predictions))
       # err=(fp+fn)/(tp+fn+fp+tn)
        print("log_loss: \n",log_loss(y_test, train_predictions))
        log_entry = pd.DataFrame([[name, acc*100, log_loss(y_test, train_predictions)]], columns=log_cols)
        log = log.append(log_entry)
    except Exception as e:
        print (e)
    
print("_"*30)

plt.subplots(figsize=(8,4))
sns.barplot(x="Classifier", y="Accuracy",data=log,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('CKD Accuracy Comparison before applying pca')
plt.show()

log_cols=["Classifier", "Accuracy", 'Log Loss']
log1 = pd.DataFrame(columns=log_cols)
row_index=0
for clf in classifiers:
    clf.fit(X1_train, y1_train)
    name = clf.__class__.__name__
    
    print("_"*30)
    print(name)
    try:
        print('****Results****')
        train_predictions = clf.predict(X1_test)
        acc = accuracy_score(y1_test, train_predictions)
        print("Accuracy: {:.4%}".format(acc))
#         print("accuracy_score: \n",accuracy_score(y_test, train_predictions))
        #print("precision_score: \n",precision_score(y1_test, train_predictions))
        print("f1_score: \n",f1_score(y1_test, train_predictions))
#         print("classification_report: \n",classification_report(y_test, train_predictions))
        cm=confusion_matrix(y1_test, train_predictions)
        tp=cm[0][0]
        fn=cm[0][1]
        fp=cm[1][0]
        tn=cm[1][1]
        print("confusion_matrix: \n",cm)
        err=(fp+fn)/(tp+fn+fp+tn)
        print("Basic Evaluation Measures")
        print("Error Rate:",err)
        print("Accuracy:",1-err)
        print("Sensitivity (Recall or True positive rate):",tp/(tp+fn))
        print("Specificity (True negative rate):",tn/(tn+fp))
        print("Precision (Positive predictive value)",tp/tp+fp)
        print("False positive rate:",fp/tn+fp)
        print("log_loss: \n",log_loss(y1_test, train_predictions))
        log_entry = pd.DataFrame([[name, acc*100, log_loss(y1_test, train_predictions)]], columns=log_cols)
        log1 = log1.append(log_entry)
    except Exception as e:
        print (e)
    
print("_"*30)

plt.subplots(figsize=(8,4))
sns.barplot(x="Classifier", y="Accuracy",data=log1,palette='hot',edgecolor=sns.color_palette('dark',7))
plt.xticks(rotation=90)
plt.title('CKD Accuracy Comparison after applying Correlation and pca')
plt.show()
df.hist()
plt.show()
df.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
plt.show()
 