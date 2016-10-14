
# coding: utf-8

# # Titanic

# In[1]:

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
get_ipython().magic(u'matplotlib inline')

df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# ## Data Exploration and Cleaning

# In[2]:

df.head()


# In[3]:

df.info()


# ### Missing values in column Age, Cabin and Embarked

# ### drop column PassengerId, Ticket, Cabin , for they seems not very useful

# In[4]:

df.drop(['PassengerId','Ticket','Cabin'], inplace=True, axis=1)


# In[5]:

df.describe()


# In[6]:

df.hist(figsize=(12,9
                ))


# In[7]:

df.plot(kind='box', return_type='dict' , subplots = True, figsize=(15,5))


# ### Outliers exist in Age, SibSp, Fare, how do we want to treat them?
""" clustering using K-Means
from sklearn.cluster import KMeans
ages = df['Age'].dropna()
ages = ages.reshape(714,1)
kmeans = KMeans(n_clusters=4, random_state=0).fit(ages)
kmeans.cluster_centers_
"""
# ### fill the missing values in Age with median values. do we have a better way?

# In[8]:

median_age = df.Age.median()
df['Age'].fillna(median_age, inplace = True)


# ### treat the missing values in Embarked

# In[9]:

df[df.Embarked.isnull()]


# In[10]:

# check Embarked values
df.Embarked.value_counts()


# In[11]:

df.pivot_table(values='Fare',
               index=['Embarked'],
               aggfunc=np.mean).plot(kind='bar')


# In[12]:

# it seems that based on Fare-Embarked relationship, the missing values are likely C.
df.Embarked.fillna('C', inplace=True)


# In[13]:

df.Embarked = df.Embarked.astype(str).map({"S":0,"C":1,"Q":2})


# In[14]:

df.Pclass.value_counts()


# In[15]:

df.pivot_table(values='Survived',
               index=['Pclass'],
               aggfunc=np.mean).plot(kind='bar')


# Pclass 1 and 2 have much better chance of surviving than Pclass 3.
# However, this would have been factored in tree based models.

# In[16]:

df.pivot_table(values='Survived',
               index=['Embarked'],
               aggfunc=np.mean).plot(kind='bar')


# ## Feature Construction

# In[17]:

df['FamilySize'] = df['SibSp'] + df['Parch']


# In[18]:

df.pivot_table(values='Survived',
               index=['FamilySize'],
               aggfunc=np.mean).plot(kind='bar')


# In[19]:

df.head()


# In[20]:

adult_age = 16
df['Person'] = df[['Age','Sex']].apply(lambda x: 'child' if x[0]<adult_age else x[1], axis=1)
df.Person = df.Person.astype(str).map({'male':1,'female':0,'child':2})


# In[21]:

df.pivot_table(values='Survived',
               index=['Person'],
               aggfunc=np.mean).plot(kind='bar')


# In[22]:

df.Person.value_counts()


# In[23]:

# according to the women and child first policy,
# if a wife who has children died, it is likely the whole family died
# and if a husband who has children survived, it is likely the whole family survived
# therefore, we use surname to check if the person is with perished_mother_wife or survive_father_husband
# to find out if he/she is survived.

# it would ideal that ML would discover this on its own, but at present it seems not.
# New features like this are useful to narrow its search

df['surname'] = df['Name'].apply(lambda name: name.split(',')[0].lower())

perishing_female_surnames = df[(df.Person==0) & (df.Survived ==0) & (df.FamilySize > 0)]['surname'].unique()
print('Total female adult victims with family:', len(perishing_female_surnames))


# In[24]:

df['perish_mother_wife'] = df['surname'].apply(lambda x: 1 if x in perishing_female_surnames else 0)


# In[25]:

survive_male_surnames = df[(df.Person==1) & (df.Survived ==1) & (df.FamilySize > 0)]['surname'].unique()


# In[26]:

df['survive_father_husband'] = df['surname'].apply(lambda x: 1 if x in survive_male_surnames else 0)


# In[27]:

# finish up and predict

df.drop(['SibSp','Parch','Name','surname','Sex','Age'],axis=1, inplace=True)


# In[28]:

predictor_var = list(df.columns[1:])
print predictor_var
outcome_var = 'Survived'


# In[29]:

categorical = ['Pclass','Person','Embarked','perish_mother_wife','survive_father_husband']
non_categorical = [ i for i in predictor_var if i not in categorical ]


# In[30]:

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_cat = ohe.fit_transform(df[categorical])


# In[31]:

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_non_cat = sc.fit_transform(df[non_categorical])


# In[32]:

from numpy import hstack
X = hstack((X_non_cat, X_cat))
y = df[outcome_var]
X.shape, X_non_cat.shape, X_cat.shape


# In[33]:

# dimension reduction then plot
#from sklearn import decomposition

#pca = decomposition.PCA(n_components=2)
#X_prime = pca.fit_transform(X)


# In[34]:

from sklearn.model_selection import learning_curve
# http://scikit-learn.org/stable/modules/generated\
#/sklearn.model_selection.learning_curve.html#sklearn.model_selection.learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# In[35]:

from sklearn.metrics import confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          precision = 3):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], precision)
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_cm(model, X, y, normalize = False, precision = 3):
    y_test = y
    y_pred = model.fit(X,y).predict(X)
    class_names = []

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=precision)


    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=normalize,
                          title='Confusion matrix', precision = precision)

    plt.show()    


# In[36]:

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()
scores = cross_val_score(lr_model, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} lr_model".format(scores.mean(), scores.std())
plot_cm(lr_model, X,y)


# In[37]:

from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_estimators = 100, max_depth=4)
scores = cross_val_score(et, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} rf_model".format(scores.mean(), scores.std())
plot_cm(et, X,y)


# In[38]:

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=300, max_features = 1, max_depth=6,random_state=42)
print rf_model.get_params()
scores = cross_val_score(rf_model, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} rf_model".format(scores.mean(), scores.std())
plot_cm(rf_model, X,y)


# In[39]:

plot_learning_curve(rf_model, "rf_model", X, y, ylim=(0.7, 1.01), n_jobs=4)


# In[40]:

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} knn".format(scores.mean(), scores.std())


# In[41]:

from sklearn import svm
svm_model = svm.SVC(kernel='rbf', random_state=0, gamma=0.10, C=1)
scores = cross_val_score(svm_model, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} svm_model".format(scores.mean(), scores.std())
plot_cm(svm_model, X,y)


# In[42]:

plot_learning_curve(svm_model, "svm_model", X, y, ylim=(0.7, 1.01), n_jobs=4)


# In[43]:

from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(adaboost, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.2%} +/-{:.2%} adaboost".format(scores.mean(), scores.std())


# In[44]:

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
scores = cross_val_score(gnb, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} GaussianNB".format(scores.mean(), scores.std())


# In[45]:

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

xgb = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 #reg_alpha = .01,
 #reg_lambda = .01,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=555)
scores = cross_val_score(xgb, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} xgb".format(scores.mean(), scores.std())


# In[46]:

from sklearn.model_selection import GridSearchCV

param_test1 = {
    "max_depth": (1,2,3),
    "min_child_weight": (1,2,3)
}

gsearch1 = GridSearchCV(estimator = xgb, 
 param_grid = param_test1, scoring='accuracy',n_jobs=4,cv=10, verbose=0)

gsearch1.fit(X, y)
#gsearch1.cv_results_, 
gsearch1.best_params_, gsearch1.best_score_


# In[47]:

max_depth, min_child_weight = gsearch1.best_params_.values()


# In[48]:

param_test = {
    "subsample": (.6,.75,.9),
    "colsample_bytree": (.6,.75,.9)
}

gsearch = GridSearchCV(estimator = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 #reg_alpha = .01,
 #reg_lambda = .01,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=555), 
 param_grid = param_test, scoring='accuracy',n_jobs=4,cv=10, verbose=0)

gsearch.fit(X, y)
gsearch.best_params_, gsearch.best_score_


# In[49]:

colsample_bytree = gsearch.best_params_['colsample_bytree']
subsample = gsearch.best_params_['subsample']


# In[50]:

param_test = {
    "gamma": (1e-5,.1, 0,10, 1e5),
}

gsearch = GridSearchCV(estimator = XGBClassifier(
 learning_rate =0.1,
 n_estimators=100,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=0,
 subsample=subsample,
 colsample_bytree=colsample_bytree,
 #reg_alpha = .01,
 #reg_lambda = .01,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=555), 
 param_grid = param_test, scoring='accuracy',n_jobs=4,cv=10, verbose=0)

gsearch.fit(X, y)
gsearch.best_params_, gsearch.best_score_


# In[51]:

xgb2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=1000,
 max_depth=max_depth,
 min_child_weight=min_child_weight,
 gamma=1e-5,
 subsample=subsample,
 colsample_bytree=colsample_bytree,
 #reg_alpha = .01,
 #reg_lambda = .01,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=555)
scores = cross_val_score(xgb2, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} xgb".format(scores.mean(), scores.std())


# In[52]:

plot_learning_curve(xgb2, "xgboost", X, y, ylim=(0.7, 1.01), n_jobs=4)


# In[53]:

from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('rf', rf_model), ('svm', svm_model),('xgb', xgb2)], voting='hard')

scores = cross_val_score(eclf, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} eclf".format(scores.mean(), scores.std())


# In[54]:

plot_learning_curve(eclf, "ensemble", X, y, ylim=(0.7, 1.01), n_jobs=4)


# In[55]:


"""
import tensorflow as tf
# Build 3 layer DNN with 1024, 512, 256 units respectively.
dnn = tf.contrib.learn.DNNClassifier(hidden_units=[1024,512,256],
n_classes=99)

scores = cross_val_score(dnn, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.3%} +/-{:.2%} eclf".format(scores.mean(), scores.std())

# Fit model.
dnn.fit(x=x_train, y=y_train, steps = 2000)

# Make prediction for test data
y = dnn.predict(x_test)
"""


# In[56]:

""" really slow
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(rf_model,
                            max_samples=0.5, max_features=0.5)
scores = cross_val_score(bagging, X, y, cv=10, scoring='accuracy')
print "Accuracy: {:.2%} +/-{:.2%} bagging".format(scores.mean(), scores.std())
"""


# In[57]:

# now work on test data
test_df.apply(lambda x: x.isnull().sum())


# In[58]:

test_df.Age.fillna(test_df.Age.median(), inplace=True)


# In[59]:

test_df[test_df.Fare.isnull()]


# In[60]:

test_df.Embarked = test_df.Embarked.astype(str).map({"S":0,"C":1,"Q":2})


# In[61]:

mean_fare = test_df[test_df.Embarked==0]['Fare'].mean() ; mean_fare


# In[62]:

test_df.pivot_table(values='Fare',
               index=['Embarked'],
               aggfunc=np.mean).plot(kind='bar')


# In[63]:

test_df.Fare.fillna(mean_fare, inplace = True)


# In[64]:

test_df['FamilySize'] = test_df.SibSp + test_df.Parch


# In[65]:

test_df['Person'] = test_df[['Age','Sex']].apply(lambda x: 'child' if x[0]<adult_age else x[1], axis=1)
# first convert to str, then map
test_df.Person = test_df.Person.astype(str).map({'male':1,'female':0,'child':2})


# In[66]:

# additional features

test_df['surname'] = test_df['Name'].apply(lambda name: name.split(',')[0].lower())

test_df['perish_mother_wife'] = test_df['surname'].apply(lambda x: 1 if x in perishing_female_surnames else 0)
test_df['survive_father_husband'] = test_df['surname'].apply(lambda x: 1 if x in survive_male_surnames else 0)


# In[67]:

test_df.drop(['Name','surname','Sex','Age','SibSp','Parch','Ticket','Cabin'], inplace=True, axis=1)
test_df[:3]


# In[68]:

# encoding, standarize
X_test_cat = ohe.transform(test_df[categorical])
X_test_non_cat = sc.transform(test_df[non_categorical])
X_test = hstack((X_test_non_cat, X_test_cat))


# In[69]:

model = rf_model
model.fit(X,y)
Y_pred = model.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('./submissions/rf10.csv', index=False)

# Test Results:
# model: score, description

# SVM1: .79426
# SVM2: .78469. use FamilySize instead of SibSp, Parch
# SVM3: .79426. Use both. no improvement
# SVM4: .79426, map sex, age to person
# xgb1, rf1 : not as good as svm1
# dt1,  .76077
# lr1: .76077, trees = 100, depth = 5
# rf2: .77512, n_estimator=300, maxdepth = 6
# rf3: .78469, n_estimator=300, maxdepth = 6, fixed a bug, use transform instead of fit_transform to generate X_test
# rf4: .79426, map sex, age to person
# rf5: .80861, use familysize instead of sibsp, parch
# xgb2: .78469, use the same config as rf5.
# ensemble1: .79426 (lr,rf,svm, hard)
# ensemble2: .79426 (rf, svm, hard)
# ensemble3: .78947 (rf, svm, adaboost, hard)

# used surnames as new features
#--------------------------------
# rf6: .81340, used surnames as new features
# SVM5: .80861, used surnames
# lr2: .80861, used surnames as new features
# xgb3: .77990, used surnames
# ensemble4: .80861, usd surnames

# changed adult age from 16 to 14
# rf6: .81340. no improvement
# ensemble5: .80861.

# add Pclass_1_2: if Pclass is 1 or 2
# rf7: .80861.
# removed changes to Pclass_1_2

# tuned xgb model
# xgb4: .80861

# ensemble6: .80861, (rf, svm and xgb)

# add Pclass1 and Pclass2
# rf8: .80861
# ensemble7: .80861
# xgb5: .80861

# references
# https://www.kaggle.com/vinceallenvince/titanic/titanic-guided-by-a-confusion-matrix
# https://www.kaggle.com/omarelgabry/titanic/a-journey-through-titanic

# add back the Pclass,
# rf9: .81340
# svm6: .80861
# xgb6: .80861. The cross validation score is promising. fix a bug here
# In[ ]:



