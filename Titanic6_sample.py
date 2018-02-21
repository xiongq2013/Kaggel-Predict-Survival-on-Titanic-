# https://www.kaggle.com/poonaml/titanic/titanic-survival-prediction-end-to-end-ml-pipeline
### Introduction

## Import libraries
# We can use the pandas library in python to read in the csv file.
import pandas as pd
#for numerical computaions we can use numpy library
import numpy as np

## Load train and test data
# This creates a pandas dataframe and assigns it to the titanic variable.
titanic = pd.read_csv("./data/train.csv")
# Print the first 5 rows of the dataframe.
titanic.head()

titanic_test = pd.read_csv("./data/test.csv")
#transpose
titanic_test.head().T
#note their is no Survived column here which is our target varible we are trying to predict

#shape command will give number of rows/samples/examples and number of columns/features/predictors in dataset
#(rows,columns)
titanic.shape

#Describe gives statistical information about numerical columns in the dataset
titanic.describe()
#info method provides information about dataset like
#total values in each column, null/not null, datatype, memory occupied etc
titanic.info()
#lets see if there are any more columns with missing values
titanic.isnull().sum()
'''Age, Embarked and cabin has missing values.'''
#how about test set??
titanic_test.isnull().sum()
'''Age, Fare and cabin has missing values.'''


## Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1)

pd.options.display.mpl_style = 'default'
titanic.hist(bins=10,figsize=(9,7),grid=False)     # grid: draw with background grid
'''we can see that Age and Fare are measured on very different scaling.
So we need to do feature scaling before predictions.'''

g = sns.FacetGrid(titanic, col="Sex", row="Survived", margin_titles=True)
g.map(plt.hist, "Age",color="purple")
#Subplot grid for plotting conditional relationships


g = sns.FacetGrid(titanic, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"seagreen", 0:"gray"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()


g = sns.FacetGrid(titanic, hue="Survived", col="Sex", margin_titles=True,
                palette="Set1",hue_kws=dict(marker=["^", "v"]))
# add para: hue_order=[1,0], hue_kws: dictionary of parameters
g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
plt.subplots_adjust(top=0.8)
g.fig.suptitle('Survival by Gender , Age and Fare')


titanic.Embarked.value_counts().plot(kind='bar', alpha=0.55)
plt.title("Passengers per boarding location")

sns.factorplot(x = 'Embarked',y="Survived", data = titanic,color="r")


sns.set(font_scale=1)
g = sns.factorplot(x="Sex", y="Survived", col="Pclass",
                    data=titanic, saturation=.5,
                    kind="bar", ci=None, aspect=.6)
(g.set_axis_labels("", "Survival Rate")
    .set_xticklabels(["Men", "Women"])
    .set_titles("{col_name} {col_var}")
    .set(ylim=(0, 1))
    .despine(left=True))
plt.subplots_adjust(top=0.8)
g.fig.suptitle('How many Men and Women Survived by Passenger Class')


ax = sns.boxplot(x="Survived", y="Age",
                data=titanic)
ax = sns.stripplot(x="Survived", y="Age",
                   data=titanic, jitter=True,
                   edgecolor="gray")
# Draw a scatterplot where one variable is categorical.
sns.plt.title("Survival by Age",fontsize=12)


titanic.Age[titanic.Pclass == 1].plot(kind='kde')     #kde: density plot
titanic.Age[titanic.Pclass == 2].plot(kind='kde')
titanic.Age[titanic.Pclass == 3].plot(kind='kde')
 # plots an axis lable
plt.xlabel("Age")
plt.title("Age Distribution within classes")
# sets our legend for our graph.
plt.legend(('1st Class', '2nd Class','3rd Class'),loc='best')


corr=titanic.corr()#["Survived"]
plt.figure(figsize=(10, 10))

sns.heatmap(corr, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between features')


#correlation of features with target variable
titanic.corr()["Survived"]
'''Looks like Pclass has got highest negative correlation with "Survived" followed by Fare, Parch and Age'''


g = sns.factorplot(x="Age", y="Embarked",
                    hue="Sex", row="Pclass",
                    data=titanic[titanic.Embarked.notnull()],
                    orient="h", size=2, aspect=3.5,
                   palette={'male':"purple", 'female':"blue"},
                    kind="violin", split=True, cut=0, bw=.2)


## Missing value imputation
'''But filling missing values with mean/median/mode is also a prediction which may not be 100% accurate,
instead you can use models like Decision Trees and Random Forest which handle missing values very well.'''

# Embarked Column
#Lets check which rows have null Embarked column
titanic[titanic['Embarked'].isnull()]
'''PassengerId 62 and 830 have missing embarked values
Both have Passenger class 1 and fare $80.
Lets plot a graph to visualize and try to guess from where they embarked'''

sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=titanic)
'''We can see that for 1st class median line is coming around fare $80 for embarked value 'C'.
So we can replace NA values in Embarked column with 'C'  '''
titanic["Embarked"] = titanic["Embarked"].fillna('C')

# Fare Column
#there is an empty fare column in test set
titanic_test[titanic_test['Fare'].isnull()]
#we can replace missing value in fare by taking median of all fares of those passengers
#who share 3rd Passenger class and Embarked from 'S'
def fill_missing_fare(df):
    median_fare=df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
#'S'
       #print(median_fare)
    df["Fare"] = df["Fare"].fillna(median_fare)
    return df

titanic_test=fill_missing_fare(titanic_test)


## Feature Engineering
'''Deck- Where exactly were passenger on the ship?'''
titanic["Deck"]=titanic.Cabin.str[0]
titanic_test["Deck"]=titanic_test.Cabin.str[0]
titanic["Deck"].unique() # 0 is for null values

g = sns.factorplot("Survived", col="Deck", col_wrap=4,
                    data=titanic[titanic.Deck.notnull()],
                    kind="count", size=2.5, aspect=.8)


titanic = titanic.assign(Deck=titanic.Deck.astype(object)).sort("Deck")
g = sns.FacetGrid(titanic, col="Pclass", sharex=False,
                  gridspec_kws={"width_ratios": [5, 3, 3]})
g.map(sns.boxplot, "Deck", "Age")


titanic.Deck.fillna('Z', inplace=True)
titanic_test.Deck.fillna('Z', inplace=True)
titanic["Deck"].unique() # Z is for null values


'''How Big is your family?'''
# Create a family size variable including the passenger themselves
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]+1
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]+1
print(titanic["FamilySize"].value_counts())

# Discretize family size
titanic.loc[titanic["FamilySize"] == 1, "FsizeD"] = 'singleton'
titanic.loc[(titanic["FamilySize"] > 1)  &  (titanic["FamilySize"] < 5) , "FsizeD"] = 'small'
titanic.loc[titanic["FamilySize"] >4, "FsizeD"] = 'large'

titanic_test.loc[titanic_test["FamilySize"] == 1, "FsizeD"] = 'singleton'
titanic_test.loc[(titanic_test["FamilySize"] >1) & (titanic_test["FamilySize"] <5) , "FsizeD"] = 'small'
titanic_test.loc[titanic_test["FamilySize"] >4, "FsizeD"] = 'large'
print(titanic["FsizeD"].unique())
print(titanic["FsizeD"].value_counts())

sns.factorplot(x="FsizeD", y="Survived", data=titanic)


# Convert Categorical variables into Numerical ones
from sklearn.preprocessing import LabelEncoder

labelEnc=LabelEncoder()

cat_vars=['Embarked','Sex',"FsizeD",'Deck']
for col in cat_vars:
    titanic[col]=labelEnc.fit_transform(titanic[col])
    titanic_test[col]=labelEnc.fit_transform(titanic_test[col])

titanic.head()


# Age Column
'''Age seems to be promising feature. So it doesnt make sense to simply
fill null values out with median/mean/mode.
We will use Random Forest algorithm to predict ages.'''

sns.set_style("whitegrid")
sns.distplot(titanic["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="red")
sns.plt.title("Age Distribution")
plt.ylabel("Count")


from sklearn.ensemble import RandomForestRegressor
# predicting missing values in age using Random Forest
def fill_missing_age(df):
    # Feature set
    age_df = df[['Age', 'Embarked', 'Fare', 'Parch', 'SibSp',
                'Pclass', 'FamilySize', 'FsizeD','Deck']]
    # Split sets into train and test
    train = age_df.loc[(df.Age.notnull())]  # known Age values
    test = age_df.loc[(df.Age.isnull())]  # null Ages

    # All age values are stored in a target array
    y = train.values[:, 0]

    # All the other values are stored in the feature array
    X = train.values[:, 1::]

    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)

    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])

    # Assign those predictions to the full data set
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df


titanic = fill_missing_age(titanic)
titanic_test = fill_missing_age(titanic_test)


sns.set_style("whitegrid")
sns.distplot(titanic["Age"].dropna(),
                 bins=80,
                 kde=False,
                 color="tomato")
sns.plt.title("Age Distribution")
plt.ylabel("Count")
plt.xlim((15, 100))

# Feature Scaling
# We can see that Age, Fare are measured on different scales,
# so we need to do Feature Scaling first before we proceed with predictions.
from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(titanic[['Age', 'Fare']])
df_std = std_scale.transform(titanic[['Age', 'Fare']])


std_scale = preprocessing.StandardScaler().fit(titanic_test[['Age', 'Fare']])
df_std = std_scale.transform(titanic_test[['Age', 'Fare']])

# Correlation of features with target
titanic.corr()["Survived"]



## Predict Survival

# LDA & QDA
# Import the linear regression class
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit


predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]
# Initialize our algorithm
lda = LinearDiscriminantAnalysis()
# Compute the accuracy score for all the cross validation folds.
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(lda, titanic[predictors],
                                          titanic["Survived"],scoring='f1',cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''.739409534928'''

# Initialize our algorithm
qda = QuadraticDiscriminantAnalysis()
# Compute the accuracy score for all the cross validation folds.
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(qda, titanic[predictors],
                                          titanic["Survived"],scoring='f1',cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''0.747560975496'''



# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]
# Initialize our algorithm
lr = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(lr, titanic[predictors],
                                          titanic["Survived"],scoring='f1',cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''0.737145659802'''


# Random Forest
from sklearn.ensemble import RandomForestClassifier

predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]

# Initialize our algorithm with the default paramters
# n_estimators is the number of trees we want to make
# min_samples_split is the minimum number of rows we need to make a split
# min_samples_leaf is the minimum number of samples we can have at the place where a tree branch ends (the bottom points of the tree)
rf = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=2,
                            min_samples_leaf=1)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(rf, titanic[predictors], titanic["Survived"],
                                          scoring='f1', cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''0.750665468523'''


# SVM
from sklearn.svm  import SVC
predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]

svm=SVC(random_state=1,kernel="poly",degree=2)
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(svm, titanic[predictors], titanic["Survived"],
                                          scoring='f1', cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''linear: 0.725850043314'''
'''poly(2): '''


# kNN
from sklearn.neighbors import KNeighborsClassifier
predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]

for i in range(8):
    knn=KNeighborsClassifier(n_neighbors=i+1)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
    scores = cross_val_score(knn, titanic[predictors], titanic["Survived"],
                                          scoring='f1', cv=cv)
    print(i, scores.mean())


# Gaussian Native Bayes
from sklearn.naive_bayes import GaussianNB
predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]

gnb=GaussianNB()
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(gnb, titanic[predictors], titanic["Survived"],
                                          scoring='f1', cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''0.677362940884'''


# Neural Network
from sklearn.neural_network import MLPClassifier

predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]

NN=MLPClassifier(random_state=1,hidden_layer_sizes=(100,),activation="tanh")
cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)
scores = cross_val_score(NN, titanic[predictors], titanic["Survived"],
                                          scoring='f1', cv=cv)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())
'''0.692266046563'''


# Predict use the RAndomforest method
predictors = ["Pclass", "Sex", "Age","SibSp", "Parch", "Fare",
              "Embarked", "FsizeD", "Deck"]
arg=LinearDiscriminantAnalysis()
arg_fit=arg.fit(titanic[predictors], titanic["Survived"])
arg_pred=arg.predict(titanic_test[predictors])

submission=pd.DataFrame({
    "PassengerId": titanic_test["PassengerId"],
    "Survived": arg_pred
})
submission.to_csv("Titanic_submission2_lda.csv",index=False)


