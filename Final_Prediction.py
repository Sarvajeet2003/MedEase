# %% [markdown]
# ### Project 2 - Predicting Hospital Readmissions
# 
# * Problem: Predict hospital readmissions within 30 days.
# * Objective: Build a predictive model for high-risk patients.
# * Tasks:
#     - Data Preprocessing
#     - Feature Engineering
#     - Model Building
#     - Model Evaluation

# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# %%
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## Read the Dataframe

# %%
df1 = pd.read_csv("hospital_with_actual_A1C.csv")
df2 = pd.read_csv("hospital_with_predicted_A1C.csv")

# %%
# Concatenate the two DataFrames along the rows (axis=0)
final_df = pd.concat([df1, df2], axis=0)

# Reset the index of the concatenated DataFrame
final_df.reset_index(drop=True, inplace=True)

# Display the concatenated DataFrame
final_df.head()

# %%
final_df['A1C_Result'].value_counts()

# %%


# %%


# %%
#shape
final_df.shape

# %%
# info
final_df.info()

# %%
#null
final_df.isnull().sum()

# %%
#duplicates
final_df.duplicated().sum()

# %%
#unique
final_df.nunique()

# %%
for column in final_df.columns:
    unique_values = final_df[column].unique()
    print(f"'{column}':\n {unique_values}\n")

# %%
# Save the Dataframe
final_df.to_csv("hospital_readmissions_final.csv", index= False)

# %% [markdown]
# ## Handling Outliers

# %%
# Calculate quartiles and IQR
Q1 = final_df.quantile(0.25)
Q3 = final_df.quantile(0.75)
IQR = Q3 - Q1

# Calculate upper and lower bounds for outliers
upper_bound = Q3 + 1.5 * IQR
lower_bound = Q1 - 1.5 * IQR

# Identify outliers
outliers = final_df[(final_df < lower_bound) | (final_df > upper_bound)]

# Count outliers
num_outliers = outliers.count()

print("Number of outliers:")
print(num_outliers)

# %% [markdown]
# ## Handling Skwness

# %%
# Function for histogram 
def plot_histograms(df, cols):

    plt.figure(figsize=(8, 15))

    for i, col in enumerate(cols):
        plt.subplot(7,2, i+1)
        sns.histplot(df[col],kde= True, bins=30, color="salmon") 
        plt.title(col)
    plt.tight_layout()
    plt.show()

# %%
columns = final_df.columns
plot_histograms(final_df, columns)

# %%
final_df.skew()

# %% [markdown]
#     Skewness is a measure of lack of symmetry
#     Skewness value range from -1 to 1:
# 
# - If the skewness is between -0.5 and 0.5, the distribution is approximately symmetric.
# - If the skewness is less than -0.5, the distribution is negatively skewed (left-skewed).
# - If the skewness is greater than 0.5, the distribution is positively skewed (right-sk

# %%
# Checking for multicollinearity

# %%
# Import library for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

# %%
calc_vif(final_df)

# %% [markdown]
# Variance Inflation Factor
# 
# - VIF values below 5 indicate that multicollinearity is not a significant concern, and the predictor variables are likely not highly correlated with each other.
# - VIF values between 5 and 10 suggest moderate multicollinearity.
# - VIF values above 10 indicate potentially severe multicollinearity.
# 
# 

# %% [markdown]
# #  Model to Readmission

# %%
# import
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, roc_curve, confusion_matrix, classification_report

from imblearn.combine import SMOTETomek

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import pickle

# %%
final_df.columns

# %%
final_df["Readmitted"].value_counts()

# %%
# Data Splitting

x_new = final_df.drop(columns=["Readmitted"],axis=1) #independent variables.
y_new = final_df["Readmitted"] #dependent variable

# %%


# %%
# Logistic Regression

# splitting train & test 
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

model = LogisticRegression(solver='liblinear').fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "Logistic Regression",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)

# %%
# SVM Classification

# splitting train & test 
x_train, x_test, y_train, y_test = train_test_split(x_new, y_new, test_size= 0.2, random_state=40)

svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
model = svm.fit(x_train, y_train)

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

#checking the accuracy_score
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

metrics ={"Algorithm": "SVM",
           "Accuracy_Train": accuracy_train,
           "Accuracy_Test": accuracy_test}
print(metrics)

# %%
# Other classification algorithms

def accuracy_checking(x_data, y_data, algorithm):
    
    # splitting train & test
    x_train, x_test, y_train, y_test= train_test_split(x_data, y_data, test_size= 0.2, random_state=40)

    model = algorithm().fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    #checking the accuracy_score
    accuracy_train = accuracy_score(y_train, y_pred_train)
    accuracy_test = accuracy_score(y_test, y_pred_test)

    metrics = {"Algorithm": algorithm.__name__,
               "Accuracy_Train": accuracy_train,
               "Accuracy_Test": accuracy_test}
    return metrics

# %%
print(accuracy_checking(x_new,y_new,DecisionTreeClassifier))
print(accuracy_checking(x_new,y_new,RandomForestClassifier))
print(accuracy_checking(x_new,y_new,ExtraTreesClassifier))
print(accuracy_checking(x_new,y_new,AdaBoostClassifier))
print(accuracy_checking(x_new,y_new,GradientBoostingClassifier))
print(accuracy_checking(x_new,y_new,XGBClassifier))

# %%


# %% [markdown]
# ## Cross Validation

# %%
# StratifiedKFold Cross Validation
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Instantiate the classification model
A1C_Model = GradientBoostingClassifier()

# Instantiate Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Stratified K-Fold Cross-Validation and calculate accuracy for each fold
accuracy_scores = cross_val_score(model, x_new, y_new, scoring='accuracy', cv=skf)
mean_accuracy = np.mean(accuracy_scores)

# Print
print("Accuracy scores for each fold:", accuracy_scores)
print("Mean Accuracy:", mean_accuracy)


# %%


# %%
# Selected Model
x_train, x_test, y_train, y_test= train_test_split(x_new, y_new, test_size= 0.2, random_state= 50)

Readmission_Model = GradientBoostingClassifier().fit(x_train, y_train)
 
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# %%


# %% [markdown]
# ### Performance Metrics

# %%
# accuracy_score for train and test

accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Accuracy score for Train and Test")
print("----------------------------------")
print("Accuracy_Train: ",accuracy_train)
print("Accuracy_Test: ",accuracy_test)

# %%
# confution matrics 

print("Confution_matrix for Test")
print("--------------------------")
print(confusion_matrix(y_true = y_test, y_pred = y_pred_test))

# %%
# classification report typically includes metrics such as precision, recall, F1-score, and support

print("Classification_report for Test")
print("-------------------------------")
print(classification_report(y_true= y_test, y_pred= y_pred_test))

# %%
# Receiver Operating Characteristic (ROC) Curve

FP, TP, Threshold = roc_curve(y_true=y_test, y_score=y_pred_test)

print(FP)
print(TP)
print(Threshold)


# %%
# Area Under the Curve (AUC)

auc_curve = auc(x=FP, y=TP)
print("auc_curve: ", auc_curve)

# %%
# create a plot for ROC and AUC curve

roc_point= {"ROC Curve (area)":round(auc_curve, 2)}
plt.plot(FP,TP,label= roc_point)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel("False Positive")
plt.ylabel("True Positive")
plt.plot([0,1],[0,1],"k--")
plt.legend(loc= "lower right")
plt.show()

# %%
# Saving the Model unsing pickle
with open("Readmission_Model.pkl","wb") as m:
    pickle.dump(Readmission_Model, m)

# %%


# %%
final_df.head()

# %%


# %% [markdown]
# # END

# %%


# %%
# testing
user_data = np.array([[0,2,1,75,29,4,0,3,5,1]])
prediction = Readmission_Model.predict(user_data)
prediction[0]

# %%
x_new.columns

# %%
# Min & Max of each Column
min_values = x_new.min()
max_values = x_new.max()

# Concatenate min_values and max_values along the columns axis
min_max_df = pd.concat([min_values, max_values], axis=1)
min_max_df.columns = ['Minimum', 'Maximum']

print("Minimum and Maximum values of all columns:")
print(" ")
print(min_max_df)

# %%



