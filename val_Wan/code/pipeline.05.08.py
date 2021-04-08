# from google.colab import drive
# drive.mount('/content/gdrive')

# root_path = 'gdrive/My Drive/week.02'
root_path = '.'

import time
#from BaseLine
import os
import json
import numpy as np
import pandas as pd

import sklearn.svm as svm
import sklearn.tree as tree
import sklearn.ensemble as ensemble
import sklearn.neighbors as neighbors
import sklearn.naive_bayes as naive_bayes
import sklearn.linear_model as linear_model

from sklearn.impute import SimpleImputer
from sklearn import preprocessing as preproc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, mean_absolute_error

#from Wan
import shutil
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import random
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.exceptions import DataConversionWarning
import warnings
from traitlets.config.configurable import Configurable
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
# sol 3
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

#from Valerie
#sol 2
import xgboost as xgb
from xgboost import XGBClassifier
#sol 4
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor 
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_log_error,r2_score 
#sol 6
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from keras import Sequential
from keras.layers import Dense
#sol 8
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import chi2  
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
###########################################################
def ObjColConvertToDigit(train_df, numericColList):
   delRowList = []

   for col in numericColList:
      if(train_df[col].dtype == "object"):
         for i, cell in enumerate(train_df[col]):
            if(cell.replace('.', '').isdigit() == False):
               delRowList.append(i)

   train_df = train_df.drop(train_df.index[delRowList])

   return train_df
###############
def columnToList(columns):
  columns = columns.tolist()[0].replace("[", "").replace("]", "").split(";")
  tempList = []
  columnList = []
  for i, cell in enumerate(columns): 
    tempList.append(cell)
    if i % 3 == 2:
      columnList.append(tempList)
      tempList = newListClass.newList()

  numeric_columns = [x[1] for x in columnList if x[2]=='numeric' or x[2]=='real' or x[2]=='integer' or x[2] =='numerical' in x ]
  categorical_columns = [x[1] for x in columnList if x[2] == 'categorical' in x ]
  datetime_columns = [x[1] for x in columnList if x[2] == 'datetime' or  x[2] == 'dateTime' in x ]
  unwanted_columns = [x[1] for x in columnList if x[2] == 'unwanted' or x[2] == 'string' in x ]
  columNameTarget  = [x[1] for x in columnList if x[2] == 'target' in x ] 

  return columnList, columns, numeric_columns, categorical_columns , datetime_columns, unwanted_columns, columNameTarget
###############
def parseMetaData(rowFilePath):
   # print(rowFilePath)
   data = pd.read_csv(rowFilePath, header = 0, delimiter = ',')
   pd.set_option("display.max_colwidth", 10000)

   # print(rowFilePath + "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

   #check data
   # for i,  col in enumerate(data.columns.values):
   #   print(i+1, "  ", col, "-> ", data[col].to_string().replace('0    ', '')) 

   # #Assign strings to metadata
   tab = '0    '
   competition_name = data['name'].to_string().replace(tab,'')
   columnList, columns,  numeric_columns, categorical_columns, datetime_columns, unwanted_columns, columNameTarget = columnToList(data['columns'])
   metric = data['performanceMetric'].to_string().replace(tab,'')
   prepro_function_call = data['preprocessing function call'].to_string().replace(tab,'')
   feature_selector_type = data['featureSelector'].to_string().replace(tab,'')
   feature_selector = data['featureSelector function call'].to_string().replace(tab,'')
   estimator_type = data['estimator1'].to_string().replace(tab,'')
   estimator = data['estimator1 function call'].to_string().replace(tab,'')
   estimator2 = data['estimator2 function call'].to_string().replace(tab,'')
   target_column = data['targetName'].to_string().replace(tab,'')
   output_type = data['outputType'].to_string().replace(tab,'')
   output_type = output_type.split(',')

   if len(columNameTarget ) == 0:
      target_column = data['targetName'].to_string().replace(tab,'')
   else:
      target_column = columNameTarget[0]

   ##check if unwanted column has target_column
   removeStr = ""
   for i, ele in enumerate(unwanted_columns):
      ele = ele.replace(' ', '')
      if ele == target_column.replace(' ', ''):
         removeStr = ele
   if(removeStr != ""):
      unwanted_columns.remove(removeStr)

   my_dict = {'competition_name': competition_name, 'columns':columns, 'metric':metric,
              'prepro_function_call':prepro_function_call, 'feature_selector':feature_selector,
              'feature_selector_type':feature_selector_type, 'estimator_type':estimator_type,
               'estimator':estimator, 'estimator2':estimator2, 'target_column':target_column, 'output_type':output_type,
               'numeric_columns':numeric_columns, 'categorical_columns':categorical_columns,
               'datetime_columns':datetime_columns, 'unwanted_columns':unwanted_columns} 


   return my_dict  
####################
class newListClass():
  def newList():
    newList = []
    return newList
#####################
def dataAugmentation(X, y):
   tempX = pd.DataFrame()
   tempX = tempX.append(X, ignore_index = True)

   newYList = y.tolist()

   for i,  col in enumerate(X.columns.values):
      qNumeric = col in metaDic['numeric_columns'] 
      qCate = col in metaDic['categorical_columns']
      if qNumeric == True and qCate == False:
         for j in range(X[col].shape[0]):
            tempX.loc[j, col] = X[col][j]*(1+random.randint(-50, 50)*0.001)


   X = X.append(tempX, ignore_index=True)
   # X = X.concat([X, tempX], axis=0, ignore_index=True)
   y =  y.append(y, ignore_index=True)

   return X, y
########################
def auxillaryDataAugmentation(X, y):
   # # read Data
   # auxDataPath = "./" + "house-prices-advanced-regression-techniques" + "/data/train.csv"
   auxDataPath ="./" + metaDic['competition_name'] + "/data/auxTrain.csv"
   auxData = pd.read_csv(auxDataPath, parse_dates = [1])
   # print(auxData.head())

   #X column selection for auxillary data
   newColList = ["LotArea"]
   auxDataX = auxData.loc[:, newColList]
   auxDataX.rename(columns={'LotArea':'area_m'}, inplace=True)

   #y column selection for auxillary data
   auxYList = auxData.loc[:, ['SalePrice']]
   auxYList.rename(columns={'SalePrice': 'price_doc'}, inplace=True)

   #column selection for X
   selectedX = X.loc[:, ['area_m']]

   # combine original X with auxillary data
   combinedX = selectedX.append(auxDataX, ignore_index=True)
   # print(combinedX.shape)  
   
   # #join original Y with auxillary y
   joinedY = pd.concat([y, auxYList['price_doc']])

   return X, y
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def preprocess(train_df):
   # drop rows which has string in numeric column 
   train_df = ObjColConvertToDigit(train_df, metaDic['numeric_columns'])

   # drop unwanted columns
   if len(metaDic['unwanted_columns']) > 0:
      train_df.drop(metaDic['unwanted_columns'], axis=1, inplace=True)

   X = train_df.drop(metaDic['target_column'], 1)
   y = train_df[metaDic['target_column']]

   #auxillary Data augmentation
   if dataRootStr.find("russian-housing-") >= 0:
      X, y = auxillaryDataAugmentation(X, y)

   X, y = feature_extraction(X, y, train_df)

   if dataRootStr.find("restaurant-revenue-prediction") >= 0 or dataRootStr.find("bike") >= 0:
      X, y = dataAugmentation(X, y)

   X_train, X_test, y_train, y_test = train_test_split(X, y)

   return X_train, X_test, y_train, y_test

######################################################################################
def feature_extraction(X, y, train_df):  
   # treat missing values
   pd.set_option('mode.chained_assignment', None) # used to subside the panda's chain assignment warning
   imp = SimpleImputer(missing_values=np.nan, strategy='mean')
   X_numeric_colums = metaDic['numeric_columns']
   if metaDic['target_column'] in X_numeric_colums:
      X_numeric_colums.remove(metaDic['target_column'])  
   for col in X_numeric_colums:
      X [[col]]= imp.fit_transform(X[[col]])

   #handle categorical target 
   if metaDic['target_column'] in metaDic['categorical_columns']:
      col_dummies = pd.get_dummies(train_df[metaDic['target_column']], prefix=col)
      y = pd.concat([y, col_dummies], axis=1)
      y.drop(metaDic['target_column'], axis=1, inplace=True)  

   # Categorial transform  
   X_categorical_columns = metaDic['categorical_columns']
   if metaDic['target_column'] in X_categorical_columns:
      X_categorical_columns.remove(metaDic['target_column'])
   for col in X_categorical_columns:
      col_dummies = pd.get_dummies(X[col], dummy_na=True)
      X = pd.concat([X, col_dummies], axis=1)
      
   # drop float column in Categorical Column
   floatColInCat = "" 
   for col in X_categorical_columns:
      Xcol = X[col]
      if(Xcol.dtype == 'float64' ):
         floatColInCat = col

   if floatColInCat != "":
      X.drop(floatColInCat, axis=1, inplace=True)

   dropThresh = int(X.shape[0]*0.01)
   X.dropna(axis=1, how='any')

   le = preprocessing.LabelEncoder()
   X = X.apply(le.fit_transform)

   #Transform datetime
   X_datetime_columns = metaDic['datetime_columns']
   for col in X_datetime_columns:
      X["hour"] = [t.hour for t in pd.DatetimeIndex(X[col])] 
      X["day"] = [t.dayofweek for t in pd.DatetimeIndex(X[col])]
      X["month"] = [t.month for t in pd.DatetimeIndex(X[col])]
      X['year'] = [t.year for t in pd.DatetimeIndex(X[col])]
      X.drop(col, axis=1, inplace=True)

   # Feature normalization
   if len(metaDic['numeric_columns']) > 0:
      X[X_numeric_colums] = preproc.scale(X[X_numeric_colums])

   # print to screen
   print("************************************************************")
   print(metaDic['competition_name'])
   print(X_numeric_colums)
   # print(metaDic['string_columns'])
   print(metaDic['datetime_columns'])
   print(metaDic['categorical_columns'])
   print(metaDic['target_column'])
   print(metaDic['metric'])
   print(metaDic['feature_selector'])
   print(metaDic['estimator'])

   return X, y

#######################################################################################
def feature_selection(X_train, X_test, y_train, y_test):

   metaDic['feature_selector'] = metaDic['feature_selector'].replace(' ',',')
   function = metaDic['feature_selector']
   selector = metaDic['feature_selector_type']

   model = eval(function)
   model.fit(X_train, y_train)

   if selector == "selectkbest":  #CODE 8
      X_train = model.transform(X_train)
      X_test = model.transform(X_test)
   else:
      pass
    
   return X_train, X_test, y_train, y_test  
##############################################################################################
def estimation(X_train, X_test, y_train, y_test):
   estimator = metaDic['estimator'].replace(' ', ', ' )
   estimator2 = metaDic['estimator2'].replace(' ', ', ' )

   if(estimator == "NONE") :
      estimator = estimator2
   
   print("estimator====", estimator)
   model = eval(estimator)

   if (metaDic['estimator_type'] == 'nn'):
       model.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

   model.fit(X_train, y_train)
   predict = model.predict(X_test)

   error = "Nothing"
   print(metaDic['metric'])
   print(metaDic['estimator_type'])

   # print(y_test)
   if metaDic['metric'] == "rmse":
      error = np.sqrt(mean_squared_error(y_test, predict))
   elif metaDic['metric'] == 'mse':
      error = mean_squared_error(y_test, predict)
   elif metaDic['metric'] == "accuracy":
      if (metaDic['estimator_type'] == 'nn'):
         predict = (predict > 0.5) 
      elif (metaDic['estimator_type'] == 'RandomForestRegressor'):
         predict = (predict > 0.5) 
      error = accuracy_score(y_test, predict)
   elif metaDic['metric'] == "auc":
      fpr, tpr, _ = roc_curve(y_test, predict)
      error = auc(fpr, tpr)
   #elif metaDic['metric'] == 'cross_val_score':
      #error = cross_val_score(model, X_test, predict, cv=2)
   elif metaDic['metric'] == 'gini':
      error = Gini(y_test, predict)
   elif metaDic['metric'] == 'logloss':
      probs = model.predict_proba(X_test)
      error = log_loss(y_test, probs)
   elif metaDic['metric'] == 'rmsle':
      error = np.sqrt(np.mean((np.log(predict) - np.log(1+y_test))**2))
   else:
      pass

   print(metaDic['metric'] + " --------------> ", error)

####################################################################
def postprocessing(y_estimation, postProcessNum):
   pass
##########################################################
def Gini(y_true, y_pred):
   # print(y_true)
   # check and get number of samples
   assert y_true.shape == y_pred.shape
   n_samples = y_true.shape[0]

   # sort rows on prediction column 
   # (from largest to smallest)
   arr = np.array([y_true, y_pred]).transpose()
   true_order = arr[arr[:,0].argsort()][::-1,0]
   pred_order = arr[arr[:,1].argsort()][::-1,0]

   # print(type(true_order))
   true_order = true_order.astype(np.float)
   pred_order = pred_order.astype(np.float)

   # get Lorenz curves
   L_true = np.cumsum(true_order) / np.sum(true_order)
   L_pred = np.cumsum(pred_order) / np.sum(pred_order)
   L_ones = np.linspace(1/n_samples, 1, n_samples)

   # get Gini coefficients (area between curves)
   G_true = np.sum(L_ones - L_true)
   G_pred = np.sum(L_ones - L_pred)

   # normalize to true Gini coefficient
   return G_pred/G_true
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
startTime1 = time.time()

# read row.csv file
rowFilePaths =[]
for root, dirs, files in os.walk(".", topdown=False):
   for FileName in files:
      dirFileName = os.path.join(root, FileName)
      if(dirFileName.find("submission") > 0 and dirFileName.find("row") > 0):
         # print(dirFileName)
         rowFilePaths.append(dirFileName)

# rowFilePaths = [rowFilePaths[0]] #Russ
# print("======>",rowFilePaths)  

dataRootStr = ""
for rowFilePath in rowFilePaths:
   startTime = time.time()
   # print("======>",rowFilePath)
   metaDic = parseMetaData(rowFilePath)
   # print(metaDic['metric'])
   dataRootStr = "./" + metaDic['competition_name'] + "/data/"
   # print(dataRootStr)
   trainData = pd.read_csv(dataRootStr + "train.csv", parse_dates = [1])
   # print(trainData)

   X_train, X_test, y_train, y_test = preprocess(trainData)

   if metaDic['feature_selector'] != "NONE" :
      X_train, X_test, y_train, y_test = feature_selection(X_train, X_test, y_train, y_test)
   # X_train, X_test, y_train, y_test = feature_selection(X_train, X_test, y_train, y_test)

   estimation(X_train, X_test, y_train, y_test)
   
   endTime = time.time()
   print("execution duration->", endTime - startTime)

endTime1 = time.time()
print("Total execution duration->", endTime1 - startTime1)
  
