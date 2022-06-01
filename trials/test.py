#Import Necessary Packages/Modules
from skXCS import XCS
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

#Load Data Using Pandas
data = pd.read_csv('./Data/trialData.csv') #REPLACE with your own dataset .csv filename
actionLabel = "phenotype"
dataFeatures = data.drop(actionLabel,axis=1).values #DEFINE actionLabel variable as the Str at the top of your dataset's action column
dataActions = data[actionLabel].values

#Shuffle Data Before CV
formatted = np.insert(dataFeatures,dataFeatures.shape[1],dataActions,1)
np.random.shuffle(formatted)
dataFeatures = np.delete(formatted,-1,axis=1)
dataActions = formatted[:,-1]

#Initialize XCS Model
model = XCS(learning_iterations = 5000)

#3-fold CV
print(np.mean(cross_val_score(model,dataFeatures,dataActions,cv=3)))