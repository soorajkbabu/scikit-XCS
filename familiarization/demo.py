from cmath import nan
import pandas as pd
from skXCS import StringEnumerator # to process string into numbers: might have to link this with IDs from graph nodes
from skXCS import XCS
import numpy as np

# XCS starting 
model = XCS()

#Read from CSV file
data = pd.read_csv("./Data/trialData_F.csv") #dummy data

#Specify the dataset's action label
actionLabel = "R"

#Derive the attribute and phenotype array using the action label
dataFeatures = data.drop(actionLabel,axis = 1).values
dataActions = data[actionLabel].values

#Optional: Retrieve the headers for each attribute as a length n array
dataHeaders = data.drop(actionLabel,axis=1).columns.values

print("Data Features")
print(dataFeatures)
print("\nData Actions")
print(dataActions)
print("\nData Headers")
print(dataHeaders)

#Initialize StringEnumerator object with csv filepath and class label.
converter = StringEnumerator("./Data/trialData_F.csv","R")

#Change Header Names to be more descriptive using change_header_name(currentName,newName)
# converter.change_header_name("A1","Component") 
# converter.change_header_name("A2","Domain")
# converter.change_header_name("A3","Theme")

#Change Phenotype Label to be more descriptive using change_class_name(newName)
# converter.change_class_name("Template")

"""

-add_attribute_converter_random(headerName):   Given an attribute name, randomly assigns each unique attribute value an
                                            integer value from 0 to n-1, where n = # of unique attribute values
                                            
-add_attribute_converter(headerName,array):   Given an attribute name, and an array of attribute values that will be
                                            converted, converter assigns each attribute value an integer value from
                                            0 to n-1 in the order of attribute values given in the array, where
                                            n = length of array. This can be useful for discrete attributes where the
                                            ordering of the attributes are important (ordinal values). For example,
                                            an attribute may have values "stage 1", "stage 2", "stage 3", "stage 4"
                                            to indicate stage of cancer, where the values are strings, but how they
                                            are enumerated is crucial.
"""
converter.add_attribute_converter("Active_Component",["Landing_Page","Editor","Behaviour_Editor","Supervision_Monitor"])
converter.add_attribute_converter("User_Goal",["select","navigate","create","manipulate","control"])
converter.add_attribute_converter_random("User_Theme")
converter.add_attribute_converter_random("Hovering_Object")
converter.add_attribute_converter_random("Clicked_Object")
converter.add_attribute_converter_random("Physiological_Data")
converter.add_attribute_converter_random("Gaze_Pattern")

# Similar to attribute converter, convert phenotype to numeric
converter.add_class_converter_random()

#Convert all attributes using convert_all_attributes()
converter.convert_all_attributes()

#Get arrays using get_params()
headers,classLabel,dataFeatures,dataPhenotypes = converter.get_params()

#converter = StringEnumerator("./Data/trialData.csv","phenotype")
#converter.print_invalid_attributes()
print("\nAFTER CONVERSION")
print("\nData Features")
print(dataFeatures)
print("\nData Phenotypes")
print(dataPhenotypes)
print("\nData Headers")
print(headers)
print("\nClass Label")
print(classLabel)

testInstance = np.array([[3,1,3,0,3,2,1,nan,nan]]) # Array has to be ndarray (mxn)
#testInstance = np.array([0,1,1,0,3,3,0,3,3])

"""
nu = Power parameter for fitness evaluation. recommended to be 1 for data with any level of noise. 
Increasing nu in clean problems improves chances of converging on optimal solution. Default of 5 is common for clean problems
"""

# Model training 
model = XCS(learning_iterations = 500) 
trainedModel = model.fit(dataFeatures,dataPhenotypes)

#print("Predictions:")
#print(trainedModel.predict(dataFeatures))

print("Single Manual Prediction Input:")
print(testInstance)

print("Single Manual Prediction Output")
print(trainedModel.predict(testInstance)) # Manual input 
#print(testInstance.astype(int))

print("Validation")
print(trainedModel.score(dataFeatures,dataPhenotypes))

trainedModel.export_final_rule_population("./Data/fileRulePopulation.csv",headers,classLabel)

