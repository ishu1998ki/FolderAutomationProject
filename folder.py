import os
import csv
import pandas as pd

rootFolderPath = r"C:\Users\Christila\Documents\PythonNewProject_Ashen\automate_model_evaluation-main"

# get list of .csv files from each folder

print("\nList of results.csv files from each folder with it's path\n\n")

# Get specific name.csv file
specificName = 'results.csv'

# store all the results.csv files into a list
resultList = []

for folder in os.listdir(rootFolderPath):

    # Join path and folder to check whether the folder is .csv
    folderPath = os.path.join(rootFolderPath, folder)
    if os.path.isdir(folderPath):

        for getResultsCsv in os.listdir(folderPath):
            if getResultsCsv in specificName:
                # store all the results.csv files into resultsList
                folderWithPath = os.path.join(folderPath,getResultsCsv)
                resultList.append(folderWithPath)

for result in resultList:
    print(result)

def split_data(resultList, index):
    neededPathList = []
    for result in resultList:
        splitPath = result.split(os.sep)
        neededPath = splitPath[index:]
        neededPathList.append(neededPath)

    return neededPathList


neededList = split_data(resultList,-2)
print(neededList)

dfNeededList = pd.DataFrame(neededList)
print(dfNeededList)

for file_path in resultList:
    df = pd.read_csv(file_path)

    for folderName in dfNeededList[0]:
        df.rename(columns={'train loss': dfNeededList+'_t'}, inplace=True)
        df.rename(columns={'val loss': dfNeededList + '_v'}, inplace=True)
