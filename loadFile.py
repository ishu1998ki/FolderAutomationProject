import os.path
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

# created a empty dataframe to get the output
empty_df = pd.DataFrame()

for file_path in resultList:
    df = pd.read_csv(file_path)
    # split file path to get the folder name for rename
    file_path = file_path.split('\\')

    # rename column using file name
    df.rename(columns={'train loss': file_path[-2]+'_t'}, inplace=True)
    df.rename(columns={'val loss': file_path[-2]+'_v'}, inplace=True)

    # getting only train loss and validation loss columns
    df = df.iloc[:,[2,4]]
    # concat the empty list with the inner dataframe to print the output
    empty_df = pd.concat([empty_df, df], axis=1)


print("\n\n******* Final DataFrame *******\n\n")
print(empty_df.head())


