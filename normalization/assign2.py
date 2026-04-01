import pandas as pd
import numpy as np
import csv

df = pd.read_csv("dataset.csv")
print("Original Data:")
print(df)

#1. Fill missing values with mean 
df['age'] = df['age'].fillna(df['age'].mean())
df['attendance'] = df['attendance'].fillna(df['attendance'].mean())
df['marks'] = df['marks'].fillna(df['marks'].mean())

print("\nAfter Filling Missing values:")
print(df)

#2. Remove duplicates Rows
df = df.drop_duplicates()

print("\nAfter Removing Duplicates:")
print(df)

#4. Detect and remove outliers using IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


#apply
df = remove_outliers_iqr(df,['age','attendance','marks'])
print("\nAfter Removing Outliers:")
print(df)
#3. Min-max Normalization
def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())
df['age'] = min_max_normalize(df['age'])
df['attendance'] = min_max_normalize(df['attendance'])
df['marks'] = min_max_normalize(df['marks'])

print("\nAfter normalization:")
print(df)



#finalResult:

print("\nFinal Cleaned Dataset:")
print(df)

# #make a file.csv
# with open("newdata.csv", "w", newline="") as file:
#     writer = csv.writer(file)

#     # Header
#     writer.writerow(df.columns)

#     # Rows
#     for i in range(len(df)):
#         writer.writerow([
#             df.loc[i, 'student_id'],
#             df.loc[i, 'age'],
#             df.loc[i, 'attendance'],
#             df.loc[i, 'marks']
#         ])

# print("CSV file saved")




