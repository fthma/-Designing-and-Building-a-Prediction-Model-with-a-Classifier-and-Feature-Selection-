import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler


df=pd.read_csv('VTargetBuyers.csv')
df.drop(columns=['CustomerKey','GeographyKey'],inplace=True)


yLabels=df[['BikeBuyer']]


df.drop(columns=['BikeBuyer'],inplace=True)

#df.head(10)

#df.select_dtypes(include='object')

#df.select_dtypes(include='int64')

dfTransform=df.copy()


LEncoder=LabelEncoder()
dfTransform['MaritalStatus']=LEncoder.fit_transform(df['MaritalStatus'])
dfTransform['Gender']=LEncoder.fit_transform(df['Gender'])

minmax=MinMaxScaler()

dfTransform['YearlyIncome']=df['YearlyIncome'].astype('float64',copy=False)
dfTransform['YearlyIncome']=pd.qcut(df['YearlyIncome'], q=5,labels=[0,1,2,3,4])
dfTransform['YearlyIncome']=minmax.fit_transform(dfTransform[['YearlyIncome']])

#totalchildern
dfTransform['TotalChildren']=df['TotalChildren'].astype('float64',copy=False)
dfTransform['TotalChildren']=minmax.fit_transform(dfTransform[['TotalChildren']])

#numberchildren at home
dfTransform['NumberChildrenAtHome']=df['NumberChildrenAtHome'].astype('float64',copy=False)
dfTransform['NumberChildrenAtHome']=minmax.fit_transform(dfTransform[['NumberChildrenAtHome']])

#number cars owned
dfTransform['NumberCarsOwned']=df['NumberCarsOwned'].astype('float64',copy=False)
dfTransform['NumberCarsOwned']=minmax.fit_transform(dfTransform[['NumberCarsOwned']])

dfTransform['CommuteDistance']=df['CommuteDistance'].replace(['0-1 Miles','1-2 Miles','2-5 Miles', '5-10 Miles', '10+ Miles'],[0,1,2,3,4])
dfTransform['CommuteDistance']=minmax.fit_transform(dfTransform[['CommuteDistance']])

#English education
dfTransform['EnglishEducation']=df['EnglishEducation'].replace(['Partial High School','High School','Partial College', 'Bachelors', 'Graduate Degree'],[0,1,2,3,4])
dfTransform['EnglishEducation']=minmax.fit_transform(dfTransform[['EnglishEducation']])

#English occupation
dfTransform['EnglishOccupation']=df['EnglishOccupation'].replace(['Manual','Skilled Manual','Clerical','Professional', 'Management'],[0,1,2,3,4])
dfTransform['EnglishOccupation']=minmax.fit_transform(dfTransform[['EnglishOccupation']])



OHE=pd.get_dummies(df['Region'],prefix='Region')
dfTransform=dfTransform.drop('Region',axis=1)
dfTransform=dfTransform.join(OHE)



dfTransform['Age']=pd.qcut(df['Age'], q=5,labels=[0,1,2,3,4])
dfTransform['Age']=minmax.fit_transform(dfTransform[['Age']])

#dfTransform.head(12)

#joining class labels to the transfomed df
dfTransform=dfTransform.join(yLabels)

dfTransform.to_csv('NormalizedData.csv',index=False)



