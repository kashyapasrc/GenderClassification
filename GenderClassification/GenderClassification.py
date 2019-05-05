from sklearn import tree
import pandas as pd
import numpy as np
col_names=['Favorite Color','Favorite Music Genre','Favorite Beverage','Favorite Soft Drink','Gender']
features=['Favorite Color','Favorite Music Genre','Favorite Beverage','Favorite Soft Drink']
target =['Gender']
train_df = pd.read_csv("./gender-classification/Transformed_Data_Set_Sheet1.csv",names=col_names)

print (train_df.head(5))


from sklearn.preprocessing import LabelEncoder

number=LabelEncoder()
train_df['Favorite Color']=number.fit_transform(train_df['Favorite Color'].astype('str'))
train_df['Favorite Music Genre']=number.fit_transform(train_df['Favorite Music Genre'].astype('str'))
train_df['Favorite Beverage']=number.fit_transform(train_df['Favorite Beverage'].astype('str'))
train_df['Favorite Soft Drink']=number.fit_transform(train_df['Favorite Soft Drink'].astype('str'))

x = train_df[features]
y= train_df.Gender


clf = tree.DecisionTreeClassifier()
clf.fit(x,y)
predicted=clf.predict([[0,6,2,1]])



print (predicted)


print (train_df.info())
