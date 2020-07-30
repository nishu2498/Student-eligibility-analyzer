
import pandas as pd
import sys

dataset = pd.read_csv(sys.argv[1])

df = pd.DataFrame(dataset)

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
df['college'] = label.fit_transform(df['College name'])
df['Areas'] = label.fit_transform(df['Areas of interest'])
df['Standard'] = label.fit_transform(df['Which-year are you studying in?'])
df['gender'] = label.fit_transform(df['Gender'])
df['major'] = label.fit_transform(df['Major/Area of Study'])
df['Creteria'] = label.fit_transform(df['Label'])

#Finding best binary classifier to classify data into “eligible/1” and “not eligible/0

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data=df[["CGPA/ percentage","Rate your written communication skills [1-10]","Rate your verbal communication skills [1-10]","Standard","gender","major"]]
target=df['Creteria']

X_train, X_test,y_train,y_test = train_test_split(data,target,random_state =42)

model=RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)
expected = y_test
predicted = model.predict(X_test)
print(f1_score(expected,predicted))
