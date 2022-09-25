#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from imblearn.over_sampling import RandomOverSampler


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
sc = StandardScaler()


# In[3]:


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.head()


# In[4]:


df['ever_married'].replace('Yes',1,inplace=True)
df['ever_married'].replace('No',0,inplace=True)
df['bmi'].fillna(df['bmi'].median(),inplace=True)
df['work_type'].replace(to_replace=['Private','Self-employed','Govt_job','children','Never_worked','Other'],value=[1,2,3,4,5,6],inplace=True)
df['smoking_status'].replace(to_replace=['formerly smoked', 'never smoked', 'smokes', 'Unknown'],value=[1,2,3,0],inplace=True)
df['Residence_type'].replace(to_replace=['Rural','Urban'],value=[0,1],inplace=True)
df['gender'].replace(to_replace=['Male','Female','Other'],value=[1,0,2],inplace=True)
x = df.iloc[:,1:11].values
y = df.iloc[:,11].values
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)
ros = RandomOverSampler(sampling_strategy="not majority")
X_res,y_res = ros.fit_resample(x,y) #for plotting the pie chart
X_res_train, y_res_train = ros.fit_resample(X_train,y_train)
X_res_test, y_res_test = ros.fit_resample(X_test,y_test)
unique_elements, counts_elements = np.unique(y_res, return_counts=True)
fig,ax = plt.subplots()
ax.pie(counts_elements,labels=unique_elements,autopct="%.2f")
ax.set_facecolor("white")
plt.title("Oversampling")
X_train = sc.fit_transform(X_res_train)
X_test = sc.transform(X_res_test)


# In[5]:


X_train = np.array(X_res_train)
y_train = np.array(y_res_train)
y_test = np.array(y_res_test)
X_test = np.array(X_res_test)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256,input_shape=(X_train.shape[1],),activation="relu"),
    tf.keras.layers.Dense(256,activation="relu"),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dense(128,activation="relu"),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(loss="binary_crossentropy",optimizer = 'Adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', 
                                   mode='max',
                                   patience=10,
                                   restore_best_weights=True)
r = model.fit(X_train,y_train,callbacks=[es],epochs=35,batch_size=10,shuffle=True,validation_data=(X_test,y_test))


# In[6]:


plt.plot(r.history['loss'],label='loss')
plt.show()


# In[7]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

preds = np.round(model.predict(X_test),0)


# In[8]:


preds


# In[9]:


print(confusion_matrix(y_test, preds))


# In[10]:


print(classification_report(y_test, preds))


# In[11]:


test_results = {}

test_results['model'] = model.evaluate(
    X_test, y_test, verbose=0)
accuracy = '{0:.2f}'.format(test_results['model'][1]*100)
print("Final Accuracy : ",accuracy,"%")


# In[12]:


inputs = []
# gender = input("Enter your gender : (Male,Female,Other)")
# if (gender=="Male" or gender=="male" or gender=="M"):
#     inputs.append(1)
# elif (gender=="Female" or gender=="female" or gender=="F"):
#     inputs.append(0)
# else:
#     inputs.append(2)
# age = float(input("Enter your age : "))
# inputs.append(age)

rest = [0,20,1,1,1,3,1,140,25.5,2]
inputs.extend(rest)
inputs = np.array(inputs)
inputs = inputs.reshape(-1,10)
inputs = sc.transform(inputs)
prediction = np.round(model.predict(inputs),0)
print(prediction[0])


# In[ ]:




