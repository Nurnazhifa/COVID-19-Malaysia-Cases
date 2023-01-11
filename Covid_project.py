
#%%
#1. Import packages
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras import Sequential, Input
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# %%
#Step 1) Data Loading
CSV_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
df = pd.read_csv(CSV_PATH)

#%% 
#Step 2) Data Inspection / 
#df.head()
df.info() # cases_new is in object(Need to be converted into float)
#df.describe().T
#%%
df.isna().sum() # to check the number of NaNs

#%% 
# Step 3) Data cleaning 
#convert into floaT
df['cases_new']=pd.to_numeric(df['cases_new'],errors='coerce') 
#df.info()

#to replace NaNs using interpolation approach
df['cases_new']=df['cases_new'].interpolate(method='polynomial',order=2)

#%%
#to replace NaNs with 0
df = df.fillna(0)
#df['cluster_workplace']=df['cluster_workplace'].interpolate(method='polynomial',order=2)
#%%
#double confirm if training data still have NaNS
df.isna().sum()

#df.info() 

#%%
plt.figure(figsize=(10,10))
plt.plot(df['cases_new'])
plt.show()

#%% 
#Step 4) Features Selection
new_cases = df['cases_new'].values

#%% 
# Step 5) Data Preprocessing
from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
new_cases= mms.fit_transform(new_cases.reshape(-1,1))

# %%
X = [] # a list
y = [] # a list
win_size = 30

for i in range(win_size,len(new_cases)): 
    X.append(new_cases[i-win_size:i]) 
    y.append(new_cases[i]) 

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2,random_state=123,shuffle=True)

# %% Model Development

model = Sequential()
#model.add(LSTM(8,return_sequences=True,input_shape=(30,1)))
model.add(Input(shape=(X_train.shape[1:])))
#model.add(Dropout(0.3))
#model.add(LSTM(32,return_sequences=True))
model.add(LSTM(32))
model.add(Dropout(0.3))
#model.add(LSTM(32))
model.add(Dense(1,activation='relu'))
model.summary()

#%%

model.compile(optimizer='adam',loss='mse',metrics=['mape','mse'])

#%%
#Define the callbacks function to use
import os,datetime
es=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True,verbose=1)

log_path = os.path.join('log_dir','covid',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb=keras.callbacks.TensorBoard(log_path)

# %%
BATCH_SIZE = 32
EPOCHS = 10
hist = model.fit(X_train,y_train,epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[tb])

model.predict(X_train)

# %%
# Model Analysis

TEST_CSV_PATH = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')

# %%
#To load the test dataset
test_df = pd.read_csv(TEST_CSV_PATH)
#%%

print(test_df.isna().sum()) # 1 NaN

#to replace NaNs using interpolation approach
test_df['cases_new']=test_df['cases_new'].interpolate(method='polynomial',order=2)

#%%

print(test_df.isna().sum()) # 1 NaN
#%%
test_df.info()

#to concatenate the data
concat = pd.concat((df['cases_new'],test_df['cases_new']))
concat=concat[len(df['cases_new'])-win_size:]

#min max transformation
concat = mms.transform(concat[::,None])



# %%
X_testtest = []
y_testtest = []

for i in range(win_size, len(concat)):
    X_testtest.append(concat[i-win_size:i])
    y_testtest.append(concat[i])

X_testtest = np.array(X_testtest)
y_testtest = np.array(y_testtest)

#%%
predicted_cases = model.predict(X_testtest)


#%% inversing the normalize price

inversed_price = mms.inverse_transform(predicted_cases)
inversed_actual = mms.inverse_transform(y_testtest)

plt.figure()
plt.plot(inversed_price, color='red')
plt.plot(inversed_actual, color='blue')
plt.legend(['Predicted','Actual'])
plt.xlabel('Time')
plt.ylabel('Number of Cases')
plt.show()

#%% model evaluation using metrics

print(mean_squared_error(inversed_actual, inversed_price))
print(mean_absolute_percentage_error(inversed_actual,inversed_price))


#%%
model.save('model.h5')
import pickle
with open('mms.pkl','wb') as f:
    pickle.dump(mms,f)