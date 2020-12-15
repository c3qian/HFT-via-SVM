#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


# In[2]:


def create_column_name(levels):
      columns = ['nil']
      number_of_levels = 30
      for i in range(1,4*number_of_levels+1):
        if i%4==1:
            name = "Ask_price"
        elif i%4==2:
            name = "Ask_size"
        elif i%4==3:
            name = "Bid_price"
        else:
            name = "Bid_size"
        key = name+"_"+str((i-1)/4+1)
        columns.append(key)
      return columns


# In[6]:


def reset_level(levels):
    temp = order[:][columns[1:4*levels+1]].copy()
    print( "orderbook size ", temp.shape)
    return temp


# In[ ]:


def generate_features(time,levels):
  
  #type_features_at_each_time 
  features = columns[:]
  ## from index 0-1-2-3;4-5-6-7, etc...
  
  names = ["spread","mid","n-i_diff_ask","n-i_diff_bid", "next_diff_ask","next_diff_bid"]
  
  new_order = pd.DataFrame()
  
  for i in range(levels):
    for j in range(4*i,4*i+4):
      new_order["Ask_price"+"_"+str(i+1)] = order["Ask_price"+"_"+str(i+1)] 
      new_order["Ask_size"+"_"+str(i+1)] = order["Ask_size"+"_"+str(i+1)] 
      new_order["Bid_price"+"_"+str(i+1)] = order["Bid_price"+"_"+str(i+1)] 
      new_order["Bid_size"+"_"+str(i+1)] = order["Bid_size"+"_"+str(i+1)] 
      for name in names:
        if name == "spread":
          new_order[name+str(i+1)] = order["Ask_price"+"_"+str(i+1)] - order["Bid_price"+"_"+str(i+1)] 
        elif name == "mid":
          new_order[name+str(i+1)] = (order["Ask_price"+"_"+str(i+1)] + order["Bid_price"+"_"+str(i+1)])*0.5
        elif name == "n-i_diff_ask":
          new_order[name+str(i+1)] = order["Ask_price"+"_"+str(levels)] - order["Ask_price"+"_"+str(i+1)]
        elif name == "n-i_diff_bid":
          new_order[name+str(i+1)] = order["Bid_price"+"_"+str(levels)] - order["Bid_price"+"_"+str(i+1)] 
        elif name == "next_diff_ask":
          new_order[name+str(i+1)] = order["Ask_price"+"_"+str(i+2)] - order["Ask_price"+"_"+str(i+1)]
        elif  name == "next_diff_bid":
          new_order[name+str(i+1)] = order["Bid_price"+"_"+str(i+2)] - order["Bid_price"+"_"+str(i+1)] 
          
  return new_order


def get_timeframed_data(n_data):
  X = pd.DataFrame()
  y = pd.DataFrame(columns = ['mid_price'])
  for i in range(n_data):
    cols = []
    for t in range(5):
      for col in new_order.columns:
        cols.append(str(t+1)+"__"+col)
    temp = pd.Series()
    temp_y = pd.Series()
    for t in range(5):
      for col in new_order.columns:
        temp[str(t+1)+"__"+col] = new_order[col][i+t]
    future = new_order['mid1'][i+5]
    now = new_order['mid1'][i]
    if future > now:
      temp_y['mid_price'] = "Up" 
    elif future == now:
      temp_y['mid_price'] = "Same"
    else:
      temp_y['mid_price'] = "Down"

    X = X.append(temp,ignore_index=True)
    y = y.append(temp_y,ignore_index=True )
  print "Class distributioin is : "
  print pd.value_counts(y['mid_price'].values)
  return X,y

def ML_train_test(X,y):
	le = LabelEncoder()
	le.fit(["Up","Same", "Down"])
	y["mid_price"] = le.transform(y["mid_price"])
	y_new = y['mid_price'].values
	from sklearn.model_selection import StratifiedKFold
	from sklearn.preprocessing import StandardScaler

	skf = StratifiedKFold(n_splits=7,random_state=42,shuffle=True)

	X_new = X.values

	scaler = StandardScaler()
	X_new =  scaler.fit_transform(X_new)
	skf.get_n_splits(X_new,y_new)

	for train_index, test_index in skf.split(X_new, y_new):
	   
	   X_train, X_test = X_new[train_index], X_new[test_index]
	   y_train, y_test = y_new[train_index], y_new[test_index]
	    
	   clf = SVC()
	   clf.fit(X_train, y_train)
	   
	   print clf.score(X_test, y_test)

if __name__ == "__main__":
  levels = 10
  time = 5
  n = 3000
  order = pd.read_csv('order.csv') 
  columns = create_column_name(levels)
  order.columns = columns[1:]

  order = reset_level(levels)
  # first was from index 1-2-3-4;etc
  columns = columns[1:4*levels+1]

  new_order = generate_features(time,levels-1).copy()

  X,y = get_timeframed_data(n)
  ML_train_test(X,y)

