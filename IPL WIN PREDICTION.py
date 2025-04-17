#!/usr/bin/env python
# coding: utf-8

# In[10]:


# import numpy as np
import pandas as pd


# In[11]:


match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv')


# In[12]:


match.head()


# In[13]:


delivery.head()


# In[14]:


total_score=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[15]:


total_score=total_score[total_score['inning']==1]


# In[16]:


match_df=match.merge(total_score[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[17]:


match_df


# In[18]:


match_df['team1'].unique()


# In[19]:


teams=['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bengaluru','Kolkata Knight Riders'
      ,'Punjab Kings','Chennai Super Kings','Rajasthan Royals','Delhi Capitals']


# In[20]:


match_df['team1']=match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2']=match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['toss_winner']=match_df['toss_winner'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['winner']=match_df['winner'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1']=match_df['team1'].str.replace('Deccan Charges','Sunrisers Hyderabad')
match_df['team2']=match_df['team2'].str.replace('Deccan Charges','Sunrisers Hyderabad')
match_df['toss_winner']=match_df['toss_winner'].str.replace('Deccan Charges','Sunrisers Hyderabad')
match_df['winner']=match_df['winner'].str.replace('Deccan Charges','Sunrisers Hyderabad')

match_df['team1']=match_df['team1'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
match_df['team2']=match_df['team2'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
match_df['toss_winner']=match_df['toss_winner'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
match_df['winner']=match_df['winner'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')

match_df['team1']=match_df['team1'].str.replace('Kings XI Punjab','Punjab Kings')
match_df['team2']=match_df['team2'].str.replace('Kings XI Punjab','Punjab Kings')
match_df['toss_winner']=match_df['toss_winner'].str.replace('Kings XI Punjab','Punjab Kings')
match_df['winner']=match_df['winner'].str.replace('Kings XI Punjab','Punjab Kings')


# In[21]:


delivery['batting_team']=delivery['batting_team'].str.replace('Delhi Daredevils','Delhi Capitals')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Delhi Daredevils','Delhi Capitals')

delivery['batting_team']=delivery['batting_team'].str.replace('Deccan Charges','Sunrisers Hyderabad')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Deccan Charges','Sunrisers Hyderabad')

delivery['batting_team']=delivery['batting_team'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')

delivery['batting_team']=delivery['batting_team'].str.replace('Kings XI Punjab','Punjab Kings')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Kings XI Punjab','Punjab Kings')


# In[22]:


match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]


# In[23]:


match_df.head(30)


# In[24]:


match_df.shape


# In[25]:


match_df=match_df[match_df['dl_applied']==0]


# In[26]:


match_df


# In[27]:


match_df.head(30)


# In[28]:


match_df=match_df[['match_id','city','winner','total_runs']]


# In[29]:


delivery_df=match_df.merge(delivery,on='match_id')


# In[30]:


delivery_df=delivery_df[delivery_df['inning']==2]


# In[31]:


delivery_df.shape


# In[32]:


delivery_df


# In[33]:


delivery_df['current_score']=delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[34]:


delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']


# In[35]:


delivery_df['balls_left']=126- (delivery_df['over']*6 + delivery_df['ball'])


# In[36]:


delivery_df


# In[37]:


delivery_df['player_dismissed']=delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed']=delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed']=delivery_df['player_dismissed'].astype('int')
wickets=delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets'] = 10-wickets
delivery_df.head()


# In[38]:


delivery_df.tail()


# In[39]:


#crr
delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])


# In[40]:


delivery_df['rrr']=(delivery_df['runs_left']*6/delivery_df['balls_left'])


# In[41]:


delivery_df


# In[42]:


def result(row):
    return 1 if row['batting_team']==row['winner'] else 0 


# In[43]:


delivery_df['result']=delivery_df.apply(result,axis=1)


# In[44]:


final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[45]:


final_df=final_df.sample(final_df.shape[0])


# In[46]:


final_df.sample()


# In[47]:


final_df.dropna(inplace=True)


# In[48]:


final_df=final_df[final_df['balls_left'] !=0]


# In[49]:


X=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[50]:


X_train


# In[51]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf=ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
],remainder='passthrough')


# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[53]:


pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[54]:


pipe.fit(X_train,y_train)


# In[55]:


X_train.describe()


# In[56]:


y_pred=pipe.predict(X_test)


# In[57]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[58]:


pipe.predict_proba(X_test)[5]


# In[59]:


delivery_df


# In[60]:


teams


# In[61]:


delivery_df['city'].unique()


# In[62]:


import pickle
pickle.dump(pipe,open('pipe3.pkl','wb'))


# In[63]:


pip install scikit-learn


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




