#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import numpy as np
import pandas as pd


# In[2]:


match=pd.read_csv('matches.csv')
delivery=pd.read_csv('deliveries.csv')


# In[3]:


match.head()


# In[4]:


delivery.head()


# In[5]:


total_score=delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[6]:


total_score=total_score[total_score['inning']==1]


# In[7]:


match_df=match.merge(total_score[['match_id','total_runs']],left_on='id',right_on='match_id')


# In[8]:


match_df


# In[9]:


match_df['team1'].unique()


# In[10]:


teams=['Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bengaluru','Kolkata Knight Riders'
      ,'Punjab Kings','Chennai Super Kings','Rajasthan Royals','Delhi Capitals']


# In[11]:


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


# In[12]:


delivery['batting_team']=delivery['batting_team'].str.replace('Delhi Daredevils','Delhi Capitals')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Delhi Daredevils','Delhi Capitals')

delivery['batting_team']=delivery['batting_team'].str.replace('Deccan Charges','Sunrisers Hyderabad')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Deccan Charges','Sunrisers Hyderabad')

delivery['batting_team']=delivery['batting_team'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Royal Challengers Bangalore','Royal Challengers Bengaluru')

delivery['batting_team']=delivery['batting_team'].str.replace('Kings XI Punjab','Punjab Kings')
delivery['bowling_team']=delivery['bowling_team'].str.replace('Kings XI Punjab','Punjab Kings')


# In[13]:


match_df=match_df[match_df['team1'].isin(teams)]
match_df=match_df[match_df['team2'].isin(teams)]


# In[14]:


match_df.head(30)


# In[15]:


match_df.shape


# In[16]:


match_df=match_df[match_df['dl_applied']==0]


# In[17]:


match_df


# In[18]:


match_df.head(30)


# In[19]:


match_df=match_df[['match_id','city','winner','total_runs']]


# In[20]:


delivery_df=match_df.merge(delivery,on='match_id')


# In[21]:


delivery_df=delivery_df[delivery_df['inning']==2]


# In[22]:


delivery_df.shape


# In[28]:


delivery_df


# In[31]:


delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce')


# In[33]:


delivery_df['current_score'] = delivery_df.groupby('match_id')['total_runs_y'].cumsum()


# In[ ]:





# In[34]:


delivery_df['runs_left']=delivery_df['total_runs_x']-delivery_df['current_score']


# In[35]:


delivery_df['balls_left']=126- (delivery_df['over']*6 + delivery_df['ball'])


# In[36]:


delivery_df


# In[38]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x: 0 if x == "0" else 1)
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype(int)

delivery_df['wickets'] = delivery_df.groupby('match_id')['player_dismissed'].cumsum()
delivery_df['wickets'] = 10 - delivery_df['wickets']

delivery_df.head()


# In[39]:


delivery_df.tail()


# In[40]:


#crr
delivery_df['crr']=(delivery_df['current_score']*6)/(120-delivery_df['balls_left'])


# In[41]:


delivery_df['rrr']=(delivery_df['runs_left']*6/delivery_df['balls_left'])


# In[42]:


delivery_df


# In[43]:


def result(row):
    return 1 if row['batting_team']==row['winner'] else 0 


# In[44]:


delivery_df['result']=delivery_df.apply(result,axis=1)


# In[45]:


final_df=delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[46]:


final_df=final_df.sample(final_df.shape[0])


# In[47]:


final_df.sample()


# In[48]:


final_df.dropna(inplace=True)


# In[49]:


final_df=final_df[final_df['balls_left'] !=0]


# In[50]:


X=final_df.iloc[:,:-1]
y=final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[51]:


X_train


# In[52]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
trf=ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
],remainder='passthrough')


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[54]:


pipe=Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[55]:


pipe.fit(X_train,y_train)


# In[56]:


X_train.describe()


# In[57]:


y_pred=pipe.predict(X_test)


# In[58]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[59]:


pipe.predict_proba(X_test)[5]


# In[60]:


delivery_df


# In[61]:


teams


# In[62]:


delivery_df['city'].unique()


# In[63]:


#import pickle
#pickle.dump(pipe,open('pipe3.pkl','wb'))
import joblib
joblib.dump(pipe, 'pipe3.pkl')


# In[64]:


pipe = joblib.load('pipe3.pkl')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




