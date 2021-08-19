#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import math
import datetime as dt
import time
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import KNNImputer


# In[5]:


def duration_calc(dataframe):
    
    diff_min_list = []
    for i in range(len(dataframe)):

        start = dt.datetime.strptime(dataframe['start_time'].iloc[i], '%H:%M:%S')
        end = dt.datetime.strptime(dataframe['end_time'].iloc[i], '%H:%M:%S')
        diff = (end - start) 
        diff_min = diff.seconds/60
        diff_min_list.append(diff_min)
        
    return diff_min_list


# In[6]:


def data_imputer(dataframe, column_name):

    features = dataframe[column_name]
    features = np.array(features)
    features = features.reshape(-1, 1)
    imputer = KNNImputer(n_neighbors = 2, weights='distance')
    imputed_data = imputer.fit_transform(features)
    
    return imputed_data


# In[7]:


def final_check(dataframe):
    
    if len(dataframe.columns[dataframe.isna().any()].tolist())==0:
        #print('No Nan...')
        return dataframe
    else:
        #print('Replacing Nan...')
        dataframe = dataframe.replace(np.nan, 0)
        return dataframe


# # App usage data:

# In[8]:
def preprocessing(uid):

    #df_app_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/app_usage/running_app_'+uid+'.csv')


    # In[9]:


    #df_app_clean = df_app_raw.drop(['id', 'device', 'RUNNING_TASKS_baseActivity_mClass',
                               #'RUNNING_TASKS_baseActivity_mPackage',
                               #'RUNNING_TASKS_topActivity_mClass',
                               #'RUNNING_TASKS_topActivity_mPackage'], axis=1)


    # In[10]:


    #df_app_clean['date'] = df_app_clean['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))


    # In[11]:


    #df_app = df_app_clean[['date', 'RUNNING_TASKS_numActivities', 'RUNNING_TASKS_numRunning']]

    #if len(df_app.columns[df_app.isna().any()].tolist())!=0:
     #   for name in df_app.columns[df_app.isna().any()].tolist():
      #      df_app[name] = data_imputer(df_app, name).astype(int)
    #else:
     #   pass


    # In[12]:


    #df_app_sum = df_app.groupby('date').sum()
    #df_app_sum.columns = ['Sum_numActivities', 'Sum_numRunning']

    #df_app_median = df_app.groupby('date').median()
    #df_app_median.columns = ['Median_numActivities', 'Median_numRunning']


    # In[13]:


    #data_app = [df_app_sum['Sum_numActivities'], df_app_sum['Sum_numRunning'],
     #      df_app_median['Median_numActivities'], df_app_median['Median_numRunning']]

    #headers_app = ['Sum_numActivities', 'Sum_numRunning',
     #         'Median_numActivities', 'Median_numRunning']

    #df_app_data = pd.concat(data_app, axis=1, keys=headers_app)
    #df_app_data = final_check(df_app_data)
    #df_app_data


    

    # # Sensing data:

    # In[19]:


    ## Timezone is the eastern Time zone


    # ## Activity

    # In[20]:


    ## 0 -> Stationary
    ## 1 -> Walking
    ## 2 -> Running
    ## 3 -> Unknown


    # In[19]:


    df_activity_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/activity/activity_'+uid+'.csv')
    #df_activity_raw


    # In[20]:


    df_activity_raw['date'] = df_activity_raw['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))


    # In[21]:


    df_activity = df_activity_raw[['date', ' activity inference']]

    if len(df_activity.columns[df_activity.isna().any()].tolist())!=0:
        for name in df_activity.columns[df_activity.isna().any()].tolist():
            df_activity[name] = data_imputer(df_activity, name).astype(int)
    else:
        pass


    # In[26]:


    df_activity_mean = df_activity.groupby('date').mean()
    df_activity_mean.columns = ['Mean_Activity_inference']

    df_activity_std = df_activity.groupby('date').std(ddof=0)
    df_activity_std.columns = ['Std_Activity_inference']

    df_activity_median = df_activity.groupby('date').median()
    df_activity_median.columns = ['Median_Activity_inference']

    df_activity_min = df_activity.groupby('date').min()
    df_activity_min.columns = ['Min_Activity_inference']

    df_activity_max = df_activity.groupby('date').max()
    df_activity_max.columns = ['Max_Activity_inference']

    df_activity_skew = df_activity.groupby('date').skew()
    df_activity_skew.columns = ['Skew_Activity_inference']

    df_activity_var = df_activity.groupby('date').var(ddof=0)
    df_activity_var.columns = ['Var_Activity_inference']

    df_activity_sum = df_activity.groupby('date').sum()
    df_activity_sum.columns = ['Sum_Activity_inference']


    # In[27]:


    data_act = [df_activity_mean["Mean_Activity_inference"], df_activity_std["Std_Activity_inference"],
            df_activity_median['Median_Activity_inference'], df_activity_min['Min_Activity_inference'], 
            df_activity_max['Max_Activity_inference'], df_activity_skew['Skew_Activity_inference'], 
           df_activity_var['Var_Activity_inference'], df_activity_sum['Sum_Activity_inference']]

    headers_act = ['Mean_Activity_inference', 'Std_Activity_inference','Median_Activity_inference',
               'Min_Activity_inference', 'Max_Activity_inference', 'Skew_Activity_inference',
              'Var_Activity_inference', 'Sum_Activity_inference']

    df_activity_data = pd.concat(data_act, axis=1, keys=headers_act)
    df_activity_data = final_check(df_activity_data)
    #df_activity_data


    # 
    # ## Audio

    # In[26]:


    ## 0 -> Silence
    ## 1 -> Voice
    ## 2 -> Noise
    ## 3 -> Unknown


    # In[28]:


    df_audio_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/audio/audio_'+uid+'.csv')
    #df_audio_raw


    # In[29]:


    df_audio_raw['date'] = df_audio_raw['timestamp'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    #df_audio_raw['time'] = df_audio_raw['timestamp'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))


    # In[30]:


    df_audio = df_audio_raw[['date', ' audio inference']]

    if len(df_audio.columns[df_audio.isna().any()].tolist())!=0:
        for name in df_audio.columns[df_audio.isna().any()].tolist():
            df_audio[name] = data_imputer(df_audio, name).astype(int)
    else:
        pass


    # In[31]:


    df_audio_mean = df_audio.groupby('date').mean().astype(int)
    df_audio_mean.columns = ['Mean_audio_inference']

    df_audio_std = df_audio.groupby('date').std(ddof=0)
    df_audio_std.columns = ['Std_audio_inference']

    df_audio_median = df_audio.groupby('date').median()
    df_audio_median.columns = ['Median_audio_inference']

    df_audio_min = df_audio.groupby('date').min()
    df_audio_min.columns = ['Min_audio_inference']

    df_audio_max = df_audio.groupby('date').max()
    df_audio_max.columns = ['Max_audio_inference']

    df_audio_skew = df_audio.groupby('date').skew()
    df_audio_skew.columns = ['Skew_audio_inference']

    df_audio_var = df_audio.groupby('date').var(ddof=0)
    df_audio_var.columns = ['Var_audio_inference']

    df_audio_sum = df_audio.groupby('date').sum()
    df_audio_sum.columns = ['Sum_audio_inference']


    # In[32]:


    data_audio = [df_audio_mean["Mean_audio_inference"], df_audio_std["Std_audio_inference"],
            df_audio_median['Median_audio_inference'], df_audio_min['Min_audio_inference'], 
            df_audio_max['Max_audio_inference'], df_audio_skew['Skew_audio_inference'],
           df_audio_var['Var_audio_inference'], df_audio_sum['Sum_audio_inference']]

    headers_audio = ['Mean_audio_inference', 'Std_audio_inference','Median_audio_inference',
               'Min_audio_inference', 'Max_audio_inference', 'Skew_audio_inference',
              'Var_audio_inference', 'Sum_audio_inference']

    df_audio_data = pd.concat(data_audio, axis=1, keys=headers_audio)
    df_audio_data = final_check(df_audio_data)
    #df_audio_data


    # ## Conversation

    # In[33]:


    df_conv_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/conversation/conversation_'+uid+'.csv')
    #df_conv_raw


    # In[34]:


    df_conv_raw['start_date'] = df_conv_raw['start_timestamp'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_conv_raw['start_time'] = df_conv_raw['start_timestamp'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))

    df_conv_raw['end_date'] = df_conv_raw[' end_timestamp'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_conv_raw['end_time'] = df_conv_raw[' end_timestamp'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))


    # In[35]:


    df_conv = df_conv_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    #df_conv


    # In[36]:


    df_conv['conv_duration'] = duration_calc(df_conv)


    # In[37]:


    df_conv_clean = df_conv[['start_date', 'conv_duration']]

    if len(df_conv_clean.columns[df_conv_clean.isna().any()].tolist())!=0:
        for name in df_conv_clean.columns[df_conv_clean.isna().any()].tolist():
            df_conv_clean[name] = data_imputer(df_conv_clean, name)
    else:
        pass


    # In[38]:


    df_conv_mean = df_conv_clean.groupby('start_date').mean()
    df_conv_mean.columns = ['Mean_conv_duration']

    df_conv_std = df_conv_clean.groupby('start_date').std(ddof=0)
    df_conv_std.columns = ['Std_conv_duration']

    df_conv_median = df_conv_clean.groupby('start_date').median()
    df_conv_median.columns = ['Median_conv_duration']

    df_conv_min = df_conv_clean.groupby('start_date').min()
    df_conv_min.columns = ['Min_conv_duration']

    df_conv_max = df_conv_clean.groupby('start_date').max()
    df_conv_max.columns = ['Max_conv_duration']

    df_conv_skew = df_conv_clean.groupby('start_date').skew()
    df_conv_skew.columns = ['Skew_conv_duration']

    df_conv_var = df_conv_clean.groupby('start_date').var(ddof=0)
    df_conv_var.columns = ['Var_conv_duration']


    # In[39]:


    data_conv = [df_conv_mean["Mean_conv_duration"], df_conv_std["Std_conv_duration"],
            df_conv_median['Median_conv_duration'], df_conv_min['Min_conv_duration'], 
            df_conv_max['Max_conv_duration'], df_conv_skew['Skew_conv_duration'], df_conv_var['Var_conv_duration']]

    headers_conv = ['Mean_conv_duration', 'Std_conv_duration','Median_conv_duration',
               'Min_conv_duration', 'Max_conv_duration', 'Skew_conv_duration', 'Var_cov_duration']

    df_conv_data = pd.concat(data_conv, axis=1, keys=headers_conv)
    df_conv_data.index.name = 'date'
    df_conv_data = final_check(df_conv_data)
    #df_conv_data


    # ## Dark

    # In[40]:


    df_dark_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/dark/dark_'+uid+'.csv')
    #df_dark_raw


    # In[41]:


    df_dark_raw['start_date'] = df_dark_raw['start'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_dark_raw['start_time'] = df_dark_raw['start'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))

    df_dark_raw['end_date'] = df_dark_raw['end'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_dark_raw['end_time'] = df_dark_raw['end'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))


    # In[42]:


    df_dark = df_dark_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    #df_dark


    # In[43]:


    df_dark['dark_duration'] = duration_calc(df_dark)


    # In[44]:


    df_dark_clean = df_dark[['start_date', 'dark_duration']]

    if len(df_dark_clean.columns[df_dark_clean.isna().any()].tolist())!=0:
        for name in df_dark_clean.columns[df_dark_clean.isna().any()].tolist():
            df_dark_clean[name] = data_imputer(df_dark_clean, name)
    else:
        pass


    # In[45]:


    df_dark_mean = df_dark_clean.groupby('start_date').mean()
    df_dark_mean.columns = ['Mean_dark_duration']

    df_dark_std = df_dark_clean.groupby('start_date').std(ddof=0)
    df_dark_std.columns = ['Std_dark_duration']

    df_dark_median = df_dark_clean.groupby('start_date').median()
    df_dark_median.columns = ['Median_dark_duration']

    df_dark_min = df_dark_clean.groupby('start_date').min()
    df_dark_min.columns = ['Min_dark_duration']

    df_dark_max = df_dark_clean.groupby('start_date').max()
    df_dark_max.columns = ['Max_dark_duration']

    df_dark_skew = df_dark_clean.groupby('start_date').skew()
    df_dark_skew.columns = ['Skew_dark_duration']

    df_dark_var = df_dark_clean.groupby('start_date').var(ddof=0)
    df_dark_var.columns = ['Var_dark_duration']


    # In[46]:


    ## To discuss: should I impute calculated statistical values?


    # In[47]:


    data_dark = [df_dark_mean["Mean_dark_duration"], df_dark_std["Std_dark_duration"],
            df_dark_median['Median_dark_duration'], df_dark_min['Min_dark_duration'], 
            df_dark_max['Max_dark_duration'], df_dark_skew['Skew_dark_duration'], df_dark_var['Var_dark_duration']]

    headers_dark = ['Mean_dark_duration', 'Std_dark_duration','Median_dark_duration',
               'Min_dark_duration', 'Max_dark_duration', 'Skew_dark_duration', 'Var_dark_duration']

    df_dark_data = pd.concat(data_dark, axis=1, keys=headers_dark)
    df_dark_data.index.name = 'date'
    df_dark_data = final_check(df_dark_data)
    #df_dark_data


    # ## Gps

    # In[48]:


    df_gps_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/gps/gps_'+uid+'.csv', index_col=False)
    #df_gps_raw


    # In[49]:


    df_gps_raw['date'] = df_gps_raw['time'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    #df_gps_raw['time'] = df_gps_raw['time'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))


    # In[50]:


    df_gps = df_gps_raw[['date', 'latitude', 'longitude', 'altitude', 'bearing', 'speed', 'travelstate']]

    df_gps['travelstate_one_hot'] = [0 if val == 'stationary' else 1 for val in df_gps['travelstate']]
    df_gps = df_gps.drop(['travelstate', 'altitude', 'bearing', 'speed'], axis=1)

    if len(df_gps.columns[df_gps.isna().any()].tolist())!=0:
        for name in df_gps.columns[df_gps.isna().any()].tolist():
            if name == 'travelstate_one_hot':
                df_gps[name] = data_imputer(df_gps, name).astype(int)
            else:
                df_gps[name] = data_imputer(df_gps, name)
    else:
        pass


    # In[51]:


    df_gps_mean = df_gps.groupby('date').mean()
    df_gps_mean.columns = ['Mean_latitude', 'Mean_longitude', 'Mean_travelstate']

    df_gps_std = df_gps.groupby('date').std(ddof=0)
    df_gps_std.columns = ['Std_latitude', 'Std_longitude', 'Std_travelstate']

    df_gps_median = df_gps.groupby('date').median()
    df_gps_median.columns = ['Median_latitude', 'Median_longitude', 'Median_travelstate']

    df_gps_min = df_gps.groupby('date').min()
    df_gps_min.columns = ['Min_latitude', 'Min_longitude', 'Min_travelstate']

    df_gps_max = df_gps.groupby('date').max()
    df_gps_max.columns = ['Max_latitude', 'Max_longitude', 'Max_travelstate']

    df_gps_skew = df_gps.groupby('date').skew()
    df_gps_skew.columns = ['Skew_latitude', 'Skew_longitude', 'Skew_travelstate']

    df_gps_var = df_gps.groupby('date').var(ddof=0)
    df_gps_var.columns = ['Var_latitude', 'Var_longitude', 'Var_travelstate']

    df_gps_sum = df_gps.groupby('date').sum()
    df_gps_sum.columns = ['Sum_latitude', 'Sum_longitude', 'Sum_travelstate']


    # In[52]:


    data_gps = [df_gps_mean['Mean_latitude'], df_gps_mean['Mean_longitude'], df_gps_mean['Mean_travelstate'],
           df_gps_std['Std_latitude'], df_gps_std['Std_longitude'], df_gps_std['Std_travelstate'],
           df_gps_median['Median_latitude'], df_gps_median['Median_longitude'], 
            df_gps_median['Median_travelstate'],
           df_gps_min['Min_latitude'], df_gps_min['Min_longitude'], df_gps_min['Min_travelstate'],
           df_gps_max['Max_latitude'], df_gps_max['Max_longitude'], df_gps_max['Max_travelstate'],
           df_gps_skew['Skew_latitude'], df_gps_skew['Skew_longitude'], df_gps_skew['Skew_travelstate'],
           df_gps_var['Var_latitude'], df_gps_var['Var_longitude'], df_gps_var['Var_travelstate'],
               df_gps_sum['Sum_travelstate']]

    headers_gps = ['Mean_latitude', 'Mean_longitude', 'Mean_travelstate', 'Std_latitude', 
              'Std_longitude', 'Std_travelstate', 'Median_latitude', 'Median_longitude', 'Median_travelstate',
              'Min_latitude', 'Min_longitude', 'Min_travelstate', 'Max_latitude', 'Max_longitude', 
               'Max_travelstate', 'Skew_latitude', 'Skew_longitude', 'Skew_travelstate',
              'Var_latitude', 'Var_longitude', 'Var_travelstate', 'Sum_travelstate']

    df_gps_data = pd.concat(data_gps, axis=1, keys=headers_gps)
    df_gps_data = final_check(df_gps_data)
    df_gps_data['Mean_travelstate'] = df_gps_data['Mean_travelstate'].astype(int)
    #df_gps_data


    # ## Phone charge

    # In[53]:


    df_phone_charge_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/phonecharge/phonecharge_'+uid+'.csv')
    #df_phone_charge_raw


    # In[54]:


    df_phone_charge_raw['start_date'] = df_phone_charge_raw['start'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_phone_charge_raw['start_time'] = df_phone_charge_raw['start'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))

    df_phone_charge_raw['end_date'] = df_phone_charge_raw['end'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_phone_charge_raw['end_time'] = df_phone_charge_raw['end'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))


    # In[55]:


    df_phone_charge = df_phone_charge_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    #df_phone_charge


    # In[56]:


    df_phone_charge['Phone_charge_duration'] = duration_calc(df_phone_charge)


    # In[57]:


    df_phone_charge_clean = df_phone_charge[['start_date', 'Phone_charge_duration']]

    if len(df_phone_charge_clean.columns[df_phone_charge_clean.isna().any()].tolist())!=0:
        for name in df_phone_charge_clean.columns[df_phone_charge_clean.isna().any()].tolist():
            df_phone_charge_clean[name] = data_imputer(df_phone_charge_clean, name)
    else:
        pass


    # In[58]:


    df_phone_charge_mean = df_phone_charge_clean.groupby('start_date').mean()
    df_phone_charge_mean.columns = ['Mean_phone_charge_duration']

    df_phone_charge_std = df_phone_charge_clean.groupby('start_date').std(ddof=0)
    df_phone_charge_std.columns = ['Std_phone_charge_duration']

    df_phone_charge_median = df_phone_charge_clean.groupby('start_date').median()
    df_phone_charge_median.columns = ['Median_phone_charge_duration']

    df_phone_charge_min = df_phone_charge_clean.groupby('start_date').min()
    df_phone_charge_min.columns = ['Min_phone_charge_duration']

    df_phone_charge_max = df_phone_charge_clean.groupby('start_date').max()
    df_phone_charge_max.columns = ['Max_phone_charge_duration']

    df_phone_charge_skew = df_phone_charge_clean.groupby('start_date').skew()
    df_phone_charge_skew.columns = ['Skew_phone_charge_duration']

    df_phone_charge_var = df_phone_charge_clean.groupby('start_date').var(ddof=0)
    df_phone_charge_var.columns = ['Var_phone_charge_duration']


    # In[60]:


    data_charge = [df_phone_charge_mean["Mean_phone_charge_duration"], df_phone_charge_std["Std_phone_charge_duration"],
            df_phone_charge_median['Median_phone_charge_duration'], 
            df_phone_charge_min['Min_phone_charge_duration'], 
            df_phone_charge_max['Max_phone_charge_duration'], df_phone_charge_skew['Skew_phone_charge_duration'],
           df_phone_charge_var['Var_phone_charge_duration']]

    headers_charge = ['Mean_phone_charge_duration', 'Std_phone_charge_duration','Median_phone_charge_duration',
               'Min_phone_charge_duration', 'Max_phone_charge_duration', 'Skew_phone_charge_duration',
              'Var_phone_charge_duration']

    df_phone_charge_data = pd.concat(data_charge, axis=1, keys=headers_charge)
    df_phone_charge_data.index.name = 'date'
    df_phone_charge_data = final_check(df_phone_charge_data)
    #df_phone_charge_data


    # ## Phonelock

    # In[61]:


    df_phone_lock_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/phonelock/phonelock_'+uid+'.csv')
    #df_phone_lock_raw


    # In[62]:


    df_phone_lock_raw['start_date'] = df_phone_lock_raw['start'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_phone_lock_raw['start_time'] = df_phone_lock_raw['start'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))

    df_phone_lock_raw['end_date'] = df_phone_lock_raw['end'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))
    df_phone_lock_raw['end_time'] = df_phone_lock_raw['end'].apply(lambda x: time.strftime("%H:%M:%S",time.gmtime(x)))


    # In[63]:


    df_phone_lock = df_phone_lock_raw[['start_date', 'start_time', 'end_date', 'end_time']]
    #df_phone_lock


    # In[64]:


    df_phone_lock['Phone_lock_duration'] = duration_calc(df_phone_lock)


    # In[65]:


    df_phone_lock_clean = df_phone_lock[['start_date', 'Phone_lock_duration']]

    if len(df_phone_lock_clean.columns[df_phone_lock_clean.isna().any()].tolist())!=0:
        for name in df_phone_lock_clean.columns[df_phone_lock_clean.isna().any()].tolist():
            df_phone_lock_clean[name] = data_imputer(df_phone_lock_clean, name)
    else:
        pass


    # In[66]:


    df_phone_lock_mean = df_phone_lock_clean.groupby('start_date').mean()
    df_phone_lock_mean.columns = ['Mean_phone_lock_duration']

    df_phone_lock_std = df_phone_lock_clean.groupby('start_date').std(ddof=0)
    df_phone_lock_std.columns = ['Std_phone_lock_duration']

    df_phone_lock_median = df_phone_lock_clean.groupby('start_date').median()
    df_phone_lock_median.columns = ['Median_phone_lock_duration']

    df_phone_lock_min = df_phone_lock_clean.groupby('start_date').min()
    df_phone_lock_min.columns = ['Min_phone_lock_duration']

    df_phone_lock_max = df_phone_lock_clean.groupby('start_date').max()
    df_phone_lock_max.columns = ['Max_phone_lock_duration']

    df_phone_lock_skew = df_phone_lock_clean.groupby('start_date').skew()
    df_phone_lock_skew.columns = ['Skew_phone_lock_duration']

    df_phone_lock_var = df_phone_lock_clean.groupby('start_date').var(ddof=0)
    df_phone_lock_var.columns = ['Var_phone_lock_duration']


    # In[68]:


    data_lock = [df_phone_lock_mean["Mean_phone_lock_duration"], df_phone_lock_std["Std_phone_lock_duration"],
            df_phone_lock_median['Median_phone_lock_duration'], df_phone_lock_min['Min_phone_lock_duration'], 
            df_phone_lock_max['Max_phone_lock_duration'], df_phone_lock_skew['Skew_phone_lock_duration'], 
           df_phone_lock_var['Var_phone_lock_duration']]

    headers_lock = ['Mean_phone_lock_duration', 'Std_phone_lock_duration','Median_phone_lock_duration',
               'Min_phone_lock_duration', 'Max_phone_lock_duration', 'Skew_phone_lock_duration',
              'Var_phone_lock_duration']

    df_phone_lock_data = pd.concat(data_lock, axis=1, keys=headers_lock)
    df_phone_lock_data.index.name = 'date'
    df_phone_lock_data = final_check(df_phone_lock_data)
    #df_phone_lock_data


    # ## Wifi location

    # In[69]:


    df_wifi_loc_raw = pd.read_csv('/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/sensing/wifi_location/wifi_location_'+uid+'.csv', index_col = False)
    #df_wifi_loc_raw


    # In[70]:


    df_wifi_loc_raw['date'] = df_wifi_loc_raw['time'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))


    # In[71]:


    df_wifi_loc = df_wifi_loc_raw[['date', 'location']]


    # In[72]:


    key_feature_list = []
    loc_feature_list =[]
    for key, item in df_wifi_loc.groupby('date'):
        key_feature_list.append(key)
        loc_feature_list.append(len(item['location'].unique()))


    # In[73]:


    ## feature will be the number of distinct locations in a given day -> Measure of mobility
    df_wifi_loc_data = pd.DataFrame(loc_feature_list, columns = ['number of distinct locations'], index = key_feature_list)
    df_wifi_loc_data.index.name = 'date'
    #df_wifi_loc_data


    

    # # EMA data:

    # In[76]:


    ## calculate the length of the dataframe for eveery user for their emas e see which participantas I can use


    # ## PAM

    # In[79]:


    file = '/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/EMA/response/PAM/PAM_'+uid+'.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)
        
    pam_ema_raw = pd.DataFrame.from_dict(dict_train)
    pam_ema_clean=pam_ema_raw
    pam_ema_clean['date'] = pam_ema_clean['resp_time'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))

    pam_ema = pam_ema_clean[['date', 'picture_idx']]

    if len(pam_ema.columns[pam_ema.isna().any()].tolist())!=0:
        for name in pam_ema.columns[pam_ema.isna().any()].tolist():
            pam_ema[name] = data_imputer(pam_ema, name).astype(int)
    else:
        pass


    # In[80]:


    pam_ema_mean = pam_ema.groupby('date').mean().astype(int)
    pam_ema_mean.columns = ['Mean_pic_id']

    pam_ema_std = pam_ema.groupby('date').std(ddof=0)
    pam_ema_std.columns = ['Std_pic_id']

    pam_ema_median = pam_ema.groupby('date').median()
    pam_ema_median.columns = ['Median_pic_id']

    pam_ema_min = pam_ema.groupby('date').min()
    pam_ema_min.columns = ['Min_pic_id']

    pam_ema_max = pam_ema.groupby('date').max()
    pam_ema_max.columns = ['Max_pic_id']

    pam_ema_skew = pam_ema.groupby('date').skew()
    pam_ema_skew.columns = ['Skew_pic_id']

    pam_ema_var = pam_ema.groupby('date').var(ddof=0)
    pam_ema_var.columns = ['Var_pic_id']

    pam_ema_sum = pam_ema.groupby('date').sum()
    pam_ema_sum.columns = ['Sum_pic_id']


    # In[81]:


    data_pam = [pam_ema_mean["Mean_pic_id"], pam_ema_std["Std_pic_id"],
            pam_ema_median['Median_pic_id'], pam_ema_min['Min_pic_id'], 
            pam_ema_max['Max_pic_id'], pam_ema_skew['Skew_pic_id'], 
           pam_ema_var['Var_pic_id'], pam_ema_sum['Sum_pic_id']]

    headers_pam = ['Mean_pic_id', 'Std_pic_id','Median_pic_id',
               'Min_pic_id', 'Max_pic_id', 'Skew_pic_id', 'Var_pic_id', 'Sum_pic_id']

    pam_ema_data = pd.concat(data_pam, axis=1, keys=headers_pam)
    pam_ema_data = final_check(pam_ema_data)
    #pam_ema_data


    # ## Sleep

    # In[87]:


    file = '/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/EMA/response/Sleep/Sleep_'+uid+'.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)
        
    sleep_ema_raw = pd.DataFrame.from_dict(dict_train)
    sleep_ema_clean = sleep_ema_raw.drop(['location'], axis=1)

    sleep_ema_clean['date'] = sleep_ema_clean['resp_time'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))

    sleep_ema = sleep_ema_clean[['date', 'hour', 'rate', 'social']]

    if len(sleep_ema.columns[sleep_ema.isna().any()].tolist())!=0:
        for name in sleep_ema.columns[sleep_ema.isna().any()].tolist():
            sleep_ema[name] = data_imputer(sleep_ema, name).astype(int)
    else:
        sleep_ema['hour'] = sleep_ema['hour'].astype(int)
        sleep_ema['rate'] = sleep_ema['hour'].astype(int)
        sleep_ema['social'] = sleep_ema['hour'].astype(int)

    #sleep_ema


    # In[88]:


    sleep_ema_mean = sleep_ema.groupby('date').mean()
    sleep_ema_mean.columns = ['Mean_hour', 'Mean_rate', 'Mean_social']

    sleep_ema_std = sleep_ema.groupby('date').std(ddof=0)
    sleep_ema_std.columns = ['Std_hour', 'Std_rate', 'Std_social']

    sleep_ema_median = sleep_ema.groupby('date').median()
    sleep_ema_median.columns = ['Median_hour', 'Median_rate', 'Median_social']

    sleep_ema_min = sleep_ema.groupby('date').min()
    sleep_ema_min.columns = ['Min_hour', 'Min_rate', 'Min_social']

    sleep_ema_max = sleep_ema.groupby('date').max()
    sleep_ema_max.columns = ['Max_hour', 'Max_rate', 'Max_social']

    sleep_ema_skew = sleep_ema.groupby('date').skew()
    sleep_ema_skew.columns = ['Skew_hour', 'Skew_rate', 'Skew_social']

    sleep_ema_var = sleep_ema.groupby('date').var(ddof=0)
    sleep_ema_var.columns = ['Var_hour', 'Var_rate', 'Var_social']


    # In[89]:


    data_sleep = [sleep_ema_mean["Mean_hour"], sleep_ema_mean["Mean_rate"], sleep_ema_mean["Mean_social"],
            sleep_ema_std["Std_hour"], sleep_ema_std["Std_rate"], sleep_ema_std["Std_social"],
            sleep_ema_median['Median_hour'], sleep_ema_median['Median_rate'], sleep_ema_median['Median_social'],
            sleep_ema_min['Min_hour'], sleep_ema_min['Min_rate'], sleep_ema_min['Min_social'], 
            sleep_ema_max['Max_hour'], sleep_ema_max['Max_rate'], sleep_ema_max['Max_social'],
           sleep_ema_skew['Skew_hour'], sleep_ema_skew['Skew_rate'], sleep_ema_skew['Skew_social'], 
           sleep_ema_var['Var_hour'], sleep_ema_var['Var_rate'], sleep_ema_var['Var_social']]

    headers_sleep = ["Mean_hour_sleep", "Mean_rate_sleep", "Mean_social_sleep",
              "Std_hour_sleep", "Std_rate_sleep", "Std_social_sleep", 
              'Median_hour_sleep', 'Median_rate_sleep', 'Median_social_sleep',
              'Min_hour_sleep', 'Min_rate_sleep', 'Min_social_sleep', 
              'Max_hour_sleep', 'Max_rate_sleep', 'Max_social_sleep',
              'Skew_hour_sleep', 'Skew_rate_sleep', 'Skew_social_sleep',
              'Var_hour_sleep', 'Var_rate_sleep', 'Var_social_sleep']

    sleep_ema_data = pd.concat(data_sleep, axis=1, keys=headers_sleep)
    sleep_ema_data = final_check(sleep_ema_data)
    #sleep_ema_data


    # ## Social

    # In[90]:


    file = '/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/EMA/response/Social/Social_'+uid+'.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)
        
    social_ema_clean = pd.DataFrame.from_dict(dict_train)

    social_ema_clean['date'] = social_ema_clean['resp_time'].apply(lambda x: time.strftime("%Y-%m-%d",time.gmtime(x)))

    social_ema = social_ema_clean[['date','number']]
    social_ema['number'].iloc[2]=4
    social_ema['number'].iloc[3]=3

    if len(social_ema.columns[social_ema.isna().any()].tolist())!=0:
        for name in social_ema.columns[social_ema.isna().any()].tolist():
            social_ema[name] = data_imputer(social_ema, name).astype(int)
    else:
        social_ema['number'] = social_ema['number'].astype(int)
    #social_ema


    # In[91]:


    social_ema_mean = social_ema.groupby('date').mean()
    social_ema_mean.columns = ['Mean_number']

    social_ema_std = social_ema.groupby('date').std(ddof=0)
    social_ema_std.columns = ['Std_number']

    social_ema_median = social_ema.groupby('date').median()
    social_ema_median.columns = ['Median_number']

    social_ema_min = social_ema.groupby('date').min()
    social_ema_min.columns = ['Min_number']

    social_ema_max = social_ema.groupby('date').max()
    social_ema_max.columns = ['Max_number']

    social_ema_skew = social_ema.groupby('date').skew()
    social_ema_skew.columns = ['Skew_number']

    social_ema_var = social_ema.groupby('date').var(ddof=0)
    social_ema_var.columns = ['Var_number']

    social_ema_sum = social_ema.groupby('date').sum()
    social_ema_sum.columns = ['Sum_number']


    # In[92]:


    data_social = [social_ema_mean["Mean_number"], social_ema_std["Std_number"],
            social_ema_median['Median_number'], social_ema_min['Min_number'], 
            social_ema_max['Max_number'], social_ema_skew['Skew_number'], social_ema_var['Var_number'],
                  social_ema_sum['Sum_number']]

    headers_social = ['Mean_number_social', 'Std_number_social','Median_number_social',
               'Min_number_social', 'Max_number_social', 'Skew_number_social', 'Var_number_social',
                     'Sum_number_social']

    social_ema_data = pd.concat(data_social, axis=1, keys=headers_social)
    social_ema_data = final_check(social_ema_data)
    #social_ema_data


    # ## Stress

    # In[93]:


    file = '/home/diogo_mota/Dropbox/QMUL/MSc_Project/dataset/EMA/response/Stress/Stress_'+uid+'.json'
    with open(file) as train_file:
        dict_train = json.load(train_file)

    stress_ema_raw = pd.DataFrame.from_dict(dict_train)
    stress_ema_clean = stress_ema_raw.drop(['location'], axis=1)

    stress_ema_clean['date'] = stress_ema_clean['resp_time'].apply(lambda x: time.strftime("%Y-%m-%d",
                                                                                           time.gmtime(x)))
    stress_ema_clean['time'] = stress_ema_clean['resp_time'].apply(lambda x: time.strftime("%H:%M:%S",
                                                                                           time.gmtime(x)))

    stress_ema = stress_ema_clean[['date', 'level']]
    stress_ema['level'].iloc[3]=1
    stress_ema['level'].iloc[4]=1

    if len(stress_ema.columns[stress_ema.isna().any()].tolist())!=0:
        for name in stress_ema.columns[stress_ema.isna().any()].tolist():
            stress_ema[name] = data_imputer(stress_ema, name).astype(int)
    else:
        stress_ema['level'] = stress_ema['level'].astype(int)


    # In[94]:


    stress_ema_mean = stress_ema.groupby('date').mean()
    stress_ema_mean.columns = ['Mean_stress_level']

    stress_ema_std = stress_ema.groupby('date').std(ddof=0)
    stress_ema_std.columns = ['Std_stress_level']

    stress_ema_median = stress_ema.groupby('date').median()
    stress_ema_median.columns = ['Median_stress_level']

    stress_ema_min = stress_ema.groupby('date').min()
    stress_ema_min.columns = ['Min_stress_level']

    stress_ema_max = stress_ema.groupby('date').max()
    stress_ema_max.columns = ['Max_stress_level']

    stress_ema_skew = stress_ema.groupby('date').skew()
    stress_ema_skew.columns = ['Skew_stress_level']

    stress_ema_var = stress_ema.groupby('date').var(ddof=0)
    stress_ema_var.columns = ['Var_stress_level']

    stress_ema_sum = stress_ema.groupby('date').sum()
    stress_ema_sum.columns = ['Sum_stress_level']


    # In[96]:


    data_stress = [stress_ema_mean["Mean_stress_level"], stress_ema_std["Std_stress_level"],
            stress_ema_median['Median_stress_level'], stress_ema_min['Min_stress_level'], 
            stress_ema_max['Max_stress_level'], stress_ema_skew['Skew_stress_level'], 
           stress_ema_var['Var_stress_level'], stress_ema_sum['Sum_stress_level']]

    headers_stress = ['Mean_stress_level', 'Std_stress_level','Median_stress_level',
               'Min_stress_level', 'Max_stress_level', 'Skew_stress_level',
              'Var_stress_level', 'Sum_stress_level']

    stress_ema_data = pd.concat(data_stress, axis=1, keys=headers_stress)
    stress_ema_data = final_check(stress_ema_data)
    #stress_ema_data


    # # Student Feature dataframe

    # In[97]:


    ## decide which features to actually use since too many get may training data too short


    # In[98]:


    student_df = pd.merge(df_activity_data, df_audio_data, left_index=True, right_index=True)

    df_list = [df_conv_data, df_dark_data, df_gps_data,
              df_phone_charge_data, df_phone_lock_data, df_wifi_loc_data,
            pam_ema_data, sleep_ema_data, social_ema_data, stress_ema_data]

    for df in df_list:
        student_df = pd.merge(student_df, df, left_index=True, right_index=True)
    
    return student_df
#student_df


# In[ ]:




