import numpy as np
import pandas as pd
import datetime as dt

gen_1=pd.read_csv('/content/Plant_1_Generation_Data.csv')
gen_1.drop('PLANT_ID',1,inplace=True)
sens_1= pd.read_csv('/content/Plant_1_Weather_Sensor_Data.csv')
sens_1.drop('PLANT_ID',1,inplace=True)

#format datetime
gen_1['DATE_TIME']= pd.to_datetime(gen_1['DATE_TIME'],format='%d-%m-%Y %H:%M')
sens_1['DATE_TIME']= pd.to_datetime(sens_1['DATE_TIME'],format='%Y-%m-%d %H:%M:%S')

df_gen=gen_1.groupby('DATE_TIME').sum().reset_index()
df_gen['time']=df_gen['DATE_TIME'].dt.time

final_df = pd.merge(df_gen, sens_1, how='outer', on='DATE_TIME')
final_df.drop(['SOURCE_KEY'], axis=1, inplace=True)
final_df.to_csv('./final_data.csv')