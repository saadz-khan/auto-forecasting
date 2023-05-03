import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


# Initialize the Firebase app with the credentials and database URL
firebase_admin.initialize_app({
    'databaseURL': 'https://forecasting-1609-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

df = pd.read_csv('./future_predictions.csv')

# Convert the dataframe to a dictionary
data_dict = df.to_dict(orient='records')

# Upload the data to Firebase under a new child node named 'dataframe'
data_ref = db.reference('dataframe')
data_ref.set(data_dict)

print('Data uploaded successfully')
