import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            return str(obj)
        return super(CustomJSONEncoder, self).default(obj)

aDict = {
    # Your Firebase credentials here
}

jsonString = json.dumps(aDict)
jsonFile = open("./serviceAccountKey.json", "w")
jsonFile.write(jsonString)
jsonFile.close()

# Load the JSON file containing the Firebase service account key
cred = credentials.Certificate('./serviceAccountKey.json')

# Check if the default Firebase app already exists, and initialize the app if it doesn't
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://forecasting-1609-default-rtdb.asia-southeast1.firebasedatabase.app/'
    }, name='forecasting')

df = pd.read_csv('./future_predictions.csv')

# Convert the dataframe to a dictionary
data_dict = df.to_dict(orient='records')

# Serialize the data_dict using the custom JSON encoder
json_data = json.dumps(data_dict, cls=CustomJSONEncoder)

# Deserialize the JSON data back to a Python object
data_dict_str = json.loads(json_data)

# Upload the data to Firebase under a new child node named 'dataframe'
data_ref = db.reference('dataframe')
data_ref.set(data_dict_str)

print('Data uploaded successfully')