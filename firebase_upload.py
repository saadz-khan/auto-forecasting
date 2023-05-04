import pandas as pd
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import json

adict = {
  "type": "service_account",
  "project_id": "forecasting-1609",
  "private_key_id": "b0d3e147d92e6706e4d980f2877adf089db9bd22",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDF/lNKpE+JMnLp\nwfB5ju1xov9SJSJjLIrkoxwWYYQVv/UBUqNsJKAsedrtpOGVTtrXG22x4BKe5PNQ\n9N9Tmzpi9L+JKtOrIQUsDzzkx83xjDXX57FUIPi23XeXqnw292xDBFqmcOeKem6Q\nnf0iHUXIczEj4wXc56MPj07IhrZ83vJt8ewHW+JxioUg3TYSlgc5Svdrhij5loK+\nDk3zhgt6tRsn/O2cweC706kP+v68wMvGhzJZ3yF/aOm+eJS63zQ8ciMaRLe4wnZV\niBmNaLzbfMMGUAfsAyDn8+hHBb07ycshvyEgBNxOLRjpUzzeYsQWnlKs9QP0FQcJ\nA/20rxV1AgMBAAECggEAVBHK5RlCqmNQ9t/CD6N4P/Wxivw4jzrFGRMHJJuwfZm9\n2xtvR8Z/epMdOiX2ITNT6d18JPibf3uAs0Hep/3z+zuoLoKQUQwayZwIw63xTNy3\nqh73pVzArsj39F7QEnjZr9Uj9qNfNWYKF372Emuyi0waM/48KZHzo3I6xhupVfiH\nbffnbbvIz/T4C/7x4k+yRqNs1VLTwyp9ljIBPr72CUc3EI4ltSV5Eay3iGnKOwbk\nI8FWupzY9UmaPbML8jLFb5m2NWJgLfdpw11SSViieI6gvrtpWWwrqvjNOL9hJ8P0\nzC0nyOK/DXZKa4CM46TopjtO7MRNoleEupLd2KaMmwKBgQDjCMF/t+7DAJrqZDkY\n4+z/Fx/VFT1rocjmqPlKPvhUycxQ2eZfF4Ns0vs5VEwiSqkQqHJiFFTDgwcMg7oU\n/lAhdvU1FoufgacoVr+IXrOrNA4GCglxoDweLroDloYJ8+wNOjAKkp3lB2AmQc7n\nP35O4uzfa9YVMENnc8y8uJKs2wKBgQDfQQ88mafDyw58u6ETHGvlPbzWKs9t33UH\nmHgOpCdMGAd43keWet/aHEN8IzPSFYagCrwhVXIfAS8Sw2xtuLwGjj5S9vjBOKkY\nTw7MsCZp2lB0htCNPsThTz0Vck8eVGr8e65ZNQpYeOlwncQE66tBzFnBFOyW0hX+\nX79NR6Kv7wKBgQDdwvy3E2PeCCZJ01HOVTx2wxCt44xWKPaDEARJGI8ayu6O/vi+\nY8pVLxSDoCqpyEGx6sens/JD8yWcfMLzpKGCvp6vl3EUgeFUTr/RpadcEIGuNnmZ\nCooOdVfvD0WoTSGCnha1DRe70eOCGv/N88AdsBzt83huw/DjCUKwCJ4ExwKBgCVk\nv0mWLvxjaXTI9Iy/6O6HrbZtizrK/84a4pkJM6/SIoZDLNhukkn1C/Sh7M8WXes2\ngRZQOoIo75qD8whZtdyfPJ8m0LmEZQY2T7NHGdgHAgzSe+H8jntyizepYskguJ67\nlJNKiAYTkn4TXr64PoIR2UK2333PiHGH1gH6qqwvAoGBAKlMRdT4tiAtFP40S7qo\nHZEnhFj69zbinfxN+ODrVHIQWPBC3w8bW2EHSjQT+uTwWHIijVUV41Y/jKzbpmnN\nNLaszHTuSO1bDJwTQgqrCvvkafUWbf54krlKwNi0tG/0jzEJYaNbXN5UODpK1r5L\n9sKiRrc7/p1OwS0gcQe2Q+d7\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-lxedx@forecasting-1609.iam.gserviceaccount.com",
  "client_id": "117718759065308210613",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-lxedx%40forecasting-1609.iam.gserviceaccount.com"
}

jsonString = json.dumps(aDict)
jsonFile = open("serviceAccountKey.json", "w")
jsonFile.write(jsonString)
jsonFile.close()

# Load the JSON file containing the Firebase service account key
cred = credentials.Certificate('./serviceAccountKey.json')

# Initialize the Firebase app with the credentials and database URL
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://forecasting-1609-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

df = pd.read_csv('./future_predictions.csv')

# Convert the dataframe to a dictionary
data_dict = df.to_dict(orient='records')

# Upload the data to Firebase under a new child node named 'dataframe'
data_ref = db.reference('dataframe')
data_ref.set(data_dict)

print('Data uploaded successfully')
