import requests
import json
import csv
import time

firebase_url = "https://homeautomation-6f713-default-rtdb.firebaseio.com/.json"
weather_url = "https://weather-1609-default-rtdb.asia-southeast1.firebasedatabase.app/.json"
generation_url = ""

# Set the number of retries and delay between retries
max_retries = 5
retry_delay = 1

# Try to establish a connection to the Firebase Realtime Database
for retry_count in range(max_retries):
    try:
        response = requests.get(firebase_url, timeout=20)
        break
    except requests.exceptions.RequestException:
        print(f"Connection failed, retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)

# If all retries fail, print an error message and exit the script
else:
    print("Failed to establish a connection to the Firebase Realtime Database.")
    exit()

# If a connection was established, continue with processing the data
data = json.loads(response.content)
with open("weather.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["key", "value"])
    for key, value in data.items():
        writer.writerow([key, value])