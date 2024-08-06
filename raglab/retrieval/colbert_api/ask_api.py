
import requests
import pdb
import time
from pprint import pprint
query = "What is the airspeed velocity of an unladen swallow?"
k = 10
url = f"http://localhost:8893/api/search?query={query}&k={k}"
start_time = time.time()
response = requests.get(url)
response = response.json()
delay = time.time() - start_time
pprint(response)
print(f"Time taken: {delay:.3f} seconds")