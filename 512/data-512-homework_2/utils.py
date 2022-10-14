import time
import urllib.parse

import requests
from unidecode import unidecode


def download_csv(url, filename):
    """Download a CSV file from a URL and save it to disk."""
    # ensure utf-8 encoding
    response = requests.get(url, headers={'Content-Type': 'text/csv; charset=utf-8'})

    with open(filename, 'wb') as f:
        f.write(response.content)

API_LATENCY_ASSUMED = 0.002       # Assuming roughly 2ms latency on the API and network
API_THROTTLE_WAIT = (1.0/100.0)-API_LATENCY_ASSUMED

# TODO: refactor code to remove duplicate HTTP cruft
def get_article_rating(revid):
    # now, create a request URL by combining the endpoint_url with the parameters for the request
    request_url = "https://ores.wikimedia.org/v3/scores/enwiki"
    
    params = {
        "models": "wp10",
        "revids": str(int(revid)),
    }
    

    # make the request
    try:
        # we'll wait first, to make sure we don't exceed the limit in the situation where an exception
        # occurs during the request processing - throttling is always a good practice with a free
        # data source like Wikipedia - or other community sources
        if API_THROTTLE_WAIT > 0.0:
            time.sleep(API_THROTTLE_WAIT)
        response = requests.get(request_url, params=params)
        json_response = response.json()
    except Exception as e:
        print(e)
        json_response = None
    return json_response

def get_article_revid(article_title):
    # Titles are supposed to have spaces replaced with "_" and be URL encoded
    article_title_encoded = urllib.parse.quote(article_title.replace(' ','_'))
    
    # now, create a request URL by combining the endpoint_url with the parameters for the request
    request_url = "https://en.wikipedia.org/w/api.php"
    
    params = {
        "action": "query",
        "format": "json",
        "titles": article_title,
        "prop": "info",
    }
    
    # make the request
    try:
        # we'll wait first, to make sure we don't exceed the limit in the situation where an exception
        # occurs during the request processing - throttling is always a good practice with a free
        # data source like Wikipedia - or other community sources
        if API_THROTTLE_WAIT > 0.0:
            time.sleep(API_THROTTLE_WAIT)
        response = requests.get(request_url, params=params)
        json_response = response.json()
    except Exception as e:
        print(e)
        json_response = None
    return json_response
