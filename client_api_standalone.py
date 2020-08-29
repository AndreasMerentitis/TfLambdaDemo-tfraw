import logging
import time
try:
    from urllib2 import urlopen
    from urllib2 import HTTPError
except ImportError:
    from urllib.request import urlopen
    from urllib.error import HTTPError


import os
import subprocess
import click

import requests
import json


import pdb


@click.command(short_help='Classifies datapoints from json input file.')

def datapoint_classification_client(apollo_opts):
    """Classifies datapoints from json input file.

    Takes one or more datapoints and returns the results from a TensorFlow"""
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    with open('data_input.json') as json_file:  
        points = json.load(json_file)

    logger.info('Predicting...')
    start_time = time.time()

    method = 'POST'
    headers = {'Content-Type': 'application/json'} 
    service = 'execute-api'
    data1 = 'https://qcqzz1dqtd.execute-api.eu-west-1.amazonaws.com/Test_Carparts47_LSQSL/carparts47-lsqsl'
    data2 = 'https://xjnv2iidq2.execute-api.eu-west-1.amazonaws.com/API_Carparts47_LTSQSL/carparts47-ltsqsl'
    region = 'eu-west-1'

    print('')
    print(data1)
    print(data2)
    print('')

    logger.info('Creating request...')

    auth = {}
    data = {}
    data['data_to_classify'] = tuple(points['features'])
    json_data = json.dumps(data)

    json_response1 = requests.request(method, url1, auth=auth, data=json_data, headers=headers)
    json_response2 = requests.request(method, url2, auth=auth, data=json_data, headers=headers)

    end_time = time.time()
    
    # Extract text from JSON
    response1 = json.loads(json_response1.text)
    response2 = json.loads(json_response2.text)

    flattened_response1 = [val for sublist in response1 for val in sublist]
    flattened_response2 = [val for sublist in response2 for val in sublist]

    for d1, d2 in zip(flattened_response1, flattened_response2):
       if isinstance(d1, str) or isinstance(d2, str):
          pass
       else:
          for key, value in d1.items():
              if value != d2[key]:
                 print (key, value, d2[key])


    print ('')
    print(flattened_response1)

    print ('')
    print(flattened_response2)
    logger.info('End-to-end prediction (including network transfers) of '
                '{} images took {} seconds.'.format(
                    len(urls['image_urls']), end_time - start_time))


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


if __name__ == '__main__':
    image_classification_client()




