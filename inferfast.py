try:
  import unzip_requirements
except ImportError:
  pass

import json
import os
import tarfile

import boto3
import tensorflow as tf
import numpy as np

import census_data

import logging

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

FILE_DIR = '/tmp/'
BUCKET = os.environ['BUCKET']

import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from boto3 import client as boto3_client

q = queue.Queue()
lambda_client = boto3_client('lambda')


def feed_the_workers(datapoints, spacing):
    """ Simulate outside actors sending in work to do, request each url twice """
    for datapoint in datapoints:
        #time.sleep(spacing)
        q.put(datapoint)
    return "DONE FEEDING"

def process_one_datapoint(executor, payload_one_item):
    """ Process a single item """
    
    epoch_files = '' 
    # Download model from S3 and extract
    boto3.Session(
        ).resource('s3'
        ).Bucket(BUCKET
        ).download_file(
            os.path.join(epoch_files,'model.tar.gz'),
            FILE_DIR+'model.tar.gz')

    tarfile.open(FILE_DIR+'model.tar.gz', 'r').extractall(FILE_DIR)

    # Create feature columns
    wide_cols, deep_cols = census_data.build_model_columns()

    # Load model
    classifier = tf.estimator.LinearClassifier(
                    feature_columns=wide_cols,
                    model_dir=FILE_DIR+'tmp/model_'+epoch_files+'/',
                    warm_start_from=FILE_DIR+'tmp/model_'+epoch_files+'/')

    # Setup prediction
    predict_iter = classifier.predict(
                        lambda:_easy_input_function(payload_one_item))

    # Iterate over prediction and convert to lists
    predictions = []
    for prediction in predict_iter:
        for key in prediction:
            prediction[key] = prediction[key].tolist()

        predictions.append(prediction)

    return predictions
    

def _easy_input_function(data_dict, batch_size=64):
    """
    data_dict = {
        '<csv_col_1>': ['<first_pred_value>', '<second_pred_value>']
        '<csv_col_2>': ['<first_pred_value>', '<second_pred_value>']
        ...
    }
    """

    # Convert input data to numpy arrays
    for col in data_dict:
        col_ind = census_data._CSV_COLUMNS.index(col)
        dtype = type(census_data._CSV_COLUMN_DEFAULTS[col_ind][0])
        data_dict[col] = np.array(data_dict[col],
                                        dtype=dtype)

    try: 
        labels = data_dict.pop('income_bracket')
    except:
        pass

    ds = tf.data.Dataset.from_tensor_slices((data_dict, labels))
    ds = ds.batch(64)

    return ds
    

def inferfastHandler(event, context):
    body = json.loads(event.get('body'))

    # Read in prediction data as dictionary
    # Keys should match _CSV_COLUMNS, values should be lists
    predict_input = body['input']
    
    logging.warning('predict_input type is %s', type(predict_input))
    logging.warning('predict_input is %s', predict_input)
    
    # Read in epoch
    epoch_files = body['epoch']
    logging.warning('epoch_files is %s', epoch_files)
    
    if isinstance(predict_input, list): 
        predict_datapoints = predict_input
    else: 
        predict_datapoints = [predict_input]
        
    logging.warning('predict_datapoints is %s', predict_datapoints)
    
    results = []
    results_datapoint_order = []
    
    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    
        # start a future for a thread which sends work in through the queue
        future_to_datapoint = {
            executor.submit(feed_the_workers, predict_datapoints, 0): 'FEEDER DONE'}
    
        while future_to_datapoint:
            # check for status of the futures which are currently working
            done, not_done = concurrent.futures.wait(
                future_to_datapoint, timeout=0.0,
                return_when=concurrent.futures.FIRST_COMPLETED)
    
            # if there is incoming work, start a new future
            while not q.empty():
    
                # fetch a url from the queue
                datapoint = q.get()
                
                payload_one_item = {'data_point': [datapoint]}
                logging.warning('payload_one_item value is %s', payload_one_item)
    
                # Start the load operation and mark the future with its URL
                future_to_datapoint[executor.submit(process_one_datapoint, executor, payload_one_item)] = payload_one_item
    
            # process any completed futures
            for future in done:
                datapoint = future_to_datapoint[future]
                try:
                    data = future.result()
                    data = json.loads(data)
                    logging.warning('data value is %s', data) 
                    results.append(data)
                    results_datapoint_order.append(datapoint)
                except Exception as exc:
                    print('%r generated an exception: %s' % (datapoint, exc))
                else:
                    if datapoint == 'FEEDER DONE':
                        print(data)
                    else:
                        print('%r page is %d bytes' % (datapoint, len(data)))
    
                # remove the now completed future
                del future_to_datapoint[future]
    
    datapoints_result_order = []
    for item_in_list in results_datapoint_order:
        datapoints_result = item_in_list['predict_datapoints']
        datapoints_result_order.append(datapoints_result[0])

    order_list_idx = []
    for item_in_list in datapoints_result_order:
        order_list_idx.append(predict_datapoints.index(item_in_list))
    
    logging.warning('predict_datapoints value is %s', predict_datapoints) 
    logging.warning('results_datapoint_order value is %s', results_datapoint_order) 
    logging.warning('order_list_idx value is %s', order_list_idx)
    
    results_ordered = [x for _,x in sorted(zip(order_list_idx,results))]
    
    logging.warning('results value is %s', results) 
    logging.warning('results_ordered value is %s', results_ordered) 

    response = {
        "statusCode": 200,
        "body": json.dumps(results_ordered,
                            default=lambda x: x.decode('utf-8'))
    }

    return response
