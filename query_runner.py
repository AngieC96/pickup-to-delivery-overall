# ----------------------------------------------------------------------------------------------------------------------
# Owner: Performance
# Description: This module allows to query any remote database in Glovo
# Input:
#   base_query_path: path to SQL query
#   query_name: name of the SQL query (without .sql suffix)
#   connection: output of the query_runner module
#   parameters_list: values for any parameters within the SQL query that follow the '[parameter_i]' format
#   parameters_dict: values for any parameters within the SQL query, represented by key to find and value to replace
#   replace_delim: the characters that identify parameters to replace using the parameters_dict - e.g. '[]' or '{}'
#   query_from_file: default TRUE, if FALSE will use the query_name parameter as the SQL string
#   is_athena: default FALSE, if TRUE will run query in athena
#   print_query: default FALSE, if TRUE will print the query used in the terminal (after replacing parameters)
#   chunk_size: default 100000, if query is too large, it will be split into chunks of this size
# Output: DataFrame with query output
# ----------------------------------------------------------------------------------------------------------------------


import pandas as pd
import os
import datetime
import numpy as np
import psycopg2 as pg2
import mysql.connector
import trino
import warnings
import boto3
import re
import time

DEFAULT_CHUNK_SIZE = 100000 # Default chunk size for queries

class Query:

    def __init__(self, base_query_path, query_name, connection, parameters_list=[], parameters_dict = {}, replace_delim = '[]', query_from_file = True, is_athena = False, print_query = False, chunk_size = DEFAULT_CHUNK_SIZE):
        self.base_query_path = base_query_path
        self.query_name = query_name
        self.parameters_list = parameters_list
        self.parameters_dict = parameters_dict
        self.replace_delim = replace_delim
        self.query_from_file = query_from_file
        self.final_query = ''
        self.result = np.nan
        self.connection = connection
        self.is_athena = is_athena
        self.print_query = print_query
        self.chunk_size = chunk_size

    def build_query(self):

        if self.query_from_file:
            query_path = self.base_query_path + self.query_name + '.sql'

            with open(os.path.expanduser(query_path)) as f:
                data = f.readlines()

            query = ''.join(data)
        else:
            query = self.query_name

        for i in range(len(self.parameters_list)):
            query = query.replace('[parameter_' + str(i) + ']', str(self.parameters_list[i]))

        for key, value in self.parameters_dict.items():
            query = query.replace(self.replace_delim[0] + key + self.replace_delim[1], value)
            
        self.final_query = query

        if self.print_query:
            print(query)

    def execute_query(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            if self.is_athena == False:
                df_lst = list()
                for chunk in pd.read_sql_query(self.final_query, self.connection, chunksize=self.chunk_size):
                    df_lst.append(chunk)
                    # print("Chunk: #", len(df_lst), " - ", chunk.shape[0], "rows")

                data = pd.concat(df_lst)

            else:
                self.connection['query'] = self.final_query
                data = read_from_s3(self.connection)
        self.result = data

    def disconnect_from_db(self):
        self.connection.close()

    def run(self):
        self.build_query()
        self.execute_query()

        return self.result


def create_connection(user=[], password=[], db='dwh'):
    if db == 'datalake':
        return (trino.dbapi.connect(host= 'starburst.g8s-data-platform-prod.glovoint.com',
                                   port= 443,
                                   http_scheme='https',
                                   auth= trino.auth.OAuth2Authentication(),
                                   timezone='UTC'))
    
    if db == 'dwh':
        return (pg2.connect(dbname='glovodwh',
                            host='glovo-dwh-prod.cmsc5llor91g.eu-west-1.redshift.amazonaws.com',
                            port=5439,
                            user=user,
                            password=password))
    
    if db == 'dwh_jenkins_instance':
        return (pg2.connect(dbname='glovodwh',
                            host='ds-redshift.prod.data.glovoint.com',
                            port=5439,
                            user=user,
                            password=password))

    if db == 'techopsdb':
        return (mysql.connector.connect(database='techopsdb',
                                        host='maxscale-prod-2.internal.glovoapp.com',
                                        port=3392,
                                        user=user,
                                        password=password))

    if db == 'livedb':
        return (mysql.connector.connect(database='glovo_live',
                                        host='maxscale-prod.internal.glovoapp.com',
                                        port=3306,
                                        user=user,
                                        password=password))

    if db == 'live_secondary':
        return (mysql.connector.connect(database='glovo',
                                        host='maxscale-prod.internal.glovoapp.com',
                                        port=3328,
                                        user=user,
                                        password=password))

    if db == 'dispatchingdb':
        return (mysql.connector.connect(database='glovo_live',
                                        host='maxscale-prod.internal.glovoapp.com',
                                        port=3315,
                                        user=user,
                                        password=password))

    if db == 'saturationdb':
        return (mysql.connector.connect(database='saturationdb',
                                        host='maxscale-prod.internal.glovoapp.com',
                                        port=3331,
                                        user=user,
                                        password=password))

    if db == 'athena_matching':
        return (
            {
                'region': 'eu-west-1',
                'database': 'matching',
                'bucket': 'dispatching-athena-query-results',
                'path': 'temp/'
            }
        )

def athena_query(client, params):
    
    response = client.start_query_execution(
        QueryString=params["query"],
        QueryExecutionContext={
            'Database': params['database']
        },
        ResultConfiguration={
            'OutputLocation': 's3://' + params['bucket'] + '/' + params['path']
        }
    )
    return response

def athena_to_s3(session, params, max_execution = 10):
    client = session.client('athena', region_name=params["region"])
    execution = athena_query(client, params)
    execution_id = execution['QueryExecutionId']
    state = 'RUNNING'

    while (max_execution > 0 and state in ['RUNNING', 'QUEUED']):
        max_execution = max_execution - 1
        response = client.get_query_execution(QueryExecutionId = execution_id)

        if 'QueryExecution' in response and \
                'Status' in response['QueryExecution'] and \
                'State' in response['QueryExecution']['Status']:
            state = response['QueryExecution']['Status']['State']
            if state == 'FAILED':
                return False
            elif state == 'SUCCEEDED':
                s3_path = response['QueryExecution']['ResultConfiguration']['OutputLocation']
                filename = re.findall('.*\/(.*)', s3_path)[0]
                return filename
            elif state == 'RUNNING':
                time.sleep(5)
        time.sleep(1)
    
    return False

def cleanup_s3(session, params):
    s3 = session.resource('s3')
    my_bucket = s3.Bucket(params['bucket'])
    for item in my_bucket.objects.filter(Prefix=params['path']):
        item.delete()

def read_from_s3(params):

    session = boto3.Session() 
    # Requires authentication to be preset. How: https://glovoapp.atlassian.net/wiki/spaces/TECH/pages/920158474/Access+to+AWS+accounts
    
    # Query Athena and get the s3 filename as a result
    s3_filename = athena_to_s3(session, params)

    if s3_filename:

        # From filename to df
        client = session.client('s3')
        obj = client.get_object(Bucket= params['bucket'], Key= params['path'] + s3_filename) 
        try:
            df = pd.read_csv(obj['Body'])

            #Deletes generated s3 files
            cleanup_s3(session, params)

            return df
        except Exception:
            print("No files found")
    
    return False
