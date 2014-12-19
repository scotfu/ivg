import datetime
import json
import bson
import csv
import math
from bson.objectid import ObjectId

import numpy as np

from pymongo import MongoClient
from flask import g
from . import app

MAX = 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def connect_db():

    """Connects to the specific database."""
    client = MongoClient('mongodb://localhost:27017/')
    db = client['test-database']
    
    return db


def get_db():

    """Opens a new database connection if there is none yet for the

    current application context.

    """

    if not hasattr(g, 'mongodb'):
        g.mongodb = connect_db()
    return g.mongodb


#@app.teardown_appcontext
def close_db(error):

    """Closes the database again at the end of the request."""

    if hasattr(g, 'mongodb'):
        g.mongodb.close()

def get_collections():
    db = get_db()
    cases = list(db['fsc_case'].find())
    case_names = [case.get('name') for case in cases]
    return case_names

def get_case_info(name):
    db = get_db()
    collection = db['fsc_case']
    case = collection.find({'name':name})[0]
    print case.get('url'), case.get('content')
    return case.get('url'), case.get('content')

    
def csv_handler(file_name,collection_name, url, content):
    data_set = []
    db = get_db()
    collection = db[collection_name]
    case_collection = db['fsc_case']
    numeric_headers = []
    with open(file_name) as file_handler:
        reader = csv.reader(file_handler)
        header = reader.next()
        n = len(header)
        count = 1
        for row in reader:
            data = {'coordinate':[]}
            for column in range(n):
                try:
                    row[column] = float(row[column])
                    data['coordinate'].append(row[column])
                    if count == 1:
                        numeric_headers.append(header[column])
                except ValueError:
                    data[header[column]] = row[column]
            data_set.append(data['coordinate'])        
            try:
                data['_id'] = count
                collection.insert(data)
            except Exception,e:
                print e
            if count >= MAX:
                break
            count += 1
                
#   run_mds(file_name)           


    from sklearn import manifold
    from sklearn.metrics import euclidean_distances
    from sklearn.decomposition import PCA
    #n_samples = 20
    seed = np.random.RandomState(seed=3)
    #X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
    #print X_true

    X_true = data_set[:MAX]
#X_true = X_true.reshape((n_samples, 2))
#print X_true
# Center the data
#X_true -= X_true.mean()

    similarities = euclidean_distances(X_true)

# Add noise to the similarities
#noise = np.random.rand(n_samples, n_samples)
#noise = noise + noise.T
#noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
#similarities += noise

    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=-1)
    pos = mds.fit(similarities).embedding_


    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=seed, n_jobs=-1,
                    n_init=1)
    npos = nmds.fit_transform(similarities, init=pos)
# Rescale the data
#pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
#npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())

# Rotate the data
    clf = PCA(n_components=2)
    X_true = clf.fit_transform(X_true)

    pos = clf.fit_transform(pos)

    npos = clf.fit_transform(npos)
    for n in range(len(npos)):
        collection.update({'_id' : n+1 },
                          {'$set':
                           {'mds' : list(pos[n]),
                            'nmds' : list(npos[n]),
                            'pca' : list(X_true[n]), 
                           }
                          })
    print url,content
    case_collection.insert({'name':collection_name, 'header':numeric_headers, 'url':url,
                            'content': content})


class CustomEncoder(json.JSONEncoder):
    """A C{json.JSONEncoder} subclass to encode documents that have fields of
    type C{bson.objectid.ObjectId}, C{datetime.datetime}
    """

    def default(self, obj):
        if isinstance(obj, bson.objectid.ObjectId):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
            
    
def case_query(collection_name, algorithm):
    db = get_db()
    collection = db[collection_name]

    all_points = list(collection.find({}, {'_id':1, algorithm:1}))
    all_points_only_coor =   [point.get(algorithm) for point in all_points]
    max_x = max([ point[0] for point in all_points_only_coor])
    max_y = max([ point[1] for point in all_points_only_coor])
    min_x = min([ point[0] for point in all_points_only_coor])
    min_y = min([ point[1] for point in all_points_only_coor])
    margin = [[max_x,max_y],[min_x,min_y]]

    return all_points, margin



def kmeans_query(collection_name, selected_points, algorithm):
    db = get_db()
    collection = db[collection_name]
    all_points = list(collection.find({}, {'_id':1, algorithm:1}))

    points =[point.get(algorithm) for point in all_points]
    selected_points = map(lambda x: int(x), selected_points)
    selected_points =[point.get(algorithm) for point in list(collection.find({'_id':{'$in':selected_points}}, {'_id':1, algorithm:1}))]
    cluster_matrix, centroids = KMeans(points, selected_points)
    all_points_only_coor =   [point.get(algorithm) for point in all_points]
    max_x = max([ point[0] for point in all_points_only_coor])
    max_y = max([ point[1] for point in all_points_only_coor])
    min_x = min([ point[0] for point in all_points_only_coor])
    min_y = min([ point[1] for point in all_points_only_coor])
    margin = [[max_x,max_y],[min_x,min_y]]

    return cluster_matrix, centroids, all_points, margin

    
def kmeans2_query(collection_name, selected_points, second_selected_points, algorithm):
    db = get_db()
    collection = db[collection_name]
    cluster_matrix, centroids, all_points, margin = kmeans_query(collection_name, selected_points, algorithm)
    #test which cluster it is
    one_point = int(second_selected_points[0]) -1 # pitfall -1 here
    temp_k = 0
    while True:
        if cluster_matrix[temp_k][one_point] == 1:
            break
        else:
            temp_k += 1

    all_points = [ all_points[i] for i in range(len(all_points))  if cluster_matrix[temp_k][i] == 1 ]

    second_selected_points = map(lambda x: int(x), second_selected_points)
    second_selected_points =[point.get(algorithm) for point in list(collection.find({'_id':{'$in':second_selected_points}}, {'_id':1, algorithm:1}))]
    all_points_only_coor =   [point.get(algorithm) for point in all_points]
    cluster_matrix, centroids = KMeans(all_points_only_coor, second_selected_points)

    return cluster_matrix, centroids, all_points, margin
    

def get_all_points(collection_name, algorithm, second_selected_points=None):
    db = get_db()
    collection = db[collection_name]
    return [point for point in list(collection.find({}, {'_id':1, algorithm:1}))]

    


def kmeans_2_query(collection_name):
    db = get_db()
    collection = db[collection_name]
    ec = CustomEncoder()
    return ec.encode(list(collection.find({}, {'coordinate' : 0 })))

def aggregation(collection_name, ids):
    db = get_db()
    collection = db[collection_name]
    ids = map(lambda x: int(x), ids)
    points = list(collection.find({'_id':{'$in':ids}}, {'coordinate':1}))
    n = float(len(points))
    dimension = len(points[0]['coordinate'])
    data = [0] * dimension
    mins = [float('inf')] * dimension
    maxs = [float('-inf')] * dimension
    for point in points:
        for i in range(dimension):
            data[i] += point['coordinate'][i]
    data = map(lambda x: x/n, data)
    header = db['fsc_case'].find({'name':collection_name})[0].get('header')
    all_points = list(collection.find({}, {'coordinate':1}))
    for point in all_points:
        for i in range(dimension):
            mins[i] = min(mins[i], point['coordinate'][i])
            maxs[i] = max(maxs[i], point['coordinate'][i])
    
    return data, header, mins, maxs
#kmeans part starts
    
#watch out: float flaw 
def squared_euclidean_distance(pointA, pointB):
    if len(pointA) != len(pointB):
        raise ValueError('the two points should have the same degree of dimension')

    dimension = len(pointA)
    distance = 0.0
    for i in range(dimension):
        distance += math.pow(pointA[i] - pointB[i], 2)

    return distance
    
    
def assignment(points, centroids):
    '''
    assgin points to k clusters, the first step of k-means iteration
    Randomly pick k points as centroids then assgin points to the nearest centroid
    '''
    k = len(centroids)
    num_of_points = len(points)
    cluster_matrix = [[0 for i in range(num_of_points)] for j in range(k)]

    for i in range(len(points)):
        distance = float('inf')
        assign_to = None
        for j in range(len(centroids)):
            e_distance = squared_euclidean_distance(points[i], centroids[j])
            if e_distance <= distance:
                distance = e_distance
                assign_to = j
        cluster_matrix[assign_to][i] = 1
        
    return cluster_matrix


def update_centroids(points, cluster_matrix):
    new_centroids = []
    dimension = len(points[0])
    for cluster in cluster_matrix:
        coordinate = [0 for i in range(dimension)]
        for position in range(len(cluster)):
            if cluster[position] == 1:
               coordinate = map(sum, zip(coordinate, points[position]))

        count = float(cluster.count(1))
        mean = map(lambda x:x/count, coordinate)
        new_centroids.append(mean)    
    return new_centroids

def KMeans(points,centroids):
    #import pprint
    #centroids = random.sample(points, k)
    cluster_matrix = assignment(points, centroids)
    #cluster_matrix
    centroids = update_centroids(points, cluster_matrix)
    return cluster_matrix,centroids
    #plot(points, cluster_matrix, centroids)


def histogram(collection_name,di=0):
    db = get_db()
    collection = db[collection_name]
    coors = collection.find({},{'coordinate':1})
    input = [ point.get('coordinate')[di] for point in list(coors) ]
    counts, bins = np.histogram(input,50)
    return list(counts), list(bins)
    
