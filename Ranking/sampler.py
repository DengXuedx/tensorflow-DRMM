import numpy as np
from queue import Queue
import threading
import random
import Ranking.utils

def random_neg(pos, user_grid, s):
    '''
    pos: positive one
    user_grid: number of items
    s: size of samples.
    '''
    candi_list = list(set(user_grid) - set(pos))
    
    return random.sample(candi_list,s)

def sample_function(data, user_grid, n_items, batch_size, n_qitems, neg_size, result_queue, SEED, neg_method='rand'):
    '''
    data: DataFrame of train data, user_id: user, geohash: a set of all query items.
    user_grid: a list of all user's visit grid
    batch_size: number of samples in a batch.
    neg_size: number of negative samples.
    '''
    user_num = len(user_grid)
    def sample():
        query_loc = np.random.choice(a=range(0,data.shape[0]))
        q_doc = data.iloc[query_loc]
        query_str = q_doc['query'].split(',')
        pos_u = q_doc['user_id']
        # sample a slice from queries randomly
        neg_u = list(np.random.randint(0,user_num,neg_size))
        while pos_u in neg_u:
            temp = random.randint(0,user_num)
            neg_u.remove(pos_u)
            neg_u.append(temp) 
        doc_u = [pos_u] + neg_u

        query = []
        for q in query_str:
            if q !='nan':
                query.append(np.int32(float(q))) 
        while len(query ) < n_qitems:
            query +=[n_items]  #padding query 
        if len(query) > n_qitems:
            query = random.sample(query,n_qitems)

        docs=[]
        for u in doc_u:
            docs.append(user_grid[u])
        label = [1] + neg_size*[-1]
    
        return (query, docs, label)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(list(zip(*one_batch)))


class Sampler(object):
    def __init__(self, data, user_grid, n_items,max_size= int(2e5), batch_size=128, n_qitems=5, neg_size=10, n_workers=10, neg_method='rand'):
        self.result_queue = Queue(maxsize=int(max_size))
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                threading.Thread(target=sample_function, args=(data,
                                                    user_grid,
                                                    n_items,
                                                    batch_size,  
                                                    n_qitems,
                                                    neg_size, 
                                                    self.result_queue, 
                                                    np.random.randint(2e9),
                                                    neg_method)))
 

        self.processors[-1].daemon = True
        self.processors[-1].start()
    
    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.join()
