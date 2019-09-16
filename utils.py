import pandas as pd 
import numpy as np 
import json
import argparse
import os
from sklearn.utils import shuffle 
import random


parser = argparse.ArgumentParser(description="Grid Ranking")
parser.add_argument("--data",type = str,default='../data/dataset/', help='data set name')
parser.add_argument("--dict",type = str,default = '../data/dict/', help = 'dict path')
parser.add_argument("--emb", type=str,default='../data/emb/', help="grid embedding path")



def data_generator(args):
    if not os.path.exists( args.data + 'train_data.csv'):
        dataset = pd.read_csv('./data/positive_smaple.csv').dropna()
        item2id = pd.read_csv(args.dict + 'geohash_dic.csv', index_col='geohash')
        UNK = len(item2id)
        with open(args.dict + 'geohash2id.json','r') as fr:
            item_dict = json.load(fr)
        with open(args.dict + 'user2id.json','r') as fr:
            user_dict = json.load(fr)

        query = []
        q_id = 0
        for data in dataset.itertuples():
            try: 
                u_id = user_dict[data[1]]
                geohash = list(data[2].split(','))
                try:
                    grid_ids = list(item2id['id'].loc[geohash].values.astype(str))
                except:
                    grid_ids = []
                    for q in geohash:
                        try:
                            grid_ids.append(str(item_dict[q]))
                        except:
                            grid_ids.append(str(UNK))
                q_data = [q_id, u_id, ",".join(grid_ids)]
                q_id +=1
                query.append(q_data)
            except:
                pass 
        query = pd.DataFrame(query, columns=['q_id', 'user_id', 'query'])

        num_query = query.shape[0]
        test_size = np.int(num_query/4)
        valid_size = np.int((num_query - test_size)/4)
        train_size = num_query - valid_size - test_size

        dataset = shuffle(query)

        train_data = dataset.iloc[:train_size]
        valid_data = dataset.iloc[train_size: train_size+valid_size ]
        test_data = dataset.iloc[train_size+valid_size: ]

        train_data.to_csv(args.data + 'train_data.csv' ,index=False)
        valid_data.to_csv( args.data + 'valid_data.csv' ,index=False)
        test_data.to_csv( args.data + 'test_data.csv' ,index=False)


    train_data = pd.read_csv( args.data + 'train_data.csv') 
    valid_data = pd.read_csv(args.data + 'valid_data.csv') 
    test_data = pd.read_csv( args.data + 'test_data.csv') 

    return train_data, valid_data, test_data

def user_item_generator(args):
    item2id = pd.read_csv(args.dict + 'geohash_dic.csv', index_col='geohash')
    UNK = len(item2id)
    
    if not os.path.exists( args.data + 'user_item_id.json'):
        #user_track = pd.read_csv( '../data/lg_user_place.csv',header=None,usecols =[0,1],names=['user_id','geohash'])

        with open(args.dict + 'geohash2id.json','r') as fr:
            item_dict = json.load(fr)
        with open(args.dict + 'user2id.json','r') as fr:
            user_dict = json.load(fr)
        with open(args.data + 'user_item.json') as fr:
            user_grid = json.load(fr)

        #user_grid = user_track.groupby('user_id')['geohash'].apply(list).to_dict()
        user_grid_ids = {}
        for key in user_grid.keys(): 
            geohash = list(user_grid[key])
            try:
                user_grid_ids[ user_dict[key] ] = list(item2id['id'].loc[geohash].values.astype(str))
            except:
                grid_ids = []
                for q in geohash:
                    try:
                        grid_ids.append(str(item_dict[q]))
                    except:
                        grid_ids.append(str(UNK))
                user_grid_ids[ user_dict[key] ] = grid_ids

        
        with open( args.data + 'user_item_id.json','w') as fr:
            fr.write(json.dumps(user_grid_ids))

    with open( args.data + 'user_item_id.json','r') as fr:
            user_grid = json.load(fr)
    with open(args.dict + 'user2id.json','r') as fr:
            user_dict = json.load(fr)
    

    return user_grid, user_dict, UNK

def padding_seq(user_grid, max_len):
    docs = []
    u_num = len(user_grid)
    for u in range(u_num):
        data = user_grid[str(u)]
        if len(data) <= max_len:
                idx = 0
        else:
                idx = np.random.randint(0, len(data) - max_len +1 )

        seq = np.zeros([max_len], dtype=np.int32)

        for i, itemid in enumerate(data[idx:idx+max_len]):
            seq[i] = int(itemid)
        docs.append(seq)
  
    return docs

    

if __name__ == "__main__":
    args = parser.parse_args()
    train_data, val_data, test_data = data_generator(args)
    user_grid, user_dict, n_items = user_item_generator(args)
    user_grid_pad = padding_seq(user_grid, max_len = 25)
        
