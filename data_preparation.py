import pandas as pd
import networkx as nx  
import numpy as np 
import json
import time
import os 
import threading

def build_graph(loc, tim, col_name, batch_size, n_workers, cut_byday=True):
    """
    loc: locations of users have visit
    time: visit time
    col_name: col_name of locations,eg. "geohash","aoihash"
    batch_size: number of users each thread process
    n_workers: number od threads
    cut_byday: wether cut time by day when add edge

    """
    
    G = nx.Graph()
    user_item = {}
    user_list = list(loc.keys())

    def multi_build_graph(start_loc, end_loc):
        for value in user_list[start_loc:end_loc] :
            grid_num = len(value)
            df = pd.DataFrame([loc[value], tim[value]]).T
            df.columns=[col_name,'time']
            next_loc = pd.DataFrame(df.iloc[1:])
            next_loc = next_loc.append(next_loc.iloc[len(df)-2]).reset_index(drop=True)
            node_list = df[df[col_name]!=next_loc[col_name]]
            user_item[value] = list(node_list[col_name])
            node_list = list(node_list[col_name])
            loc_num = len(node_list)
            for i in range(loc_num-1):
                c_loc = node_list[i]
                n_loc = node_list[i+1]
                if cut_byday:
                    if(time.localtime(n_loc["time"]).tm_mday==time.localtime(c_loc["time"]).tm_mday):
                        if G.has_edge(c_loc[col_name],n_loc[col_name]):
                            weight = G.get_edge_data(c_loc[col_name],n_loc[col_name])['weight']+1
                        else:
                            weight = 1
                        G.add_edge(c_loc[col_name],n_loc[col_name],weight=weight)
                    else:
                        G.add_node(c_loc[col_name])
                else:
                    if G.has_edge(c_loc[col_name],n_loc[col_name]):
                            weight = G.get_edge_data(c_loc[col_name],n_loc[col_name])['weight']+1
                    else:
                        weight = 1
                    G.add_edge(c_loc[col_name],n_loc[col_name],weight=weight)

            G.add_node(node_list[col_name].iloc[loc_num-1])

        
        nx.write_gpickle(G,"./data/graph/grid_graph.gpickle")
        with open("./data/dataset/user_item.json",'w') as fr:
            fr.write(json.dumps(user_item))

    if not os.path.exists("./data/graph"):
        os.mkdir("./data/graph")
    if not os.path.exists("./data/dataset"):
        os.mkdir("./data/dataset")

    name = locals()
    for j in range(n_workers-1):
        name["p%d"%j] = threading.Thread(target=multi_build_graph,args=(batch_size*j,batch_size*(j+1)))
    name["p%d"%(n_workers-1)] = threading.Thread(target=multi_build_graph,args=(batch_size*(n_workers-1),len(user_list)))
    for j in range(n_workers):
        name["p%d"%j].start()
    for j in range(n_workers):
        name["p%d"%j].join()

    return G 

def make_itemdict(item_list, dict_path, index_col):
    """
    item_list: all the locations
    dict_path: path to save item2id dict
    index_col: col_name of location, eg "geohash","aoihash"
    """

    if not os.path.exists("./data/dict"):
        os.mkdir("./data/dict")

    dic = pd.DataFrame(item_list,columns=[index_col])
    item2id = {}
    for data in dic.itertuples():
        item2id[data[1]] = data[0]

    dic = dic.reset_index().set_index(index_col).rename(columns={'index':'id'})
    dic.to_csv(dict_path + '_dic.csv')

    with open(dict_path + '2id.json', 'w') as fr:
        fr.write( json.dumps(item2id) )
    return dic, item2id

def make_item_feature(item2id, feature_from, feature_to, index_col):
    feature = pd.read_csv(feature_from, index_col=index_col)
    item = pd.read_csv(item2id, index_col= index_col)

    item_feature = item.join(feature)
    na_index = item_feature[item_feature.isnull().T.any()].index
    item_feature['pad'] = item_feature.isnull().T.any().astype('int')
    item_feature = item_feature.drop(columns=['id']).fillna(0)
    item_feature = item.drop(columns=['id'])
    np.save(feature_to, item_feature)

def load_data(user_item_interaction):
    if not os.path.exists('./data/user_loc_interaction'):
        os.mkdir('./data/user_loc_interaction')
    if not os.path.exists('./data/user_loc_interaction/user_geohash.json'):
        tr =  user_item_interaction.set_index('geohash').groupby('user_id').apply(lambda x: x.sort_values(by='time',ascending=True)).drop(columns=['user_id']).reset_index()
        tr['time'] = tr['time']/1000
        geohash = tr.groupby('user_id')['geohash'].apply(list).to_dict()
        tim = tr.groupby('user_id')['time'].apply(list).to_dict()
        with open('./data/user_loc_interaction/user_geohash.json','w') as fr:
            fr.write(json.dumps(geohash))
        with open('./data/user_loc_interaction/user_time.json','w') as fr:
            fr.write(json.dumps(tim))

    with open('./data/user_loc_interaction/user_geohash.json','r') as fr:
        geohash = json.load(fr)
    with open('./data/user_loc_interaction/user_time.json','r') as fr:
        tim = json.load(fr)
    
    return geohash, tim

def query_idf(user_track, query_dict):
    unique_ui = user_track.drop_duplicates()
    N_docs = len(pd.unique(unique_ui['user_id']))
    query_n = unique_ui.groupby('geohash').size()
    Idf = pd.DataFrame(np.log2( N_docs/query_n), columns=['IDF'])
    Idf_table = query_dict.join(Idf).drop(columns=['id'])

    if os.path.exists('./data/emb'):
        os.mkdir('./data/emb')
    np.save('./data/emb/idf' , Idf_table.values)
    
    return Idf_table

if __name__ == "__main__":
    user_item_interaction = pd.read_csv("./data/user_loc_intercation",sep='\t',header=None,usecols =[0,1,4],names=['user_id','time','geohash'])
    
    geohash, tim = load_data(user_item_interaction)
    build_graph(geohash, tim, 'geohash', 10000,20, False)

    user_grid = user_item_interaction
    item_dict,_ = make_itemdict(pd.unique(user_grid['geohash']), './data/dict/geohash', 'geohash')
    _,_ = make_itemdict(pd.unique(user_grid['user_id']), './data/dict/user', 'user_id')
    
    make_item_feature('./data/dict/geohash_dic.csv', './data/node_features/geohash_features.csv','./data/node_features/geohash_features', 'geohash')
    
    idf = query_idf(user_grid[['user_id','geohash']], item_dict)
   