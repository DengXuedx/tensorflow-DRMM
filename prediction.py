import tensorflow as tf 
import numpy as np 
import random
import argparse
from Ranking.model import DRMM
import Ranking.utils as utils
import Ranking.sampler as sampler 
import sys


path_to_data = './data/'
parser = argparse.ArgumentParser(description="Grid Ranking")
parser.add_argument("--emb", type=str,default= path_to_data +'emb/', help="grid embedding path")
parser.add_argument("--data",type = str,default= path_to_data + 'dataset/', help='data set name')
parser.add_argument("--dict",type = str,default = path_to_data + 'dict/', help = 'dict path')
parser.add_argument("--emb_path",type = str,default = path_to_data + 'emb/graphsage_seq_small_128/', help = 'embedding path')
parser.add_argument('--embsize', type=int, default= 128, help='dimension of item embedding (default: 100)')
parser.add_argument("--bins", type=int, default = 100, help='number of bins for vec similarity to mapping')
parser.add_argument('--clip', type=float, default=1., help='gradient clip (default: 1.)')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for Adam (default: 0.001)')
parser.add_argument('--batch_size', type=int, default= 25, help='batch size (default: 128)')
parser.add_argument('--seq_len', type=int, default= 50, help='max sequence length (default: 20)')
parser.add_argument('--qitems', type=int, default=3, help='max num query items (default: 5)')
parser.add_argument('--list_size', type=int, default= 5, help='number of docs for each query (default: 2)')
parser.add_argument('--worker', type=int, default= 1, help='number of sampling workers (default: 10)')
parser.add_argument('--log_interval', type=int, default=2e1, help='log interval (default: 1e2)')
parser.add_argument('--eval_interval', type=int, default=1e2, help='eval/test interval (default: 1e3)')
parser.add_argument('--l2_reg', type=float, default=0.0, help='regularization scale (default: 0.0)')


args = parser.parse_args()
_, _, test_data = utils.data_generator(args)
user_grid, user_dict, n_items = utils.user_item_generator(args)
user_grid_pad = utils.padding_seq(user_grid, max_len = args.seq_len)

checkpoint_dir = '_'.join(['./model/q_doc', str(args.list_size), str(args.qitems)])

models = DRMM(args, args.qitems, args.bins, n_items)



def sample_function(data, user_grid, n_items, batch_size, n_qitems, start_q, start_u, neg_size):
    '''
    data: DataFrame of train data, user_id: user, geohash: a set of all query items.
    user_grid: a list of all user's visit grid
    batch_size: number of samples in a batch.
    neg_size: number of negative samples.
    '''
    user_num = len(user_grid)
    def sample(query_loc):
        q_doc = data.iloc[query_loc]
        query_str = q_doc['query'].split(',')
        label = q_doc['user_id']
        
        doc_u = list(np.arange(start_u, start_u+neg_size))

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
    
        return (query, docs, label)

    one_batch = []
    for i in range(batch_size):
        one_batch.append(sample(start_q+i))
    return list(zip(*one_batch))
        
def _hit_at_k(pred_ids, target_id, topk):
    hit = 0.0
    
    if target_id in set(pred_ids[:topk]):
        hit = 1.0
    rank = np.where(pred_ids ==target_id)[0][0]

    return hit, rank


def evaluate(score,target):
    top_k = 1
    total_hit_k = 0.0
    total_rank = 0.0
    
    
    for j in range(args.batch_size):
        doc_score = score[j,:]
        pred_ids = np.argsort(-doc_score, axis=-1) #Descend
        hit, rank = _hit_at_k(pred_ids, target[j], top_k)
        total_hit_k += hit
        total_rank += rank
    
        #total_ndcg_k += ndcg 由于不知道真实排序，因此不计算NDCG值

    val_hit = total_hit_k / args.batch_size 
    val_rank = total_rank / args.batch_size
    
    return [val_hit, val_rank]



def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    new_saver = tf.train.import_meta_graph(checkpoint_dir+'/model.ckpt.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))


    user_num = len(user_grid_pad)
    batch_size = args.list_size * 50
    batch_num = int(user_num/batch_size)

    test_size = test_data.shape[0]
    test_num = int(test_size / args.batch_size)

    for j in range(test_num):
        score = np.zeros([args.batch_size,1])
        for i in range(batch_num):
                cur_batch = sample_function(data=test_data,
                user_grid= user_grid_pad, 
                n_items = n_items, 
                batch_size = args.batch_size, 
                n_qitems = args.qitems,
                start_q = j*args.batch_size, 
                start_u = i*batch_size, 
                neg_size =batch_size)
                feed_dict = {models.query: cur_batch[0], models.doc:cur_batch[1]}
                s = sess.run(models.score, feed_dict = feed_dict) 
                score = np.hstack((score,s))
       

            
        cur_batch = sample_function(test_data, user_grid_pad, n_items, args.batch_size, args.qitems,j*args.batch_size, batch_num*batch_size, user_num-batch_num*batch_size)
        feed_dict = {models.query: cur_batch[0], models.doc:cur_batch[1]}
        s = sess.run(models.score, feed_dict = feed_dict)
        score = np.hstack((score,s))
        score = np.delete(score,0,axis=1)
        val_hit, val_rank = evaluate(score, cur_batch[2])
        print('hit@1 {:8.5f} |mean rank {:8.5f}'.format(val_hit, val_rank))
        
        
           


if __name__ == "__main__":
    main()