import tensorflow as tf 
import numpy as np 
import argparse
from Ranking.model import DRMM
from Ranking.model import PACRR
import Ranking.utils as utils
import Ranking.sampler as sampler 
import sys
import os 


path_to_data = './data/'
parser = argparse.ArgumentParser(description="Grid Ranking")
#Core params
parser.add_argument('--embsize', type=int, default= 128, help='dimension of item embedding (default: 100)')
parser.add_argument("--bins", type=int, default = 100, help='number of bins for vec similarity to mapping in DRMM')
parser.add_argument("--kmax", type=int, default= 10, help='number of maxk similarity to mapping in PACRR,should be smaller than seq_len')
parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for Adam (default: 0.001)')
parser.add_argument('--batch_size', type=int, default= 25, help='batch size (default: 128)')
parser.add_argument('--seq_len', type=int, default= 50, help='max sequence length (default: 20)')
parser.add_argument('--qitems', type=int, default=3, help='max num query items (default: 5)')
parser.add_argument('--list_size', type=int, default= 5, help='number of docs for each query (default: 2)')
parser.add_argument('--model', type=str, default='DRMM', help='model to use')

#Path
parser.add_argument("--emb", type=str,default= path_to_data +'emb/', help="grid embedding path")
parser.add_argument("--data",type = str,default= path_to_data + 'dataset/', help='data set name')
parser.add_argument("--dict",type = str,default = path_to_data + 'dict/', help = 'dict path')
parser.add_argument("--emb_path",type = str,default = path_to_data + 'emb/graphsage_seq_128/', help = 'embedding path')
parser.add_argument("--pos_sample_path",type = str,default = path_to_data + 'positive_sample.csv', help = 'positive sample path')

#Training Setup
parser.add_argument('--clip', type=float, default=1., help='gradient clip (default: 1.)')
parser.add_argument('--worker', type=int, default= 1, help='number of sampling workers (default: 10)')
parser.add_argument('--log_interval', type=int, default=2e1, help='log interval (default: 1e2)')
parser.add_argument('--eval_interval', type=int, default=1e2, help='eval/test interval (default: 1e3)')
parser.add_argument('--l2_reg', type=float, default=0.0, help='regularization scale (default: 0.0)')


args = parser.parse_args()
train_data, val_data, test_data = utils.data_generator(args)
#train_data, val_data, test_data = utils.posneg_data_generator(args)
user_grid, user_dict, n_items = utils.user_item_generator(args)
user_grid_pad = utils.padding_seq(user_grid, max_len = args.seq_len)

train_sampler = sampler.Sampler(
    data = train_data,
    user_grid = user_grid_pad,
    n_items = n_items,
    max_size = int(2e3),
    batch_size = args.batch_size,
    n_qitems = args.qitems,
    neg_size= args.list_size-1,
    n_workers=args.worker,
    neg_method='rand'
)

val_sampler = sampler.Sampler(
    data = val_data,
    user_grid = user_grid_pad,
    n_items = n_items,
    max_size = int(2e2),
    batch_size = args.batch_size,
    n_qitems = args.qitems,
    neg_size= 100-1,
    n_workers=args.worker,
    neg_method='rand'
)

checkpoint_dir = '_'.join(['./model/'+ args.model ,str(args.embsize), str(args.list_size), str(args.qitems)])
if args.model =='DRMM':
    models = DRMM(args, args.qitems, args.bins, n_items)
elif args.model == 'PACRR':
    models = PACRR(args, args.qitems, args.bins, n_items,top_k= args.kmax)
else:
    raise Exception('Error: Model name unrecognized')

lr = args.lr

def _hit_at_k(pred_ids, target_id, topk):
    hit = 0.0
    
    if target_id in set(pred_ids[:topk]):
        hit = 1.0
    rank = np.where(pred_ids ==target_id)[0][0]

    return hit, rank


def evaluate(score):
    top_k = 1
    total_hit_k = 0.0
    total_rank = 0.0
    
    
    for j in range(args.batch_size):
        doc_score = score[j,:]
        pred_ids = np.argsort(-doc_score, axis=-1) #Descend
        hit, rank = _hit_at_k(pred_ids, 0, top_k)
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
    init = tf.global_variables_initializer()
    sess.run(init)

    merged = tf.summary.merge_all()
    if not os.path.exists('./model'):
        os.mkdir('./model')
    train_writer = tf.summary.FileWriter(checkpoint_dir + '/train')
    test_writer = tf.summary.FileWriter(checkpoint_dir + '/test')
    train_writer.add_graph(sess.graph)

    train_loss_l = 0.
    step_count = 0

    global lr

    all_val_hit = [-1]
    early_stop_cn = 0


    print('Start training...')
    try:
        while True:
            cur_batch = train_sampler.next_batch()
            feed_dict = {models.doc: cur_batch[1], models.query: cur_batch[0], models.label: cur_batch[2] , models.lr: lr}
            train_summary ,_, train_loss = sess.run([merged, models.train_op, models.total_loss], feed_dict= feed_dict)
            train_loss_l += train_loss
            step_count += 1
            train_writer.add_summary( train_summary)

            if step_count % args.log_interval == 0:
                cur_loss = train_loss_l / args.log_interval
                acc, rec, f1_score = sess.run([models.accuracy, models.recall, models.f1_score], feed_dict= feed_dict)
                print('| Totol step {:10d} | loss {:5.3f}'.format(step_count, cur_loss))
                print('| Train accuracy {:2.3f} | recall {:2.3f} | f1_score {:2.3f}'.format(acc, rec, f1_score))
                sys.stdout.flush()
                train_loss_l = 0.
            
            if step_count % args.eval_interval == 0:
                val_batch = val_sampler.next_batch()
                feed_d = {models.doc: val_batch[1], models.query: val_batch[0], models.label: val_batch[2]}
                test_summary, acc, rec, f1_score = sess.run([merged, models.accuracy, models.recall, models.f1_score], feed_dict= feed_d)
                test_writer.add_summary( test_summary)
                

                print("*****************************************")
                print('| Totol step {:10d} '.format(step_count))
                print('| Val accuracy {:2.3f} | recall {:2.3f} | f1_score {:2.3f}'.format(acc, rec, f1_score))
                sys.stdout.flush()

                score = sess.run(models.score, feed_dict=feed_d)
                val_hit, val_rank = evaluate(score)

                all_val_hit.append(val_hit)
                print('-' * 90)
                print('| End of step {:10d} | valid hit@1 {:8.5f} | valid mean rank {:8.5f}'.format(
                        step_count, val_hit, val_rank))
                print('=' * 90)
                sys.stdout.flush()

                if all_val_hit[-1] <= all_val_hit[-2]:
                    lr /= 2.
                    lr = max(lr, 1e-6)
                    early_stop_cn += 1
                else:
                    early_stop_cn = 0
                    models.saver.save(sess, checkpoint_dir + '/model.ckpt')
                if step_count == 1e5:
                    print('Validation hit decreases in three consecutive epochs. Stop Training!')
                    sys.stdout.flush()
                    break
                print("**********************************************************")
            
    except Exception as e:
        print(str(e))
        train_sampler.close()
        exit(1)
    train_sampler.close()
    print('Done')

if __name__ == "__main__":
    main()
    