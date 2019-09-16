import tensorflow as tf 
import numpy as np 

class DRMM(object):
    def __init__(self, args, n_qitems, n_bins, n_items):
        self.args = args
        self.n_qitems = n_qitems
        self.seq_len = self.args.seq_len
        self.n_bins = n_bins
        self.batch_size = args.batch_size
        self.emb_size = args.embsize
        self.n_items = n_items
        self._build()
        self.saver = tf.train.Saver()
    
    def _build(self):
        with tf.name_scope("Input"):
            self.doc = tf.placeholder(tf.int32, shape= (None, None, self.seq_len), name= "docs")
            self.query  = tf.placeholder(tf.int32, shape=(None, self.n_qitems), name= "query")
            self.label = tf.placeholder(tf.int32, shape=(None, None), name="labels")
            self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
            self.dropout = tf.placeholder_with_default(0., shape=())
            self.item_embedding = tf.reshape(tf.convert_to_tensor(np.load(self.args.emb + 'val.npy'),dtype=tf.float32),[-1,self.emb_size])
            self.idf_table = tf.convert_to_tensor(np.load(self.args.emb + 'idf.npy'), dtype= tf.float32)

        self.net = DeepNet(4,64,[128,64,32,1])
        
        with tf.name_scope("Embedding_Layer"):
            doc_item = tf.nn.embedding_lookup(self.item_embedding, self.doc) #(-1,n_docs,seq_len,emb_size)
        
            query_item = tf.nn.embedding_lookup(self.item_embedding, self.query) #(,n_qitems,emb_size)
            query_idf = tf.nn.embedding_lookup(self.idf_table, self.query) #(-1,n_qitems)

        q_gate = self._GatingNetwork(query_idf) #(-1,n_qitems)
        score = []
        doc_vec = tf.reshape(doc_item, [self.batch_size, -1, self.emb_size])
        for j in range(self.n_qitems):
            with tf.name_scope("Local_Interaction"):
                q_vec = query_item[:,j,:]
                similarity = tf.squeeze(self._cosine(doc_vec, q_vec),-1)#(batch,n_doc*seq_len)
                rs_similar = tf.reshape(similarity, [-1, self.seq_len])#(batch*n_doc, seq_len)

            with tf.name_scope("Matching_Histogram_Mapping"):
                hist = tf.map_fn(lambda x: tf.cast( tf.histogram_fixed_width(values= x,value_range=[1e-3,1],nbins=self.n_bins), tf.float32), rs_similar,dtype=tf.float32)#梯度消失
                #hist = tf.nn.top_k(rs_similar,self.seq_len).values

            with tf.name_scope("Deep_Matching_Network"):
                query_out = tf.squeeze(self.net(hist) , -1)#(batch*n_doc,1)
                query_out = tf.reshape(query_out, [self.batch_size, -1]) #(-1,n_doc)

            with tf.name_scope("Score_Aggregation"):
                s_i = tf.multiply(query_out , tf.expand_dims(q_gate[:, j], -1)) #(-1,n_doc)
                score.append(s_i)
        
        with tf.name_scope("Matching_Score"):
            score = tf.reshape(score,[self.n_qitems, self.batch_size, -1])
            self.score = tf.reduce_sum(score,axis=0) #(batch_size,n_doc)
        
        with tf.name_scope("Pairwise_Ranking_Loss"):
            pos = tf.cast(tf.equal(1,self.label), tf.float32)
            neg = tf.cast(tf.equal(-1,self.label), tf.float32)
            self.l_loss = tf.reduce_mean(tf.multiply(neg , self.score),-1)-tf.reduce_mean(tf.multiply(pos , self.score),-1)
            #self.l_loss = tf.multiply(neg , self.score) - tf.multiply(pos , self.score)
            
            loss = tf.nn.relu(1+self.l_loss)

            self.total_loss = tf.reduce_sum(loss)
            tf.summary.scalar("Loss", self.total_loss)

        with tf.name_scope("Correct_Predtion"):
            self.prediction = tf.sign(self.score,name="predicte_label")
            self.accuracy, self.recall, self.precision, self.f1_score = self._confusion_metrics(self.prediction, self.label)
            tf.summary.scalar("Accuracy", self.accuracy)
            tf.summary.scalar("Recall", self.recall)
            tf.summary.scalar("Precision", self.precision)
            tf.summary.scalar("F1_Score", self.f1_score)
            
        with tf.name_scope("Train"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.total_loss)
            capped_gvs = [(tf.clip_by_value(grad, -self.args.clip, self.args.clip), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)        
      



    def _cosine(self, doc, query):
        
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(doc), axis = -1))#(-1,n_doc*seq_len)
        x2_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(query), axis = -1)),-1) #(-1,1)
    
        x2_reshape = tf.expand_dims(query,-1)#(-1,256,1)
    
        x1_x2_norm = tf.expand_dims(tf.multiply( x1_norm,x2_norm),-1) #(-1,seq_len,1)       

        return tf.div( tf.matmul(doc, x2_reshape), x1_x2_norm+1e-8)

    def _GatingNetwork(self,idf):
        with tf.name_scope("Term_Gating_Network"):
            w = tf.Variable(tf.random_normal(shape=[self.n_qitems]),name='gating_weight')
            vec = w * tf.squeeze(idf, -1)
            tf.summary.histogram("gating_weight", w)
        return tf.nn.softmax(vec)
    
    
    def _confusion_metrics(self, predict, real):
        predictions = tf.greater(predict,0)
        actuals = tf.greater(real,0)
        differ = tf.logical_xor(actuals, predictions)
        same = tf.logical_not(differ)
        
        tp = tf.reduce_sum( tf.cast(tf.logical_and(same, actuals), tf.float32) )
        tn = tf.reduce_sum( tf.cast(tf.logical_and(same, tf.logical_not(actuals) ), tf.float32) )

        fp = tf.reduce_sum( tf.cast(tf.logical_and(differ, predictions), tf.float32) )
        fn = tf.reduce_sum( tf.cast(tf.logical_and(differ, tf.logical_not(predictions) ), tf.float32) )
        
        tpr = tp /(tp + fn)
        fpr = fp/(fp + tn)
        fnr = fn/(tp + fn)
    
        accuracy = (tp + tn)/(tp + fp + fn + tn)
    
        recall = tpr
        precision = tp/(tp + fp)
    
        f1_score = (2 * (precision * recall)) / (precision + recall +1e-8)

        return accuracy, recall, precision, f1_score


    
    
  



class DeepNet(object):
    def __init__(self, layers=1, hidden_units=100, _HIDDEN_LAYER_DIMS = None,hidden_activation="tanh", dropout=0.2):
        if hidden_activation == "tanh":
            self.hidden_activation = tf.nn.tanh
        elif hidden_activation == "relu":
            self.hidden_activation = tf.nn.relu
        else: 
            raise NotImplementedError
        self.dropout = dropout
        if _HIDDEN_LAYER_DIMS:
            self._HIDDEN_LAYER_DIMS = _HIDDEN_LAYER_DIMS
        else:
            self._HIDDEN_LAYER_DIMS = [hidden_units] * layers

    def __call__(self, inputs):
        '''
        inputs: the similarity of each docs. (batch_size, num_docs, num_bins)
     
        '''
        cur_layer = tf.layers.batch_normalization(inputs, name="batch_normalize", reuse= tf.AUTO_REUSE)
        for i, layer_width in enumerate(int(d) for d in self._HIDDEN_LAYER_DIMS):
            with tf.name_scope("Layer%d"%i):
                cur_layer = tf.layers.dense(cur_layer, units=layer_width, name= "dense_h%d"%i, reuse= tf.AUTO_REUSE)
                tf.summary.histogram("dense_layer", cur_layer)
                cur_layer = tf.layers.batch_normalization(cur_layer, name= "normalize_h%d"%i, reuse= tf.AUTO_REUSE)
                tf.summary.histogram("batch_normalize", cur_layer)
                cur_layer = self.hidden_activation(cur_layer, name="activation")
                tf.summary.histogram("activation", cur_layer)
                cur_layer = tf.layers.dropout(
                    inputs=cur_layer, rate=self.dropout, name="dropout")
        
        return cur_layer




