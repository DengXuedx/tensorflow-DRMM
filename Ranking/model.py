import tensorflow as tf 
import numpy as np 


class Model(object):
    def __init__(self, args, n_qitems, n_bins, n_items):
        self.args = args
        self.n_qitems = n_qitems
        self.seq_len = self.args.seq_len
        self.n_bins = n_bins
        self.batch_size = args.batch_size
        self.emb_size = args.embsize
        self.n_items = n_items
        self.build()

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.name_scope("Input"):
            self.doc = tf.placeholder(tf.int32, shape= (None, None, self.seq_len), name= "docs")
            self.query  = tf.placeholder(tf.int32, shape=(None, self.n_qitems), name= "query")
            self.label = tf.placeholder(tf.int32, shape=(None, None), name="labels")
            self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
            self.dropout = tf.placeholder_with_default(0., shape=())
            self.item_embedding = tf.reshape(tf.convert_to_tensor(np.load(self.args.emb_path + 'val.npy'),dtype=tf.float32),[-1,self.emb_size])
            self.idf_table = tf.convert_to_tensor(np.load(self.args.emb + 'idf.npy'), dtype= tf.float32)
        
        with tf.name_scope("Embedding_Layer"):
            self.doc_item = tf.nn.embedding_lookup(self.item_embedding, self.doc) #(-1,n_docs,seq_len,emb_size)
        
            self.query_item = tf.nn.embedding_lookup(self.item_embedding, self.query) #(,n_qitems,emb_size)
            self.query_idf = tf.nn.embedding_lookup(self.idf_table, self.query) #(-1,n_qitems)

    def calcu_loss(self):
        with tf.name_scope("Pairwise_Ranking_Loss"):
            pos = tf.cast(tf.equal(1,self.label), tf.float32)
            neg = tf.cast(tf.equal(-1,self.label), tf.float32)
            #self.l_loss = tf.reduce_mean(tf.multiply(neg , self.score),-1)-tf.reduce_mean(tf.multiply(pos , self.score),-1)
            self.l_loss = tf.multiply(neg , self.score) - tf.multiply(pos , self.score)
            
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
    
    def train(self):
        with tf.name_scope("Train"):
            optimizer = tf.train.AdamOptimizer(self.lr)
            gvs = optimizer.compute_gradients(self.total_loss)
            capped_gvs = [(tf.clip_by_value(grad, -self.args.clip, self.args.clip), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)


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

class DRMM(Model):
    def __init__(self, args, n_qitems, n_bins, n_items):
        super(DRMM, self).__init__(args, n_qitems, n_bins, n_items)
        self._build()
        self.calcu_loss()
        self.train()
        self.saver = tf.train.Saver()
    
    def _build(self):
        self.net = DeepNet(4,64,[128,64,32,1])
        q_gate = self._GatingNetwork(self.query_idf) #(-1,n_qitems)
        score = []
        doc_vec = tf.reshape(self.doc_item, [self.batch_size, -1, self.emb_size])
        for j in range(self.n_qitems):
            with tf.name_scope("Local_Interaction"):
                q_vec = self.query_item[:,j,:]
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
                
      


    def _GatingNetwork(self,idf):
        with tf.name_scope("Term_Gating_Network"):
            w = tf.Variable(tf.random_normal(shape=[self.n_qitems]),name='gating_weight')
            vec = w * tf.squeeze(idf, -1)
            tf.summary.histogram("gating_weight", w)
        return tf.nn.softmax(vec)
    
    def _cosine(self, doc, query):
        
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(doc), axis = -1))#(-1,n_doc*seq_len)
        x2_norm = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(query), axis = -1)),-1) #(-1,1)
    
        x1_x2_norm = tf.expand_dims(tf.multiply( x1_norm,x2_norm),-1) #(-1,seq_len,1)    

        x2_reshape = tf.expand_dims(query,-1)#(-1,256,1)   

        return tf.div( tf.matmul(doc, x2_reshape), x1_x2_norm+1e-8)
    
    


class PACRR(Model):
    def __init__(self, args, n_qitems, n_bins, n_items, top_k):
        super(PACRR, self).__init__(args, n_qitems, n_bins, n_items)
        self._build(top_k)
        self.calcu_loss()
        self.train()
        self.saver = tf.train.Saver()
    
    def _build(self, _k):
        with tf.name_scope("Local_Interaction"):
            q_vec = tf.transpose(self.query_item,perm=[0,2,1]) #(batch,emb_szie,q_item)
            doc_vec = tf.reshape(self.doc_item, [self.batch_size, -1, self.emb_size]) #(batch,n_doc*seq_len,emb_size)
            similarity = self._cosine(doc_vec, q_vec)#(batch*n_doc,seq_len,q_item)
        
        with tf.name_scope("Convolution"):
            conv = tf.layers.conv2d(tf.expand_dims(similarity, -1), filters=4, kernel_size=2,padding='same')
            max_pool = tf.reduce_max(conv,-1)
        with tf.name_scope("MaxPooling"):
            q_encoding = tf.nn.top_k(tf.transpose(max_pool, [0,2,1]), _k).values
            q_encoding = tf.nn.l2_normalize(q_encoding, axis=-1)
            q_encoding = tf.reshape(q_encoding, [self.batch_size, -1, self.n_qitems, _k])
            q_encoding = tf.transpose(q_encoding, [0,2,1,3])
            q_encoding = tf.reshape(q_encoding,[self.batch_size, self.n_qitems, -1])#(batch,n_q,n_d*k)
        with tf.name_scope("Query_Term_IDF"):
            q_idf = tf.nn.softmax(tf.squeeze(self.query_idf, -1))#(batch,n_q)
            q_term = tf.multiply(q_encoding, tf.expand_dims(q_idf, -1)) #(batch,n_q,n_d*k)
            q_term = tf.reshape(q_term, [self.batch_size, self.n_qitems, -1, _k])#(batch,n_q,n_d,k)
          
        with tf.name_scope("Deep_Matching_Network"):
            self.net = DeepNet(4,64,[64,32,10,1])
            score = []
            for i in range(self.n_qitems):
                q_i_term = q_term[:,i,:,:]
                s_i =  tf.squeeze(self.net(q_i_term), -1)
                score.append(s_i)
            score = tf.reshape(score, [self.batch_size, -1, self.n_qitems])
            
            self.score = tf.squeeze(tf.layers.dense(score,units=1),-1)

           

    def _cosine(self, doc, query):
        
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(doc), axis = -1)) #(batch,n_doc*seq_len)
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(query), axis = -2)) #(batch, q_item)
        
        
        x1_x2_norm = []
        for i in range(self.n_qitems):
            q_i = tf.multiply(x1_norm,tf.expand_dims(x2_norm[:,i], -1))
            x1_x2_norm.append(q_i)

        x1_x2_norm = tf.reshape( x1_x2_norm,[self.batch_size, -1, self.n_qitems])
        
        cos = tf.div( tf.matmul(doc, query), x1_x2_norm+1e-8)

        return tf.reshape(cos, [-1, self.seq_len, self.n_qitems])

    

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




