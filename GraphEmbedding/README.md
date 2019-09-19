## GraphSAGE
除了使用aoi_name先进行word_embedding外，由于grid只有geohash没有名称，我们希望能够利用grid中的POI/职住人数/用户标签等feature为地点建立embedding模型。  
传统的DeepWalk(RandomWalk+SkipGram)的Graph Embedding不再适用，我们需要将节点信息加入Embedding中。  
仍然以AOI数据为例
### 数据准备
1. graphsage/utils.py
* load_data()  
需要的数据为图结构和节点feature.npy，以及节点到节点id的dict。  
我们以aoihash建立的graph为基准，不使用word_embedding而是aoi的poi_num_feature作为节点特征。  
Graph: ./data/graph/user_aoihash.gpickle  
Feature: ./data/node_features/aoi_poi_num_feature.npy
ID_Map: ./data/dict/aoi2id.json

* 另一组数据  
Graph: ./data/graph1.0/user_aoihash.gpickle   
Feature: ./data/node_features/aoi_word_emb_feature.npy
ID_Map: ./data/dict/aoi2id.json

2. graphsage/unsupervised_train.py
* MODEL: graphsage_mean  
包含Node Feature,算法详情见Paper  
《Inductive Representation Learning on Large Graphs》  
OUTPUT: ./unsup-/graphsage_mean_poi/  
./unsup-/graphsage_mean_aoi_wordemb/


* MODEL: n2v  
不包含Node Feature，为Node2Vec的tensorflow实现  
测试数据：  
Graph: ./data/graph/words_graph.gpickle  
ID_Map: ./data/dict/word2id.json  
OUTPUT: ./unsup-/n2v_word_emb/

4. make_embeddings.py
* get_tf_embeddings()  
从./unsup-/中读入模型输出的embeddings文件，主要包含  
val.npy：embedding的数据矩阵  
val.txt: 矩阵每一行对应的索引  
将二者合并为DataFrame后可以调用plot_aoi_emdbedding()  
OUTPUT: ./data/embeddings/aoi_poi_graphse_embeddings.csv,aoi_wordemb_graphse_embeddings.csv,word_n2v_embeddings
