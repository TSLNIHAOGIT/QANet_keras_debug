## QANet in keras
QANet: https://arxiv.org/abs/1804.09541

This keras model refers to QANet in tensorflow (https://github.com/NLPLearn/QANet). 

> I find that the conv based multi-head attention in tensor2tensor (https://github.com/NLPLearn/QANet/blob/master/layers.py) performs 3%~4% better than the multiplying matrices based one in (https://github.com/bojone/attention/blob/master/attention_keras.py).

## Pipline
1. Download squad data `dev-v1.1.json` and `train-v1.1.json` from (https://rajpurkar.github.io/SQuAD-explorer/) to the folder `./original_data`.

2. Download `glove.840B.300d.txt` from (https://nlp.stanford.edu/projects/glove/) to the folder `./original_data`.

3. Run `python preprocess.py` to get the wordpiece based preprocessed data.

4. Run `python train_QANet.py` to start training.

## Updates
- [x] Add EMA (with about 3% improvement)
- [x] Add multi gpu (speed up)
- [x] Support adding handcraft features
- [x] Revised the MultiHeadAttention and PositionEmbedding in keras
- [x] Support parallel multi-gpu training and inference
- [x] Add layer dropout and revise the dropout bug (with about 2% improvement)
- [x] Update the experimental results and related hyper-parameters (in `train_QANet.py`)
- [x] Revise the output Layer `QAoutputBlock.py`(with about 1% improvement)
- [x] Replace the **BatchNormalization** with the **LayerNormalization** in `layer_norm.py`(about 0.5% improvement)
- [x] Add **slice operation** to QANet (double speed up)
- [x] Add Cove (about 1.4% improvement)
- [x] Implement the EMA in keras-gpu. (30% spped up)
- [x] Add WordPiece in keras (from BERT) (0.5% improvement)
- [ ] Add data augmentation

~~I find that EMA in keras is hard to implement with GPU, and the training speed is greatly affected by it in keras. Besides, it's hard to add the slice op in keras too, so the training speed is further slower(cost about twice as much time compared with the optimized tensorflow version...).~~

Now, the gpu-version EMA can work perporly in keras.

## Results
All models are set in 8 heads, 128 filters.

| setting | epoch | EM/F1 |
| ------ | ------ | ------ |
| batch_size=24 | 11 | 66.24% / 76.75% |
| batch_size=24 + ema_decay=0.9999 | 14 | 69.51% / 79.13% |
| batch_size=24 + ema_decay=0.9999 + wordpiece | 17 | 70.07% / 79.52% |
| batch_size=24 + ema_decay=0.9999 + wordpiece + Cove | 13 | 71.48% / 80.85% |


Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, None)]       0                                            
__________________________________________________________________________________________________
lambda (Lambda)                 (None, None)         0           input_1[0][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, None)         0           input_2[0][0]                    
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 1)            0           lambda[0][0]                     
__________________________________________________________________________________________________
input_3 (InputLayer)            [(None, None, 16)]   0                                            
__________________________________________________________________________________________________
lambda_3 (Lambda)               (None, 1)            0           lambda_1[0][0]                   
__________________________________________________________________________________________________
input_4 (InputLayer)            [(None, None, 16)]   0                                            
__________________________________________________________________________________________________
batch_slice_2 (BatchSlice)      (None, None, None)   0           input_3[0][0]                    
                                                                 lambda_2[0][0]                   
__________________________________________________________________________________________________
batch_slice_3 (BatchSlice)      (None, None, None)   0           input_4[0][0]                    
                                                                 lambda_3[0][0]                   
__________________________________________________________________________________________________
char_embedding (Embedding)      (None, None, None, 6 78912       batch_slice_2[0][0]              
                                                                 batch_slice_3[0][0]              
__________________________________________________________________________________________________
lambda_4 (Lambda)               (None, 16, 64)       0           char_embedding[0][0]             
__________________________________________________________________________________________________
lambda_5 (Lambda)               (None, 16, 64)       0           char_embedding[1][0]             
__________________________________________________________________________________________________
char_conv (Conv1D)              (None, 12, 128)      41088       lambda_4[0][0]                   
                                                                 lambda_5[0][0]                   
__________________________________________________________________________________________________
batch_slice (BatchSlice)        (None, None)         0           input_1[0][0]                    
                                                                 lambda_2[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d (GlobalMax (None, 128)          0           char_conv[0][0]                  
__________________________________________________________________________________________________
batch_slice_1 (BatchSlice)      (None, None)         0           input_2[0][0]                    
                                                                 lambda_3[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 128)          0           char_conv[1][0]                  
__________________________________________________________________________________________________
word_embedding (Embedding)      (None, None, 300)    3000000     batch_slice[0][0]                
                                                                 batch_slice_1[0][0]              
__________________________________________________________________________________________________
lambda_6 (Lambda)               (None, None, 128)    0           global_max_pooling1d[0][0]       
__________________________________________________________________________________________________
lambda_7 (Lambda)               (None, None, 128)    0           global_max_pooling1d_1[0][0]     
__________________________________________________________________________________________________
concatenate (Concatenate)       (None, None, 428)    0           word_embedding[0][0]             
                                                                 lambda_6[0][0]                   
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, None, 428)    0           word_embedding[1][0]             
                                                                 lambda_7[0][0]                   
__________________________________________________________________________________________________
highway_input_projection (Conv1 (None, None, 128)    54912       concatenate[0][0]                
                                                                 concatenate_1[0][0]              
__________________________________________________________________________________________________
highway0_linear (Conv1D)        (None, None, 128)    16512       highway_input_projection[0][0]   
                                                                 highway_input_projection[1][0]   
__________________________________________________________________________________________________
dropout (Dropout)               (None, None, 128)    0           highway0_linear[0][0]            
__________________________________________________________________________________________________
highway0_gate (Conv1D)          (None, None, 128)    16512       highway_input_projection[0][0]   
                                                                 highway_input_projection[1][0]   
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, None, 128)    0           highway0_linear[1][0]            
__________________________________________________________________________________________________
lambda_8 (Lambda)               (None, None, 128)    0           dropout[0][0]                    
                                                                 highway0_gate[0][0]              
                                                                 highway_input_projection[0][0]   
__________________________________________________________________________________________________
lambda_10 (Lambda)              (None, None, 128)    0           dropout_2[0][0]                  
                                                                 highway0_gate[1][0]              
                                                                 highway_input_projection[1][0]   
__________________________________________________________________________________________________
highway1_linear (Conv1D)        (None, None, 128)    16512       lambda_8[0][0]                   
                                                                 lambda_10[0][0]                  
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, None, 128)    0           highway1_linear[0][0]            
__________________________________________________________________________________________________
highway1_gate (Conv1D)          (None, None, 128)    16512       lambda_8[0][0]                   
                                                                 lambda_10[0][0]                  
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, None, 128)    0           highway1_linear[1][0]            
__________________________________________________________________________________________________
lambda_9 (Lambda)               (None, None, 128)    0           dropout_1[0][0]                  
                                                                 highway1_gate[0][0]              
                                                                 lambda_8[0][0]                   
__________________________________________________________________________________________________
lambda_11 (Lambda)              (None, None, 128)    0           dropout_3[0][0]                  
                                                                 highway1_gate[1][0]              
                                                                 lambda_10[0][0]                  
__________________________________________________________________________________________________
position__embedding (Position_E (None, None, 128)    0           lambda_9[0][0]                   
__________________________________________________________________________________________________
position__embedding_1 (Position (None, None, 128)    0           lambda_11[0][0]                  
__________________________________________________________________________________________________
layer_normalization (LayerNorma (None, None, 128)    256         position__embedding[0][0]        
__________________________________________________________________________________________________
layer_normalization_6 (LayerNor (None, None, 128)    256         position__embedding_1[0][0]      
__________________________________________________________________________________________________
dropout_4 (Dropout)             (None, None, 128)    0           layer_normalization[0][0]        
__________________________________________________________________________________________________
dropout_8 (Dropout)             (None, None, 128)    0           layer_normalization_6[0][0]      
__________________________________________________________________________________________________
depthwise_conv1d (DepthwiseConv (None, None, 128)    17408       dropout_4[0][0]                  
                                                                 dropout_8[0][0]                  
__________________________________________________________________________________________________
layer_dropout (LayerDropout)    (None, None, 128)    0           depthwise_conv1d[0][0]           
                                                                 position__embedding[0][0]        
__________________________________________________________________________________________________
layer_dropout_6 (LayerDropout)  (None, None, 128)    0           depthwise_conv1d[1][0]           
                                                                 position__embedding_1[0][0]      
__________________________________________________________________________________________________
layer_normalization_1 (LayerNor (None, None, 128)    256         layer_dropout[0][0]              
__________________________________________________________________________________________________
layer_normalization_7 (LayerNor (None, None, 128)    256         layer_dropout_6[0][0]            
__________________________________________________________________________________________________
depthwise_conv1d_1 (DepthwiseCo (None, None, 128)    17408       layer_normalization_1[0][0]      
                                                                 layer_normalization_7[0][0]      
__________________________________________________________________________________________________
layer_dropout_1 (LayerDropout)  (None, None, 128)    0           depthwise_conv1d_1[0][0]         
                                                                 layer_dropout[0][0]              
__________________________________________________________________________________________________
layer_dropout_7 (LayerDropout)  (None, None, 128)    0           depthwise_conv1d_1[1][0]         
                                                                 layer_dropout_6[0][0]            
__________________________________________________________________________________________________
layer_normalization_2 (LayerNor (None, None, 128)    256         layer_dropout_1[0][0]            
__________________________________________________________________________________________________
layer_normalization_8 (LayerNor (None, None, 128)    256         layer_dropout_7[0][0]            
__________________________________________________________________________________________________
dropout_5 (Dropout)             (None, None, 128)    0           layer_normalization_2[0][0]      
__________________________________________________________________________________________________
dropout_9 (Dropout)             (None, None, 128)    0           layer_normalization_8[0][0]      
__________________________________________________________________________________________________
depthwise_conv1d_2 (DepthwiseCo (None, None, 128)    17408       dropout_5[0][0]                  
                                                                 dropout_9[0][0]                  
__________________________________________________________________________________________________
layer_dropout_2 (LayerDropout)  (None, None, 128)    0           depthwise_conv1d_2[0][0]         
                                                                 layer_dropout_1[0][0]            
__________________________________________________________________________________________________
layer_dropout_8 (LayerDropout)  (None, None, 128)    0           depthwise_conv1d_2[1][0]         
                                                                 layer_dropout_7[0][0]            
__________________________________________________________________________________________________
layer_normalization_3 (LayerNor (None, None, 128)    256         layer_dropout_2[0][0]            
__________________________________________________________________________________________________
layer_normalization_9 (LayerNor (None, None, 128)    256         layer_dropout_8[0][0]            
__________________________________________________________________________________________________
depthwise_conv1d_3 (DepthwiseCo (None, None, 128)    17408       layer_normalization_3[0][0]      
                                                                 layer_normalization_9[0][0]      
__________________________________________________________________________________________________
layer_dropout_3 (LayerDropout)  (None, None, 128)    0           depthwise_conv1d_3[0][0]         
                                                                 layer_dropout_2[0][0]            
__________________________________________________________________________________________________
layer_dropout_9 (LayerDropout)  (None, None, 128)    0           depthwise_conv1d_3[1][0]         
                                                                 layer_dropout_8[0][0]            
__________________________________________________________________________________________________
layer_normalization_4 (LayerNor (None, None, 128)    256         layer_dropout_3[0][0]            
__________________________________________________________________________________________________
layer_normalization_10 (LayerNo (None, None, 128)    256         layer_dropout_9[0][0]            
__________________________________________________________________________________________________
dropout_6 (Dropout)             (None, None, 128)    0           layer_normalization_4[0][0]      
__________________________________________________________________________________________________
dropout_10 (Dropout)            (None, None, 128)    0           layer_normalization_10[0][0]     
__________________________________________________________________________________________________
conv1d (Conv1D)                 (None, None, 256)    33024       dropout_6[0][0]                  
                                                                 dropout_10[0][0]                 
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, None, 128)    16512       dropout_6[0][0]                  
                                                                 dropout_10[0][0]                 
__________________________________________________________________________________________________
batch_slice_4 (BatchSlice)      (None, None)         0           lambda[0][0]                     
                                                                 lambda_2[0][0]                   
__________________________________________________________________________________________________
batch_slice_5 (BatchSlice)      (None, None)         0           lambda_1[0][0]                   
                                                                 lambda_3[0][0]                   
__________________________________________________________________________________________________
attention (Attention)           (None, None, 128)    0           conv1d[0][0]                     
                                                                 conv1d_1[0][0]                   
                                                                 batch_slice_4[0][0]              
                                                                 conv1d[1][0]                     
                                                                 conv1d_1[1][0]                   
                                                                 batch_slice_5[0][0]              
__________________________________________________________________________________________________
layer_dropout_4 (LayerDropout)  (None, None, 128)    0           attention[0][0]                  
                                                                 layer_dropout_3[0][0]            
__________________________________________________________________________________________________
layer_dropout_10 (LayerDropout) (None, None, 128)    0           attention[1][0]                  
                                                                 layer_dropout_9[0][0]            
__________________________________________________________________________________________________
layer_normalization_5 (LayerNor (None, None, 128)    256         layer_dropout_4[0][0]            
__________________________________________________________________________________________________
layer_normalization_11 (LayerNo (None, None, 128)    256         layer_dropout_10[0][0]           
__________________________________________________________________________________________________
dropout_7 (Dropout)             (None, None, 128)    0           layer_normalization_5[0][0]      
__________________________________________________________________________________________________
dropout_11 (Dropout)            (None, None, 128)    0           layer_normalization_11[0][0]     
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, None, 128)    16512       dropout_7[0][0]                  
                                                                 dropout_11[0][0]                 
__________________________________________________________________________________________________
conv1d_3 (Conv1D)               (None, None, 128)    16512       conv1d_2[0][0]                   
                                                                 conv1d_2[1][0]                   
__________________________________________________________________________________________________
layer_dropout_5 (LayerDropout)  (None, None, 128)    0           conv1d_3[0][0]                   
                                                                 layer_dropout_4[0][0]            
__________________________________________________________________________________________________
layer_dropout_11 (LayerDropout) (None, None, 128)    0           conv1d_3[1][0]                   
                                                                 layer_dropout_10[0][0]           
__________________________________________________________________________________________________
context2query_attention (contex (None, None, 512)    385         layer_dropout_5[0][0]            
                                                                 layer_dropout_11[0][0]           
                                                                 batch_slice_4[0][0]              
                                                                 batch_slice_5[0][0]              
__________________________________________________________________________________________________
conv1d_4 (Conv1D)               (None, None, 128)    65664       context2query_attention[0][0]    
__________________________________________________________________________________________________
position__embedding_2 (Position (None, None, 128)    0           conv1d_4[0][0]                   
__________________________________________________________________________________________________
layer_normalization_12 (LayerNo (None, None, 128)    256         position__embedding_2[0][0]      
__________________________________________________________________________________________________
dropout_12 (Dropout)            (None, None, 128)    0           layer_normalization_12[0][0]     
__________________________________________________________________________________________________
depthwise_conv1d_4 (DepthwiseCo (None, None, 128)    17152       dropout_12[0][0]                 
                                                                 dropout_33[0][0]                 
                                                                 dropout_54[0][0]                 
__________________________________________________________________________________________________
layer_dropout_12 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_4[0][0]         
                                                                 position__embedding_2[0][0]      
__________________________________________________________________________________________________
layer_normalization_13 (LayerNo (None, None, 128)    256         layer_dropout_12[0][0]           
__________________________________________________________________________________________________
depthwise_conv1d_5 (DepthwiseCo (None, None, 128)    17152       layer_normalization_13[0][0]     
                                                                 layer_normalization_41[0][0]     
                                                                 layer_normalization_69[0][0]     
__________________________________________________________________________________________________
layer_dropout_13 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_5[0][0]         
                                                                 layer_dropout_12[0][0]           
__________________________________________________________________________________________________
layer_normalization_14 (LayerNo (None, None, 128)    256         layer_dropout_13[0][0]           
__________________________________________________________________________________________________
dropout_13 (Dropout)            (None, None, 128)    0           layer_normalization_14[0][0]     
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, None, 256)    33024       dropout_13[0][0]                 
                                                                 dropout_34[0][0]                 
                                                                 dropout_55[0][0]                 
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, None, 128)    16512       dropout_13[0][0]                 
                                                                 dropout_34[0][0]                 
                                                                 dropout_55[0][0]                 
__________________________________________________________________________________________________
attention_1 (Attention)         (None, None, 128)    0           conv1d_5[0][0]                   
                                                                 conv1d_6[0][0]                   
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_5[1][0]                   
                                                                 conv1d_6[1][0]                   
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_5[2][0]                   
                                                                 conv1d_6[2][0]                   
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
layer_dropout_14 (LayerDropout) (None, None, 128)    0           attention_1[0][0]                
                                                                 layer_dropout_13[0][0]           
__________________________________________________________________________________________________
layer_normalization_15 (LayerNo (None, None, 128)    256         layer_dropout_14[0][0]           
__________________________________________________________________________________________________
dropout_14 (Dropout)            (None, None, 128)    0           layer_normalization_15[0][0]     
__________________________________________________________________________________________________
conv1d_7 (Conv1D)               (None, None, 128)    16512       dropout_14[0][0]                 
                                                                 dropout_35[0][0]                 
                                                                 dropout_56[0][0]                 
__________________________________________________________________________________________________
conv1d_8 (Conv1D)               (None, None, 128)    16512       conv1d_7[0][0]                   
                                                                 conv1d_7[1][0]                   
                                                                 conv1d_7[2][0]                   
__________________________________________________________________________________________________
layer_dropout_15 (LayerDropout) (None, None, 128)    0           conv1d_8[0][0]                   
                                                                 layer_dropout_14[0][0]           
__________________________________________________________________________________________________
position__embedding_3 (Position (None, None, 128)    0           layer_dropout_15[0][0]           
__________________________________________________________________________________________________
layer_normalization_16 (LayerNo (None, None, 128)    256         position__embedding_3[0][0]      
__________________________________________________________________________________________________
dropout_15 (Dropout)            (None, None, 128)    0           layer_normalization_16[0][0]     
__________________________________________________________________________________________________
depthwise_conv1d_6 (DepthwiseCo (None, None, 128)    17152       dropout_15[0][0]                 
                                                                 dropout_36[0][0]                 
                                                                 dropout_57[0][0]                 
__________________________________________________________________________________________________
layer_dropout_16 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_6[0][0]         
                                                                 position__embedding_3[0][0]      
__________________________________________________________________________________________________
layer_normalization_17 (LayerNo (None, None, 128)    256         layer_dropout_16[0][0]           
__________________________________________________________________________________________________
depthwise_conv1d_7 (DepthwiseCo (None, None, 128)    17152       layer_normalization_17[0][0]     
                                                                 layer_normalization_45[0][0]     
                                                                 layer_normalization_73[0][0]     
__________________________________________________________________________________________________
layer_dropout_17 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_7[0][0]         
                                                                 layer_dropout_16[0][0]           
__________________________________________________________________________________________________
layer_normalization_18 (LayerNo (None, None, 128)    256         layer_dropout_17[0][0]           
__________________________________________________________________________________________________
dropout_16 (Dropout)            (None, None, 128)    0           layer_normalization_18[0][0]     
__________________________________________________________________________________________________
conv1d_9 (Conv1D)               (None, None, 256)    33024       dropout_16[0][0]                 
                                                                 dropout_37[0][0]                 
                                                                 dropout_58[0][0]                 
__________________________________________________________________________________________________
conv1d_10 (Conv1D)              (None, None, 128)    16512       dropout_16[0][0]                 
                                                                 dropout_37[0][0]                 
                                                                 dropout_58[0][0]                 
__________________________________________________________________________________________________
attention_2 (Attention)         (None, None, 128)    0           conv1d_9[0][0]                   
                                                                 conv1d_10[0][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_9[1][0]                   
                                                                 conv1d_10[1][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_9[2][0]                   
                                                                 conv1d_10[2][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
layer_dropout_18 (LayerDropout) (None, None, 128)    0           attention_2[0][0]                
                                                                 layer_dropout_17[0][0]           
__________________________________________________________________________________________________
layer_normalization_19 (LayerNo (None, None, 128)    256         layer_dropout_18[0][0]           
__________________________________________________________________________________________________
dropout_17 (Dropout)            (None, None, 128)    0           layer_normalization_19[0][0]     
__________________________________________________________________________________________________
conv1d_11 (Conv1D)              (None, None, 128)    16512       dropout_17[0][0]                 
                                                                 dropout_38[0][0]                 
                                                                 dropout_59[0][0]                 
__________________________________________________________________________________________________
conv1d_12 (Conv1D)              (None, None, 128)    16512       conv1d_11[0][0]                  
                                                                 conv1d_11[1][0]                  
                                                                 conv1d_11[2][0]                  
__________________________________________________________________________________________________
layer_dropout_19 (LayerDropout) (None, None, 128)    0           conv1d_12[0][0]                  
                                                                 layer_dropout_18[0][0]           
__________________________________________________________________________________________________
position__embedding_4 (Position (None, None, 128)    0           layer_dropout_19[0][0]           
__________________________________________________________________________________________________
layer_normalization_20 (LayerNo (None, None, 128)    256         position__embedding_4[0][0]      
__________________________________________________________________________________________________
dropout_18 (Dropout)            (None, None, 128)    0           layer_normalization_20[0][0]     
__________________________________________________________________________________________________
depthwise_conv1d_8 (DepthwiseCo (None, None, 128)    17152       dropout_18[0][0]                 
                                                                 dropout_39[0][0]                 
                                                                 dropout_60[0][0]                 
__________________________________________________________________________________________________
layer_dropout_20 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_8[0][0]         
                                                                 position__embedding_4[0][0]      
__________________________________________________________________________________________________
layer_normalization_21 (LayerNo (None, None, 128)    256         layer_dropout_20[0][0]           
__________________________________________________________________________________________________
depthwise_conv1d_9 (DepthwiseCo (None, None, 128)    17152       layer_normalization_21[0][0]     
                                                                 layer_normalization_49[0][0]     
                                                                 layer_normalization_77[0][0]     
__________________________________________________________________________________________________
layer_dropout_21 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_9[0][0]         
                                                                 layer_dropout_20[0][0]           
__________________________________________________________________________________________________
layer_normalization_22 (LayerNo (None, None, 128)    256         layer_dropout_21[0][0]           
__________________________________________________________________________________________________
dropout_19 (Dropout)            (None, None, 128)    0           layer_normalization_22[0][0]     
__________________________________________________________________________________________________
conv1d_13 (Conv1D)              (None, None, 256)    33024       dropout_19[0][0]                 
                                                                 dropout_40[0][0]                 
                                                                 dropout_61[0][0]                 
__________________________________________________________________________________________________
conv1d_14 (Conv1D)              (None, None, 128)    16512       dropout_19[0][0]                 
                                                                 dropout_40[0][0]                 
                                                                 dropout_61[0][0]                 
__________________________________________________________________________________________________
attention_3 (Attention)         (None, None, 128)    0           conv1d_13[0][0]                  
                                                                 conv1d_14[0][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_13[1][0]                  
                                                                 conv1d_14[1][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_13[2][0]                  
                                                                 conv1d_14[2][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
layer_dropout_22 (LayerDropout) (None, None, 128)    0           attention_3[0][0]                
                                                                 layer_dropout_21[0][0]           
__________________________________________________________________________________________________
layer_normalization_23 (LayerNo (None, None, 128)    256         layer_dropout_22[0][0]           
__________________________________________________________________________________________________
dropout_20 (Dropout)            (None, None, 128)    0           layer_normalization_23[0][0]     
__________________________________________________________________________________________________
conv1d_15 (Conv1D)              (None, None, 128)    16512       dropout_20[0][0]                 
                                                                 dropout_41[0][0]                 
                                                                 dropout_62[0][0]                 
__________________________________________________________________________________________________
conv1d_16 (Conv1D)              (None, None, 128)    16512       conv1d_15[0][0]                  
                                                                 conv1d_15[1][0]                  
                                                                 conv1d_15[2][0]                  
__________________________________________________________________________________________________
layer_dropout_23 (LayerDropout) (None, None, 128)    0           conv1d_16[0][0]                  
                                                                 layer_dropout_22[0][0]           
__________________________________________________________________________________________________
position__embedding_5 (Position (None, None, 128)    0           layer_dropout_23[0][0]           
__________________________________________________________________________________________________
layer_normalization_24 (LayerNo (None, None, 128)    256         position__embedding_5[0][0]      
__________________________________________________________________________________________________
dropout_21 (Dropout)            (None, None, 128)    0           layer_normalization_24[0][0]     
__________________________________________________________________________________________________
depthwise_conv1d_10 (DepthwiseC (None, None, 128)    17152       dropout_21[0][0]                 
                                                                 dropout_42[0][0]                 
                                                                 dropout_63[0][0]                 
__________________________________________________________________________________________________
layer_dropout_24 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_10[0][0]        
                                                                 position__embedding_5[0][0]      
__________________________________________________________________________________________________
layer_normalization_25 (LayerNo (None, None, 128)    256         layer_dropout_24[0][0]           
__________________________________________________________________________________________________
depthwise_conv1d_11 (DepthwiseC (None, None, 128)    17152       layer_normalization_25[0][0]     
                                                                 layer_normalization_53[0][0]     
                                                                 layer_normalization_81[0][0]     
__________________________________________________________________________________________________
layer_dropout_25 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_11[0][0]        
                                                                 layer_dropout_24[0][0]           
__________________________________________________________________________________________________
layer_normalization_26 (LayerNo (None, None, 128)    256         layer_dropout_25[0][0]           
__________________________________________________________________________________________________
dropout_22 (Dropout)            (None, None, 128)    0           layer_normalization_26[0][0]     
__________________________________________________________________________________________________
conv1d_17 (Conv1D)              (None, None, 256)    33024       dropout_22[0][0]                 
                                                                 dropout_43[0][0]                 
                                                                 dropout_64[0][0]                 
__________________________________________________________________________________________________
conv1d_18 (Conv1D)              (None, None, 128)    16512       dropout_22[0][0]                 
                                                                 dropout_43[0][0]                 
                                                                 dropout_64[0][0]                 
__________________________________________________________________________________________________
attention_4 (Attention)         (None, None, 128)    0           conv1d_17[0][0]                  
                                                                 conv1d_18[0][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_17[1][0]                  
                                                                 conv1d_18[1][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_17[2][0]                  
                                                                 conv1d_18[2][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
layer_dropout_26 (LayerDropout) (None, None, 128)    0           attention_4[0][0]                
                                                                 layer_dropout_25[0][0]           
__________________________________________________________________________________________________
layer_normalization_27 (LayerNo (None, None, 128)    256         layer_dropout_26[0][0]           
__________________________________________________________________________________________________
dropout_23 (Dropout)            (None, None, 128)    0           layer_normalization_27[0][0]     
__________________________________________________________________________________________________
conv1d_19 (Conv1D)              (None, None, 128)    16512       dropout_23[0][0]                 
                                                                 dropout_44[0][0]                 
                                                                 dropout_65[0][0]                 
__________________________________________________________________________________________________
conv1d_20 (Conv1D)              (None, None, 128)    16512       conv1d_19[0][0]                  
                                                                 conv1d_19[1][0]                  
                                                                 conv1d_19[2][0]                  
__________________________________________________________________________________________________
layer_dropout_27 (LayerDropout) (None, None, 128)    0           conv1d_20[0][0]                  
                                                                 layer_dropout_26[0][0]           
__________________________________________________________________________________________________
position__embedding_6 (Position (None, None, 128)    0           layer_dropout_27[0][0]           
__________________________________________________________________________________________________
layer_normalization_28 (LayerNo (None, None, 128)    256         position__embedding_6[0][0]      
__________________________________________________________________________________________________
dropout_24 (Dropout)            (None, None, 128)    0           layer_normalization_28[0][0]     
__________________________________________________________________________________________________
depthwise_conv1d_12 (DepthwiseC (None, None, 128)    17152       dropout_24[0][0]                 
                                                                 dropout_45[0][0]                 
                                                                 dropout_66[0][0]                 
__________________________________________________________________________________________________
layer_dropout_28 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_12[0][0]        
                                                                 position__embedding_6[0][0]      
__________________________________________________________________________________________________
layer_normalization_29 (LayerNo (None, None, 128)    256         layer_dropout_28[0][0]           
__________________________________________________________________________________________________
depthwise_conv1d_13 (DepthwiseC (None, None, 128)    17152       layer_normalization_29[0][0]     
                                                                 layer_normalization_57[0][0]     
                                                                 layer_normalization_85[0][0]     
__________________________________________________________________________________________________
layer_dropout_29 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_13[0][0]        
                                                                 layer_dropout_28[0][0]           
__________________________________________________________________________________________________
layer_normalization_30 (LayerNo (None, None, 128)    256         layer_dropout_29[0][0]           
__________________________________________________________________________________________________
dropout_25 (Dropout)            (None, None, 128)    0           layer_normalization_30[0][0]     
__________________________________________________________________________________________________
conv1d_21 (Conv1D)              (None, None, 256)    33024       dropout_25[0][0]                 
                                                                 dropout_46[0][0]                 
                                                                 dropout_67[0][0]                 
__________________________________________________________________________________________________
conv1d_22 (Conv1D)              (None, None, 128)    16512       dropout_25[0][0]                 
                                                                 dropout_46[0][0]                 
                                                                 dropout_67[0][0]                 
__________________________________________________________________________________________________
attention_5 (Attention)         (None, None, 128)    0           conv1d_21[0][0]                  
                                                                 conv1d_22[0][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_21[1][0]                  
                                                                 conv1d_22[1][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_21[2][0]                  
                                                                 conv1d_22[2][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
layer_dropout_30 (LayerDropout) (None, None, 128)    0           attention_5[0][0]                
                                                                 layer_dropout_29[0][0]           
__________________________________________________________________________________________________
layer_normalization_31 (LayerNo (None, None, 128)    256         layer_dropout_30[0][0]           
__________________________________________________________________________________________________
dropout_26 (Dropout)            (None, None, 128)    0           layer_normalization_31[0][0]     
__________________________________________________________________________________________________
conv1d_23 (Conv1D)              (None, None, 128)    16512       dropout_26[0][0]                 
                                                                 dropout_47[0][0]                 
                                                                 dropout_68[0][0]                 
__________________________________________________________________________________________________
conv1d_24 (Conv1D)              (None, None, 128)    16512       conv1d_23[0][0]                  
                                                                 conv1d_23[1][0]                  
                                                                 conv1d_23[2][0]                  
__________________________________________________________________________________________________
layer_dropout_31 (LayerDropout) (None, None, 128)    0           conv1d_24[0][0]                  
                                                                 layer_dropout_30[0][0]           
__________________________________________________________________________________________________
position__embedding_7 (Position (None, None, 128)    0           layer_dropout_31[0][0]           
__________________________________________________________________________________________________
layer_normalization_32 (LayerNo (None, None, 128)    256         position__embedding_7[0][0]      
__________________________________________________________________________________________________
dropout_27 (Dropout)            (None, None, 128)    0           layer_normalization_32[0][0]     
__________________________________________________________________________________________________
depthwise_conv1d_14 (DepthwiseC (None, None, 128)    17152       dropout_27[0][0]                 
                                                                 dropout_48[0][0]                 
                                                                 dropout_69[0][0]                 
__________________________________________________________________________________________________
layer_dropout_32 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_14[0][0]        
                                                                 position__embedding_7[0][0]      
__________________________________________________________________________________________________
layer_normalization_33 (LayerNo (None, None, 128)    256         layer_dropout_32[0][0]           
__________________________________________________________________________________________________
depthwise_conv1d_15 (DepthwiseC (None, None, 128)    17152       layer_normalization_33[0][0]     
                                                                 layer_normalization_61[0][0]     
                                                                 layer_normalization_89[0][0]     
__________________________________________________________________________________________________
layer_dropout_33 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_15[0][0]        
                                                                 layer_dropout_32[0][0]           
__________________________________________________________________________________________________
layer_normalization_34 (LayerNo (None, None, 128)    256         layer_dropout_33[0][0]           
__________________________________________________________________________________________________
dropout_28 (Dropout)            (None, None, 128)    0           layer_normalization_34[0][0]     
__________________________________________________________________________________________________
conv1d_25 (Conv1D)              (None, None, 256)    33024       dropout_28[0][0]                 
                                                                 dropout_49[0][0]                 
                                                                 dropout_70[0][0]                 
__________________________________________________________________________________________________
conv1d_26 (Conv1D)              (None, None, 128)    16512       dropout_28[0][0]                 
                                                                 dropout_49[0][0]                 
                                                                 dropout_70[0][0]                 
__________________________________________________________________________________________________
attention_6 (Attention)         (None, None, 128)    0           conv1d_25[0][0]                  
                                                                 conv1d_26[0][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_25[1][0]                  
                                                                 conv1d_26[1][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_25[2][0]                  
                                                                 conv1d_26[2][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
layer_dropout_34 (LayerDropout) (None, None, 128)    0           attention_6[0][0]                
                                                                 layer_dropout_33[0][0]           
__________________________________________________________________________________________________
layer_normalization_35 (LayerNo (None, None, 128)    256         layer_dropout_34[0][0]           
__________________________________________________________________________________________________
dropout_29 (Dropout)            (None, None, 128)    0           layer_normalization_35[0][0]     
__________________________________________________________________________________________________
conv1d_27 (Conv1D)              (None, None, 128)    16512       dropout_29[0][0]                 
                                                                 dropout_50[0][0]                 
                                                                 dropout_71[0][0]                 
__________________________________________________________________________________________________
conv1d_28 (Conv1D)              (None, None, 128)    16512       conv1d_27[0][0]                  
                                                                 conv1d_27[1][0]                  
                                                                 conv1d_27[2][0]                  
__________________________________________________________________________________________________
layer_dropout_35 (LayerDropout) (None, None, 128)    0           conv1d_28[0][0]                  
                                                                 layer_dropout_34[0][0]           
__________________________________________________________________________________________________
position__embedding_8 (Position (None, None, 128)    0           layer_dropout_35[0][0]           
__________________________________________________________________________________________________
layer_normalization_36 (LayerNo (None, None, 128)    256         position__embedding_8[0][0]      
__________________________________________________________________________________________________
dropout_30 (Dropout)            (None, None, 128)    0           layer_normalization_36[0][0]     
__________________________________________________________________________________________________
depthwise_conv1d_16 (DepthwiseC (None, None, 128)    17152       dropout_30[0][0]                 
                                                                 dropout_51[0][0]                 
                                                                 dropout_72[0][0]                 
__________________________________________________________________________________________________
layer_dropout_36 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_16[0][0]        
                                                                 position__embedding_8[0][0]      
__________________________________________________________________________________________________
layer_normalization_37 (LayerNo (None, None, 128)    256         layer_dropout_36[0][0]           
__________________________________________________________________________________________________
depthwise_conv1d_17 (DepthwiseC (None, None, 128)    17152       layer_normalization_37[0][0]     
                                                                 layer_normalization_65[0][0]     
                                                                 layer_normalization_93[0][0]     
__________________________________________________________________________________________________
layer_dropout_37 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_17[0][0]        
                                                                 layer_dropout_36[0][0]           
__________________________________________________________________________________________________
layer_normalization_38 (LayerNo (None, None, 128)    256         layer_dropout_37[0][0]           
__________________________________________________________________________________________________
dropout_31 (Dropout)            (None, None, 128)    0           layer_normalization_38[0][0]     
__________________________________________________________________________________________________
conv1d_29 (Conv1D)              (None, None, 256)    33024       dropout_31[0][0]                 
                                                                 dropout_52[0][0]                 
                                                                 dropout_73[0][0]                 
__________________________________________________________________________________________________
conv1d_30 (Conv1D)              (None, None, 128)    16512       dropout_31[0][0]                 
                                                                 dropout_52[0][0]                 
                                                                 dropout_73[0][0]                 
__________________________________________________________________________________________________
attention_7 (Attention)         (None, None, 128)    0           conv1d_29[0][0]                  
                                                                 conv1d_30[0][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_29[1][0]                  
                                                                 conv1d_30[1][0]                  
                                                                 batch_slice_4[0][0]              
                                                                 conv1d_29[2][0]                  
                                                                 conv1d_30[2][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
layer_dropout_38 (LayerDropout) (None, None, 128)    0           attention_7[0][0]                
                                                                 layer_dropout_37[0][0]           
__________________________________________________________________________________________________
layer_normalization_39 (LayerNo (None, None, 128)    256         layer_dropout_38[0][0]           
__________________________________________________________________________________________________
dropout_32 (Dropout)            (None, None, 128)    0           layer_normalization_39[0][0]     
__________________________________________________________________________________________________
conv1d_31 (Conv1D)              (None, None, 128)    16512       dropout_32[0][0]                 
                                                                 dropout_53[0][0]                 
                                                                 dropout_74[0][0]                 
__________________________________________________________________________________________________
conv1d_32 (Conv1D)              (None, None, 128)    16512       conv1d_31[0][0]                  
                                                                 conv1d_31[1][0]                  
                                                                 conv1d_31[2][0]                  
__________________________________________________________________________________________________
layer_dropout_39 (LayerDropout) (None, None, 128)    0           conv1d_32[0][0]                  
                                                                 layer_dropout_38[0][0]           
__________________________________________________________________________________________________
position__embedding_9 (Position (None, None, 128)    0           layer_dropout_39[0][0]           
__________________________________________________________________________________________________
layer_normalization_40 (LayerNo (None, None, 128)    256         position__embedding_9[0][0]      
__________________________________________________________________________________________________
dropout_33 (Dropout)            (None, None, 128)    0           layer_normalization_40[0][0]     
__________________________________________________________________________________________________
layer_dropout_40 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_4[1][0]         
                                                                 position__embedding_9[0][0]      
__________________________________________________________________________________________________
layer_normalization_41 (LayerNo (None, None, 128)    256         layer_dropout_40[0][0]           
__________________________________________________________________________________________________
layer_dropout_41 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_5[1][0]         
                                                                 layer_dropout_40[0][0]           
__________________________________________________________________________________________________
layer_normalization_42 (LayerNo (None, None, 128)    256         layer_dropout_41[0][0]           
__________________________________________________________________________________________________
dropout_34 (Dropout)            (None, None, 128)    0           layer_normalization_42[0][0]     
__________________________________________________________________________________________________
layer_dropout_42 (LayerDropout) (None, None, 128)    0           attention_1[1][0]                
                                                                 layer_dropout_41[0][0]           
__________________________________________________________________________________________________
layer_normalization_43 (LayerNo (None, None, 128)    256         layer_dropout_42[0][0]           
__________________________________________________________________________________________________
dropout_35 (Dropout)            (None, None, 128)    0           layer_normalization_43[0][0]     
__________________________________________________________________________________________________
layer_dropout_43 (LayerDropout) (None, None, 128)    0           conv1d_8[1][0]                   
                                                                 layer_dropout_42[0][0]           
__________________________________________________________________________________________________
position__embedding_10 (Positio (None, None, 128)    0           layer_dropout_43[0][0]           
__________________________________________________________________________________________________
layer_normalization_44 (LayerNo (None, None, 128)    256         position__embedding_10[0][0]     
__________________________________________________________________________________________________
dropout_36 (Dropout)            (None, None, 128)    0           layer_normalization_44[0][0]     
__________________________________________________________________________________________________
layer_dropout_44 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_6[1][0]         
                                                                 position__embedding_10[0][0]     
__________________________________________________________________________________________________
layer_normalization_45 (LayerNo (None, None, 128)    256         layer_dropout_44[0][0]           
__________________________________________________________________________________________________
layer_dropout_45 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_7[1][0]         
                                                                 layer_dropout_44[0][0]           
__________________________________________________________________________________________________
layer_normalization_46 (LayerNo (None, None, 128)    256         layer_dropout_45[0][0]           
__________________________________________________________________________________________________
dropout_37 (Dropout)            (None, None, 128)    0           layer_normalization_46[0][0]     
__________________________________________________________________________________________________
layer_dropout_46 (LayerDropout) (None, None, 128)    0           attention_2[1][0]                
                                                                 layer_dropout_45[0][0]           
__________________________________________________________________________________________________
layer_normalization_47 (LayerNo (None, None, 128)    256         layer_dropout_46[0][0]           
__________________________________________________________________________________________________
dropout_38 (Dropout)            (None, None, 128)    0           layer_normalization_47[0][0]     
__________________________________________________________________________________________________
layer_dropout_47 (LayerDropout) (None, None, 128)    0           conv1d_12[1][0]                  
                                                                 layer_dropout_46[0][0]           
__________________________________________________________________________________________________
position__embedding_11 (Positio (None, None, 128)    0           layer_dropout_47[0][0]           
__________________________________________________________________________________________________
layer_normalization_48 (LayerNo (None, None, 128)    256         position__embedding_11[0][0]     
__________________________________________________________________________________________________
dropout_39 (Dropout)            (None, None, 128)    0           layer_normalization_48[0][0]     
__________________________________________________________________________________________________
layer_dropout_48 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_8[1][0]         
                                                                 position__embedding_11[0][0]     
__________________________________________________________________________________________________
layer_normalization_49 (LayerNo (None, None, 128)    256         layer_dropout_48[0][0]           
__________________________________________________________________________________________________
layer_dropout_49 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_9[1][0]         
                                                                 layer_dropout_48[0][0]           
__________________________________________________________________________________________________
layer_normalization_50 (LayerNo (None, None, 128)    256         layer_dropout_49[0][0]           
__________________________________________________________________________________________________
dropout_40 (Dropout)            (None, None, 128)    0           layer_normalization_50[0][0]     
__________________________________________________________________________________________________
layer_dropout_50 (LayerDropout) (None, None, 128)    0           attention_3[1][0]                
                                                                 layer_dropout_49[0][0]           
__________________________________________________________________________________________________
layer_normalization_51 (LayerNo (None, None, 128)    256         layer_dropout_50[0][0]           
__________________________________________________________________________________________________
dropout_41 (Dropout)            (None, None, 128)    0           layer_normalization_51[0][0]     
__________________________________________________________________________________________________
layer_dropout_51 (LayerDropout) (None, None, 128)    0           conv1d_16[1][0]                  
                                                                 layer_dropout_50[0][0]           
__________________________________________________________________________________________________
position__embedding_12 (Positio (None, None, 128)    0           layer_dropout_51[0][0]           
__________________________________________________________________________________________________
layer_normalization_52 (LayerNo (None, None, 128)    256         position__embedding_12[0][0]     
__________________________________________________________________________________________________
dropout_42 (Dropout)            (None, None, 128)    0           layer_normalization_52[0][0]     
__________________________________________________________________________________________________
layer_dropout_52 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_10[1][0]        
                                                                 position__embedding_12[0][0]     
__________________________________________________________________________________________________
layer_normalization_53 (LayerNo (None, None, 128)    256         layer_dropout_52[0][0]           
__________________________________________________________________________________________________
layer_dropout_53 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_11[1][0]        
                                                                 layer_dropout_52[0][0]           
__________________________________________________________________________________________________
layer_normalization_54 (LayerNo (None, None, 128)    256         layer_dropout_53[0][0]           
__________________________________________________________________________________________________
dropout_43 (Dropout)            (None, None, 128)    0           layer_normalization_54[0][0]     
__________________________________________________________________________________________________
layer_dropout_54 (LayerDropout) (None, None, 128)    0           attention_4[1][0]                
                                                                 layer_dropout_53[0][0]           
__________________________________________________________________________________________________
layer_normalization_55 (LayerNo (None, None, 128)    256         layer_dropout_54[0][0]           
__________________________________________________________________________________________________
dropout_44 (Dropout)            (None, None, 128)    0           layer_normalization_55[0][0]     
__________________________________________________________________________________________________
layer_dropout_55 (LayerDropout) (None, None, 128)    0           conv1d_20[1][0]                  
                                                                 layer_dropout_54[0][0]           
__________________________________________________________________________________________________
position__embedding_13 (Positio (None, None, 128)    0           layer_dropout_55[0][0]           
__________________________________________________________________________________________________
layer_normalization_56 (LayerNo (None, None, 128)    256         position__embedding_13[0][0]     
__________________________________________________________________________________________________
dropout_45 (Dropout)            (None, None, 128)    0           layer_normalization_56[0][0]     
__________________________________________________________________________________________________
layer_dropout_56 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_12[1][0]        
                                                                 position__embedding_13[0][0]     
__________________________________________________________________________________________________
layer_normalization_57 (LayerNo (None, None, 128)    256         layer_dropout_56[0][0]           
__________________________________________________________________________________________________
layer_dropout_57 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_13[1][0]        
                                                                 layer_dropout_56[0][0]           
__________________________________________________________________________________________________
layer_normalization_58 (LayerNo (None, None, 128)    256         layer_dropout_57[0][0]           
__________________________________________________________________________________________________
dropout_46 (Dropout)            (None, None, 128)    0           layer_normalization_58[0][0]     
__________________________________________________________________________________________________
layer_dropout_58 (LayerDropout) (None, None, 128)    0           attention_5[1][0]                
                                                                 layer_dropout_57[0][0]           
__________________________________________________________________________________________________
layer_normalization_59 (LayerNo (None, None, 128)    256         layer_dropout_58[0][0]           
__________________________________________________________________________________________________
dropout_47 (Dropout)            (None, None, 128)    0           layer_normalization_59[0][0]     
__________________________________________________________________________________________________
layer_dropout_59 (LayerDropout) (None, None, 128)    0           conv1d_24[1][0]                  
                                                                 layer_dropout_58[0][0]           
__________________________________________________________________________________________________
position__embedding_14 (Positio (None, None, 128)    0           layer_dropout_59[0][0]           
__________________________________________________________________________________________________
layer_normalization_60 (LayerNo (None, None, 128)    256         position__embedding_14[0][0]     
__________________________________________________________________________________________________
dropout_48 (Dropout)            (None, None, 128)    0           layer_normalization_60[0][0]     
__________________________________________________________________________________________________
layer_dropout_60 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_14[1][0]        
                                                                 position__embedding_14[0][0]     
__________________________________________________________________________________________________
layer_normalization_61 (LayerNo (None, None, 128)    256         layer_dropout_60[0][0]           
__________________________________________________________________________________________________
layer_dropout_61 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_15[1][0]        
                                                                 layer_dropout_60[0][0]           
__________________________________________________________________________________________________
layer_normalization_62 (LayerNo (None, None, 128)    256         layer_dropout_61[0][0]           
__________________________________________________________________________________________________
dropout_49 (Dropout)            (None, None, 128)    0           layer_normalization_62[0][0]     
__________________________________________________________________________________________________
layer_dropout_62 (LayerDropout) (None, None, 128)    0           attention_6[1][0]                
                                                                 layer_dropout_61[0][0]           
__________________________________________________________________________________________________
layer_normalization_63 (LayerNo (None, None, 128)    256         layer_dropout_62[0][0]           
__________________________________________________________________________________________________
dropout_50 (Dropout)            (None, None, 128)    0           layer_normalization_63[0][0]     
__________________________________________________________________________________________________
layer_dropout_63 (LayerDropout) (None, None, 128)    0           conv1d_28[1][0]                  
                                                                 layer_dropout_62[0][0]           
__________________________________________________________________________________________________
position__embedding_15 (Positio (None, None, 128)    0           layer_dropout_63[0][0]           
__________________________________________________________________________________________________
layer_normalization_64 (LayerNo (None, None, 128)    256         position__embedding_15[0][0]     
__________________________________________________________________________________________________
dropout_51 (Dropout)            (None, None, 128)    0           layer_normalization_64[0][0]     
__________________________________________________________________________________________________
layer_dropout_64 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_16[1][0]        
                                                                 position__embedding_15[0][0]     
__________________________________________________________________________________________________
layer_normalization_65 (LayerNo (None, None, 128)    256         layer_dropout_64[0][0]           
__________________________________________________________________________________________________
layer_dropout_65 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_17[1][0]        
                                                                 layer_dropout_64[0][0]           
__________________________________________________________________________________________________
layer_normalization_66 (LayerNo (None, None, 128)    256         layer_dropout_65[0][0]           
__________________________________________________________________________________________________
dropout_52 (Dropout)            (None, None, 128)    0           layer_normalization_66[0][0]     
__________________________________________________________________________________________________
layer_dropout_66 (LayerDropout) (None, None, 128)    0           attention_7[1][0]                
                                                                 layer_dropout_65[0][0]           
__________________________________________________________________________________________________
layer_normalization_67 (LayerNo (None, None, 128)    256         layer_dropout_66[0][0]           
__________________________________________________________________________________________________
dropout_53 (Dropout)            (None, None, 128)    0           layer_normalization_67[0][0]     
__________________________________________________________________________________________________
layer_dropout_67 (LayerDropout) (None, None, 128)    0           conv1d_32[1][0]                  
                                                                 layer_dropout_66[0][0]           
__________________________________________________________________________________________________
position__embedding_16 (Positio (None, None, 128)    0           layer_dropout_67[0][0]           
__________________________________________________________________________________________________
layer_normalization_68 (LayerNo (None, None, 128)    256         position__embedding_16[0][0]     
__________________________________________________________________________________________________
dropout_54 (Dropout)            (None, None, 128)    0           layer_normalization_68[0][0]     
__________________________________________________________________________________________________
layer_dropout_68 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_4[2][0]         
                                                                 position__embedding_16[0][0]     
__________________________________________________________________________________________________
layer_normalization_69 (LayerNo (None, None, 128)    256         layer_dropout_68[0][0]           
__________________________________________________________________________________________________
layer_dropout_69 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_5[2][0]         
                                                                 layer_dropout_68[0][0]           
__________________________________________________________________________________________________
layer_normalization_70 (LayerNo (None, None, 128)    256         layer_dropout_69[0][0]           
__________________________________________________________________________________________________
dropout_55 (Dropout)            (None, None, 128)    0           layer_normalization_70[0][0]     
__________________________________________________________________________________________________
layer_dropout_70 (LayerDropout) (None, None, 128)    0           attention_1[2][0]                
                                                                 layer_dropout_69[0][0]           
__________________________________________________________________________________________________
layer_normalization_71 (LayerNo (None, None, 128)    256         layer_dropout_70[0][0]           
__________________________________________________________________________________________________
dropout_56 (Dropout)            (None, None, 128)    0           layer_normalization_71[0][0]     
__________________________________________________________________________________________________
layer_dropout_71 (LayerDropout) (None, None, 128)    0           conv1d_8[2][0]                   
                                                                 layer_dropout_70[0][0]           
__________________________________________________________________________________________________
position__embedding_17 (Positio (None, None, 128)    0           layer_dropout_71[0][0]           
__________________________________________________________________________________________________
layer_normalization_72 (LayerNo (None, None, 128)    256         position__embedding_17[0][0]     
__________________________________________________________________________________________________
dropout_57 (Dropout)            (None, None, 128)    0           layer_normalization_72[0][0]     
__________________________________________________________________________________________________
layer_dropout_72 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_6[2][0]         
                                                                 position__embedding_17[0][0]     
__________________________________________________________________________________________________
layer_normalization_73 (LayerNo (None, None, 128)    256         layer_dropout_72[0][0]           
__________________________________________________________________________________________________
layer_dropout_73 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_7[2][0]         
                                                                 layer_dropout_72[0][0]           
__________________________________________________________________________________________________
layer_normalization_74 (LayerNo (None, None, 128)    256         layer_dropout_73[0][0]           
__________________________________________________________________________________________________
dropout_58 (Dropout)            (None, None, 128)    0           layer_normalization_74[0][0]     
__________________________________________________________________________________________________
layer_dropout_74 (LayerDropout) (None, None, 128)    0           attention_2[2][0]                
                                                                 layer_dropout_73[0][0]           
__________________________________________________________________________________________________
layer_normalization_75 (LayerNo (None, None, 128)    256         layer_dropout_74[0][0]           
__________________________________________________________________________________________________
dropout_59 (Dropout)            (None, None, 128)    0           layer_normalization_75[0][0]     
__________________________________________________________________________________________________
layer_dropout_75 (LayerDropout) (None, None, 128)    0           conv1d_12[2][0]                  
                                                                 layer_dropout_74[0][0]           
__________________________________________________________________________________________________
position__embedding_18 (Positio (None, None, 128)    0           layer_dropout_75[0][0]           
__________________________________________________________________________________________________
layer_normalization_76 (LayerNo (None, None, 128)    256         position__embedding_18[0][0]     
__________________________________________________________________________________________________
dropout_60 (Dropout)            (None, None, 128)    0           layer_normalization_76[0][0]     
__________________________________________________________________________________________________
layer_dropout_76 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_8[2][0]         
                                                                 position__embedding_18[0][0]     
__________________________________________________________________________________________________
layer_normalization_77 (LayerNo (None, None, 128)    256         layer_dropout_76[0][0]           
__________________________________________________________________________________________________
layer_dropout_77 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_9[2][0]         
                                                                 layer_dropout_76[0][0]           
__________________________________________________________________________________________________
layer_normalization_78 (LayerNo (None, None, 128)    256         layer_dropout_77[0][0]           
__________________________________________________________________________________________________
dropout_61 (Dropout)            (None, None, 128)    0           layer_normalization_78[0][0]     
__________________________________________________________________________________________________
layer_dropout_78 (LayerDropout) (None, None, 128)    0           attention_3[2][0]                
                                                                 layer_dropout_77[0][0]           
__________________________________________________________________________________________________
layer_normalization_79 (LayerNo (None, None, 128)    256         layer_dropout_78[0][0]           
__________________________________________________________________________________________________
dropout_62 (Dropout)            (None, None, 128)    0           layer_normalization_79[0][0]     
__________________________________________________________________________________________________
layer_dropout_79 (LayerDropout) (None, None, 128)    0           conv1d_16[2][0]                  
                                                                 layer_dropout_78[0][0]           
__________________________________________________________________________________________________
position__embedding_19 (Positio (None, None, 128)    0           layer_dropout_79[0][0]           
__________________________________________________________________________________________________
layer_normalization_80 (LayerNo (None, None, 128)    256         position__embedding_19[0][0]     
__________________________________________________________________________________________________
dropout_63 (Dropout)            (None, None, 128)    0           layer_normalization_80[0][0]     
__________________________________________________________________________________________________
layer_dropout_80 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_10[2][0]        
                                                                 position__embedding_19[0][0]     
__________________________________________________________________________________________________
layer_normalization_81 (LayerNo (None, None, 128)    256         layer_dropout_80[0][0]           
__________________________________________________________________________________________________
layer_dropout_81 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_11[2][0]        
                                                                 layer_dropout_80[0][0]           
__________________________________________________________________________________________________
layer_normalization_82 (LayerNo (None, None, 128)    256         layer_dropout_81[0][0]           
__________________________________________________________________________________________________
dropout_64 (Dropout)            (None, None, 128)    0           layer_normalization_82[0][0]     
__________________________________________________________________________________________________
layer_dropout_82 (LayerDropout) (None, None, 128)    0           attention_4[2][0]                
                                                                 layer_dropout_81[0][0]           
__________________________________________________________________________________________________
layer_normalization_83 (LayerNo (None, None, 128)    256         layer_dropout_82[0][0]           
__________________________________________________________________________________________________
dropout_65 (Dropout)            (None, None, 128)    0           layer_normalization_83[0][0]     
__________________________________________________________________________________________________
layer_dropout_83 (LayerDropout) (None, None, 128)    0           conv1d_20[2][0]                  
                                                                 layer_dropout_82[0][0]           
__________________________________________________________________________________________________
position__embedding_20 (Positio (None, None, 128)    0           layer_dropout_83[0][0]           
__________________________________________________________________________________________________
layer_normalization_84 (LayerNo (None, None, 128)    256         position__embedding_20[0][0]     
__________________________________________________________________________________________________
dropout_66 (Dropout)            (None, None, 128)    0           layer_normalization_84[0][0]     
__________________________________________________________________________________________________
layer_dropout_84 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_12[2][0]        
                                                                 position__embedding_20[0][0]     
__________________________________________________________________________________________________
layer_normalization_85 (LayerNo (None, None, 128)    256         layer_dropout_84[0][0]           
__________________________________________________________________________________________________
layer_dropout_85 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_13[2][0]        
                                                                 layer_dropout_84[0][0]           
__________________________________________________________________________________________________
layer_normalization_86 (LayerNo (None, None, 128)    256         layer_dropout_85[0][0]           
__________________________________________________________________________________________________
dropout_67 (Dropout)            (None, None, 128)    0           layer_normalization_86[0][0]     
__________________________________________________________________________________________________
layer_dropout_86 (LayerDropout) (None, None, 128)    0           attention_5[2][0]                
                                                                 layer_dropout_85[0][0]           
__________________________________________________________________________________________________
layer_normalization_87 (LayerNo (None, None, 128)    256         layer_dropout_86[0][0]           
__________________________________________________________________________________________________
dropout_68 (Dropout)            (None, None, 128)    0           layer_normalization_87[0][0]     
__________________________________________________________________________________________________
layer_dropout_87 (LayerDropout) (None, None, 128)    0           conv1d_24[2][0]                  
                                                                 layer_dropout_86[0][0]           
__________________________________________________________________________________________________
position__embedding_21 (Positio (None, None, 128)    0           layer_dropout_87[0][0]           
__________________________________________________________________________________________________
layer_normalization_88 (LayerNo (None, None, 128)    256         position__embedding_21[0][0]     
__________________________________________________________________________________________________
dropout_69 (Dropout)            (None, None, 128)    0           layer_normalization_88[0][0]     
__________________________________________________________________________________________________
layer_dropout_88 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_14[2][0]        
                                                                 position__embedding_21[0][0]     
__________________________________________________________________________________________________
layer_normalization_89 (LayerNo (None, None, 128)    256         layer_dropout_88[0][0]           
__________________________________________________________________________________________________
layer_dropout_89 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_15[2][0]        
                                                                 layer_dropout_88[0][0]           
__________________________________________________________________________________________________
layer_normalization_90 (LayerNo (None, None, 128)    256         layer_dropout_89[0][0]           
__________________________________________________________________________________________________
dropout_70 (Dropout)            (None, None, 128)    0           layer_normalization_90[0][0]     
__________________________________________________________________________________________________
layer_dropout_90 (LayerDropout) (None, None, 128)    0           attention_6[2][0]                
                                                                 layer_dropout_89[0][0]           
__________________________________________________________________________________________________
layer_normalization_91 (LayerNo (None, None, 128)    256         layer_dropout_90[0][0]           
__________________________________________________________________________________________________
dropout_71 (Dropout)            (None, None, 128)    0           layer_normalization_91[0][0]     
__________________________________________________________________________________________________
layer_dropout_91 (LayerDropout) (None, None, 128)    0           conv1d_28[2][0]                  
                                                                 layer_dropout_90[0][0]           
__________________________________________________________________________________________________
position__embedding_22 (Positio (None, None, 128)    0           layer_dropout_91[0][0]           
__________________________________________________________________________________________________
layer_normalization_92 (LayerNo (None, None, 128)    256         position__embedding_22[0][0]     
__________________________________________________________________________________________________
dropout_72 (Dropout)            (None, None, 128)    0           layer_normalization_92[0][0]     
__________________________________________________________________________________________________
layer_dropout_92 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_16[2][0]        
                                                                 position__embedding_22[0][0]     
__________________________________________________________________________________________________
layer_normalization_93 (LayerNo (None, None, 128)    256         layer_dropout_92[0][0]           
__________________________________________________________________________________________________
layer_dropout_93 (LayerDropout) (None, None, 128)    0           depthwise_conv1d_17[2][0]        
                                                                 layer_dropout_92[0][0]           
__________________________________________________________________________________________________
layer_normalization_94 (LayerNo (None, None, 128)    256         layer_dropout_93[0][0]           
__________________________________________________________________________________________________
dropout_73 (Dropout)            (None, None, 128)    0           layer_normalization_94[0][0]     
__________________________________________________________________________________________________
layer_dropout_94 (LayerDropout) (None, None, 128)    0           attention_7[2][0]                
                                                                 layer_dropout_93[0][0]           
__________________________________________________________________________________________________
layer_normalization_95 (LayerNo (None, None, 128)    256         layer_dropout_94[0][0]           
__________________________________________________________________________________________________
dropout_74 (Dropout)            (None, None, 128)    0           layer_normalization_95[0][0]     
__________________________________________________________________________________________________
layer_dropout_95 (LayerDropout) (None, None, 128)    0           conv1d_32[2][0]                  
                                                                 layer_dropout_94[0][0]           
__________________________________________________________________________________________________
concatenate_2 (Concatenate)     (None, None, 256)    0           layer_dropout_39[0][0]           
                                                                 layer_dropout_67[0][0]           
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, None, 256)    0           layer_dropout_39[0][0]           
                                                                 layer_dropout_95[0][0]           
__________________________________________________________________________________________________
conv1d_33 (Conv1D)              (None, None, 1)      257         concatenate_2[0][0]              
__________________________________________________________________________________________________
conv1d_34 (Conv1D)              (None, None, 1)      257         concatenate_3[0][0]              
__________________________________________________________________________________________________
lambda_12 (Lambda)              (None, None)         0           conv1d_33[0][0]                  
__________________________________________________________________________________________________
lambda_14 (Lambda)              (None, None)         0           conv1d_34[0][0]                  
__________________________________________________________________________________________________
lambda_13 (Lambda)              (None, None)         0           lambda_12[0][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
lambda_15 (Lambda)              (None, None)         0           lambda_14[0][0]                  
                                                                 batch_slice_4[0][0]              
__________________________________________________________________________________________________
start (Lambda)                  (None, None)         0           lambda_13[0][0]                  
__________________________________________________________________________________________________
end (Lambda)                    (None, None)         0           lambda_15[0][0]                  
__________________________________________________________________________________________________
start_pos (LabelPadding)        (None, None)         0           start[0][0]                      
__________________________________________________________________________________________________
end_pos (LabelPadding)          (None, None)         0           end[0][0]                        
__________________________________________________________________________________________________
qa_output (QAoutputBlock)       [(None, 1), (None, 1 0           start[0][0]                      
                                                                 end[0][0]                 
