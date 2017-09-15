BASE=./
python preprocess.py -train_src $BASE/src_train.txt -train_tgt $BASE/tgt_train.txt -valid_src $BASE/src_val.txt -valid_tgt $BASE/tgt_val.txt  -save_data $BASE/fixed2 -src_seq_length 1000 -tgt_seq_length 1000 -dynamic_dict -share_vocab
python train.py -data $BASE/fixed -save_model /tmp/model_name -layers 2 -rnn_size 600  -feature_vec_size 600 -word_vec_size 600 -batch_size 5  -epochs 20  -log_interval 100 -gpuid 1  -encoder_layer mean -copy_attn -dropout 0.3 -attention_type dot -truncated_decoder 100 -copy_attn_force 
python translate.py -model {model_name}  -src $BASE/src_val.txt  -gpu 1 -verbose -beam_size 30 -batch_size 1 -max_sent_length 500 -output /tmp/gen -n_best 10

