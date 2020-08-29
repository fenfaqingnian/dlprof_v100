dlprof --graphdef=auto --reports=summary,detail,iteration  \
-y=10 -d=120 --force=true \
--output_path=./results \
--iter_start 20 --iter_stop 80 \
--profile_name='simple_tf1_mlp' \
python tf_simple_mlp.py \
--num_iter=1000 \
--batch_size=128  
