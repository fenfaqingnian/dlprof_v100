dlprof --graphdef=auto --reports=summary,detail,iteration  \
-y=60 -d=120 --force=true \
--key_node='cluster_7_1/xla_run' \
--output_path=./results3 \
--iter_start 40 --iter_stop 80 \
--profile_name='tf_keras_v2' \
python tf_keras_v2.py
