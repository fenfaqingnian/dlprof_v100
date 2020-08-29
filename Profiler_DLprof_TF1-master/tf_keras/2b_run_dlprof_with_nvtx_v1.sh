dlprof --graphdef=auto --reports=summary,detail,iteration  \
-y=10 -d=120 --force=true \
--key_node='AssignVariableOp_1' \
--output_path=./results2 \
--iter_start 20 --iter_stop 60 \
--profile_name='tf_keras_v1' \
python tf_keras_v1.py
