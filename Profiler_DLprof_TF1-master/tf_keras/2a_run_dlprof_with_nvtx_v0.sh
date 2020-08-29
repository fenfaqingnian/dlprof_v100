dlprof --graphdef=auto --reports=summary,detail,iteration  \
-y=10 -d=120 --force=true \
--output_path=./results \
--iter_start 20 --iter_stop 80 \
--profile_name='tf_keras_v0' \
python tf_keras_v0.py