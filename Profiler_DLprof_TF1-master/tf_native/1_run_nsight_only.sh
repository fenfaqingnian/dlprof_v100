nsys profile -y 1 -d 20 -w true -t "cudnn,cuda,osrt,nvtx" -o ./nsight_results/ \
python tf_simple_mlp.py \
--num_iter=1000 \
--batch_size=128  

