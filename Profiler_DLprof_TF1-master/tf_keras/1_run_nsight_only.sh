nsys profile -y 1 -d 20 -w true -t "cudnn,cuda,osrt,nvtx" -o ./nsight_results/ \
python $1.py \

