EXP='RSR'
PORT='4325'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT basicsr/train_robust.py -opt 'options/train/'$EXP'.yml' --launcher pytorch 
