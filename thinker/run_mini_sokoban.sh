#!/bin/bash

. /etc/profile.d/modules.sh                
module purge                               
module load rhel8/default-amp              
module load cuda/11.1 python-3.9.6-gcc-5.4.0-sbr552h opencv-3.4.3-gcc-5.4.0-o3bzatj gcc/8 patchelf-0.9-gcc-5.4.0-v25fnm7 glew-2.0.0-gcc-5.4.0-rja5lhn
source ~/rds/hpc-work/planning/env/bin/activate
ulimit -u 127590
wandb login 766392e4b9baf40310d5b3ae093abd61b5a3973a
export WANDB_USER=28tbush

application="python train.py \
	--xpid drc_test_014 \
	--drc true \
	--actor_unroll_len 20 \
	--reg_cost 1 \
	--actor_learning_rate 4e-4 \
	--entropy_cost 1e-2 \
	--v_trace_lamb 0.97 \
	--actor_adam_eps 1e-4 \
	--has_model false \
	--use_wandb true \
	--total_steps 5000000 \
	--wandb_ckp_freq 1000000 \
	--mini true" 


CMD="$application $options"

echo "Time: `date`"
echo "Current directory: `pwd`"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD 
