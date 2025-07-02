#!/bin/bash
# Example script to run T0 experiments locally without DeepSpeed for easier debugging.

# Resolve the repository root so the script works when executed from any directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR/..")"

# Ensure local modules (e.g., the `ttt` package) are found
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

export TRANSFORMERS_CACHE="$REPO_ROOT/pretrain_models/huggingface"
export HF_DATASETS_CACHE="$REPO_ROOT/pretrain_models/huggingface"
export HF_METRICS_CACHE="$REPO_ROOT/pretrain_models/huggingface"
cache_dir="$REPO_ROOT/pretrain_models/huggingface"

# Allow Transformers to download models if they are not present locally.
# Uncomment the next line to disable network access and run strictly offline.
# export TRANSFORMERS_OFFLINE=1
# Enable wandb logging
export WANDB_MODE=online

# wandb env variables
export WANDB_PROJECT="swarm-distillation"
export WANDB_WATCH="false"

# Avoid protobuf runtime errors when using precompiled tokenizer protobufs.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Enable verbose NCCL logging and disable peer-to-peer access to help diagnose or avoid NCCL collectives hanging on some systems.
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1

export TOKENIZERS_PARALLELISM="false"
DATE=`date +%Y%m%d`

# Choose dataset name here, e.g., rte, cb, anli_r1 ...
dname="rte"

datasets=(wsc winogrande anli_r1 anli_r2 anli_r3 cb rte copa hellaswag story_cloze wic)

# Default hyperparameters (can be adjusted)
ga=4
max_steps=1000
eval_steps=50
metric="accuracy"

if [ ${dname} = "rte" ]; then
  dataset="super_glue"
  subset="rte"
  testset_name="validation"
elif [ ${dname} = "cb" ]; then
  dataset="super_glue"
  subset="cb"
  testset_name="validation"
# Add other datasets if needed
else
  echo "wrong dataset name!"
  exit
fi

seed=42
bsz=1
nprompts=5
eval_bsz=100

peft="lora"
pL=1
lora_pos="encdec"
lora_dropout=0.3
lora_alpha=4

lr=2e-5
lr_scheduler_type="polynomial"
max_epochs=50
log_steps=10
debugsize=-1
max_dev_size=1000

temp=1.0
copt="uniform"

# Save checkpoints every 10 steps and keep only the most recent 10
save_steps=10

test_mode="ttt_t0"
train_data="validation"
train_size=10000
model="T0pp"
loss_opt='consistency'
jsd=0
detach_kl_left=1
detach_kl_right=0
ensemble='avg_prob'
pseudo_weight=1.0
pseudo_dist="smooth"

disable_eval_mode=0
pseudo_target_mode="pairwise"
ensemble_subset_size=0.0

exp_name=11B_${test_mode}.train.source.${train_data}.${dataset}.${subset}.${testset_name}.${model}.peft.${peft}.lora_alpha${lora_alpha}.lora_drop${lora_dropout}.bn${pL}.pw${pseudo_weight}.np${nprompts}.bsz${bsz}.ga${ga}.lr${lr}.steps.${max_steps}
SAVE="$REPO_ROOT/checkpoints/${dname}/${exp_name}_${DATE}"
rm -rf "${SAVE}"; mkdir -p "${SAVE}"
cp "$0" "${SAVE}/run.sh"

# Launch without DeepSpeed on a single GPU
python -u "$REPO_ROOT/examples/pytorch/t0-zero-shot/run_t0.py" \
  --dataset_name ${dataset} --subset_name ${subset} --prompt_set_name ${dataset} --testset_name ${testset_name} \
  --model_name_or_path ${model} --per_device_train_batch_size ${bsz} --per_device_eval_batch_size ${eval_bsz} \
  --test_mode ${test_mode} --cache_dir ${cache_dir} --metric_name ${metric} \
  --debug_size ${debugsize} \
  --peft_option ${peft} --bottleneck_dim ${pL} \
  --do_train --logging_steps ${log_steps} --num_train_epochs ${max_epochs} --max_steps ${max_steps} \
  --adam_beta1 0.9 \
  --adam_beta2 0.98 \
  --adam_epsilon 1e-6 \
  --seed ${seed} --debug_size ${train_size} --max_dev_size ${max_dev_size} \
  --learning_rate ${lr} --evaluation_strategy "steps" --eval_steps ${eval_steps} \
  --disable_eval_mode ${disable_eval_mode} --pseudo_target_mode ${pseudo_target_mode} --ensemble_subset_size ${ensemble_subset_size} \
  --loss_option ${loss_opt} --jsd ${jsd} --detach_kl_left ${detach_kl_left} --detach_kl_right ${detach_kl_right} \
  --ensemble_option ${ensemble}  --pseudo_train_loss_weight ${pseudo_weight} --pseudo_dist ${pseudo_dist} \
  --lora_dropout ${lora_dropout} --lora_alpha ${lora_alpha} --lora_pos ${lora_pos} \
  --prob_temperature ${temp} --combine_option ${copt} \
  --train_random_n_prompts ${nprompts} --train_data_source ${train_data} \
  --save_strategy 'steps' --save_steps ${save_steps} --save_total_limit 10 \
  --warmup_steps 100 --gradient_accumulation_steps ${ga} \
  --lr_scheduler_type ${lr_scheduler_type} \
  --output_dir ${SAVE} --overwrite_output_dir --report_to "wandb" \
  --bf16 \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt
