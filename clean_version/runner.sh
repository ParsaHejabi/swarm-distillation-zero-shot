#!/bin/bash
# Run the clean version of the T0 training pipeline on a single GPU without DeepSpeed.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(realpath "$SCRIPT_DIR")"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

export HF_HOME="$REPO_ROOT/../pretrain_models/huggingface"
cache_dir="$REPO_ROOT/../pretrain_models/huggingface"

# Uncomment to disable network access and rely on local files only
# export TRANSFORMERS_OFFLINE=1

export WANDB_MODE=online
export WANDB_PROJECT="swarm-distillation"
export WANDB_WATCH="false"

DATE=$(date +%Y%m%d)

# Dataset selection; modify dname to switch tasks
# (only a subset is shown here for brevity)
dname="rte"

datasets=(wsc winogrande anli_r1 anli_r2 anli_r3 cb rte copa hellaswag story_cloze wic)

# Default hyperparameters (can be adjusted)
ga=4
max_steps=1000
eval_steps=50
metric="accuracy"

case "${dname}" in
  rte)
    dataset="super_glue"; subset="rte"; testset_name="validation";;
  cb)
    dataset="super_glue"; subset="cb"; testset_name="validation";;
  anli_r1)
    dataset="anli"; subset="none"; testset_name="dev_r1";;
  anli_r2)
    dataset="anli"; subset="none"; testset_name="dev_r2";;
  anli_r3)
    dataset="anli"; subset="none"; testset_name="dev_r3";;
  wsc)
    dataset="super_glue"; subset="wsc.fixed"; testset_name="validation";;
  winogrande)
    dataset="winogrande"; subset="winogrande_xl"; testset_name="validation";;
  copa)
    dataset="super_glue"; subset="copa"; testset_name="validation";;
  hellaswag)
    dataset="hellaswag"; subset="none"; testset_name="validation";;
  story_cloze)
    dataset="story_cloze"; subset="2016"; testset_name="validation";;
  wic)
    dataset="super_glue"; subset="wic"; testset_name="validation";;
  *)
    echo "wrong dataset name!"; exit 1;;
esac

seed=42
bsz=1
nprompts=5
eval_bsz=100

peft="lora"
pL=1
lora_pos="encdec"
lora_dropout=0.3
lora_alpha=4

lr=4e-5
lr_scheduler_type="polynomial"
max_epochs=50
log_steps=10
debugsize=-1
max_dev_size=1000

temp=1.0
copt="uniform"

save_steps=50
save_total_limit=20

# Swarm distillation configuration
test_mode="ttt_t0"
train_data="validation"
train_size=10000
model="T0_3B"
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

exp_name=3B_${test_mode}.train.source.${train_data}.${dataset}.${subset}.${testset_name}.${model}.peft.${peft}.lora_alpha${lora_alpha}.lora_drop${lora_dropout}.bn${pL}.pw${pseudo_weight}.np${nprompts}.bsz${bsz}.ga${ga}.lr${lr}.steps.${max_steps}
SAVE="${REPO_ROOT}/checkpoints/${dname}/${exp_name}_${DATE}"
rm -rf "${SAVE}"
mkdir -p "${SAVE}"
cp "$0" "${SAVE}/run.sh"

# Launch the script on a single GPU
CUDA_VISIBLE_DEVICES=0 python -u "${REPO_ROOT}/run_t0.py" \
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
  --learning_rate ${lr} --eval_strategy "steps" --eval_steps ${eval_steps} \
  --disable_eval_mode ${disable_eval_mode} --pseudo_target_mode ${pseudo_target_mode} --ensemble_subset_size ${ensemble_subset_size} \
  --loss_option ${loss_opt} --jsd ${jsd} --detach_kl_left ${detach_kl_left} --detach_kl_right ${detach_kl_right} \
  --ensemble_option ${ensemble} --pseudo_train_loss_weight ${pseudo_weight} --pseudo_dist ${pseudo_dist} \
  --lora_dropout ${lora_dropout} --lora_alpha ${lora_alpha} --lora_pos ${lora_pos} \
  --prob_temperature ${temp} --combine_option ${copt} \
  --train_random_n_prompts ${nprompts} --train_data_source ${train_data} \
  --save_strategy 'steps' --save_steps ${save_steps} --save_total_limit ${save_total_limit} \
  --warmup_steps 100 --gradient_accumulation_steps ${ga} \
  --gradient_checkpointing True \
  --eval_accumulation_steps 1 \
  --eval_on_start True \
  --lr_scheduler_type ${lr_scheduler_type} \
  --output_dir ${SAVE} --overwrite_output_dir --report_to "wandb" \
  --bf16 \
  --disable_tqdm "True" 2>&1 | tee ${SAVE}/log.txt