ADV_ESTIMATOR=reinforce
LOG_AGG_MODE="token-mean"
PROJECT_NAME="vlr"
REWARD_CORRECT_MAX=1.0
REWARD_CORRECT_MIN=1.0
REWARD_FORMAT_ONLY=0.0
REWARD_WRONG=0.0
SCORE_TYPE=compute_score_vanilla
FILTER_TRUNCATED=False
EXP_NAME=${ADV_ESTIMATOR}_${LOG_AGG_MODE}_${REWARD_CORRECT_MAX}_${REWARD_CORRECT_MIN}_${REWARD_FORMAT_ONLY}_${REWARD_WRONG}_${SCORE_TYPE}_${FILTER_TRUNCATED}
DUMP_NAME=$PWD/dump/${EXP_NAME}_$(date '+%Y%m%d_%H%M%S').ndjson

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$ADV_ESTIMATOR \
    data.train_files=$PWD/data/math/train.parquet \
    data.val_files=$PWD/data/math/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1682 \
    data.max_response_length=1500 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.loss_agg_mode=$LOG_AGG_MODE \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=327680 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    custom_reward_function.path=$PWD/vlr_reward_fn.py \
    custom_reward_function.name=$SCORE_TYPE \
    custom_val_reward_function.path=$PWD/vlr_reward_fn.py \
    custom_val_reward_function.name=val_compute_score \
    reward_model.reward_manager=rich \
    +reward_model.reward_kwargs.correct_max=$REWARD_CORRECT_MAX \
    +reward_model.reward_kwargs.correct_min=$REWARD_CORRECT_MIN \
    +reward_model.reward_kwargs.format_only=$REWARD_FORMAT_ONLY \
    +reward_model.reward_kwargs.wrong=$REWARD_WRONG \
    +reward_model.reward_kwargs.filter_truncated=$FILTER_TRUNCATED \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.dump_path=$DUMP_NAME \
    trainer.logger=['console'] \
    trainer.n_gpus_per_node=2 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.default_local_dir=$PWD/checkpoints/${PROJECT_NAME}/${EXP_NAME} \
    trainer.total_epochs=1