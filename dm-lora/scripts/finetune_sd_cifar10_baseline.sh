subjects="cifar10_32" # Subject Name
data_path="/p/fzv6enresearch/gap/dataset/cifar10/train_32.zip"
model_resolution=256
sensitive_resolution=32
batch_size_per_gpu=85
gradient_accumulation_steps=16
lower_name="base"
eps=10
# teapot subject images are available at dataset link provided in the README
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5" # Model card
OUTPUT_DIR="../exp/lora_${subjects}_4096bs_100steps_eps${eps}" # Where to save the model


#------------------------------------------------------------------------------------
#                                    Hyperparameters
#------------------------------------------------------------------------------------
attn_update_unet="kqvo"

lr=5e-4
steps=100


accelerate launch train_dreambooth_lora.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --instance_data_dir=$subjects \
    --bench_path=$data_path \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision="fp16" \
    --instance_prompt="" \
    --validation_prompt="An image of an airplane." \
    --resolution=$model_resolution \
    --train_batch_size=$batch_size_per_gpu \
    --gradient_accumulation_steps=$gradient_accumulation_steps \
    --learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=$steps \
    --total_steps=$steps \
    --adapter_type="lora" \
    --seed="0" \
    --diffusion_model=$lower_name \
    --use_8bit_adam \
    --enable_xformers_memory_efficient_attention \
    --attn_update_unet=$attn_update_unet \
    --checkpointing_steps 100 \
    --dataloader_num_workers 0 \
    --unet_lora_rank_k 4 \
    --unet_lora_rank_v 4 \
    --unet_lora_rank_q 4 \
    --unet_lora_rank_out 4 \
    --micro_batch_size 1 \
    --eps $eps


python generate_sd_bench.py --batch_size 30 --data_name $subjects --output_dir $OUTPUT_DIR"/lora_k4q4v4o4_"$lower_name"_0.0005" --num_per_cls 6000 --target_size $sensitive_resolution --model_id $MODEL_NAME --gen_size $model_resolution