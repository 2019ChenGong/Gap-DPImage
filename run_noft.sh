export HF_HOME='/bigtemp/fzv6en/diffuser_cache'
cd /p/fzv6enresearch/gap/dm-lora/

subjects="octmnist_128" # Subject Name
# subjects="camelyon_96" # Subject Name
# subjects="celeba_male_256" # Subject Name
model_resolution=512
sensitive_resolution=128

lower_name="med"
MODEL_NAME="Nihirc/Prompt2MedImage"


python generate_sd_bench_ori.py --batch_size 30 --data_name $subjects --output_dir "../exp/"$lower_name"_${subjects}_noft" --num 60000 --target_size $sensitive_resolution --model_id $MODEL_NAME --gen_size $model_resolution

lower_name="realv6"
MODEL_NAME="SG161222/Realistic_Vision_V6.0_B1_noVAE"


python generate_sd_bench_ori.py --batch_size 30 --data_name $subjects --output_dir "../exp/"$lower_name"_${subjects}_noft" --num 60000 --target_size $sensitive_resolution --model_id $MODEL_NAME --gen_size $model_resolution

lower_name="v2-1-base"
MODEL_NAME="Manojb/stable-diffusion-2-1-base"


python generate_sd_bench_ori.py --batch_size 30 --data_name $subjects --output_dir "../exp/"$lower_name"_${subjects}_noft" --num 60000 --target_size $sensitive_resolution --model_id $MODEL_NAME --gen_size $model_resolution

lower_name="base"
MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5" # Model card


python generate_sd_bench_ori.py --batch_size 30 --data_name $subjects --output_dir "../exp/"$lower_name"_${subjects}_noft" --num 60000 --target_size $sensitive_resolution --model_id $MODEL_NAME --gen_size $model_resolution