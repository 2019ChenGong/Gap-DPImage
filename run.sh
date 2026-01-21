
# python run_metric.py -m PE-Select -sd mnist
# python run_metric.py -m PE-Select -sd cifar10
# python run_metric.py -m PE-Select -sd octmnist
# python run_metric.py -m PE-Select -sd celeba_male
# python run_metric.py -m PE-Select -sd camelyon
export CUDA_VISIBLE_DEVICES=0
export HF_HOME='/bigtemp/fzv6en/diffuser_cache'
cd /p/fzv6enresearch/gap/dm-lora/
bash scripts/finetune_sd_cifar10.sh