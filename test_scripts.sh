python run.py setup.n_gpus_per_node=4 train.cut_noise=20 -m DP-FETA -dn mnist_28 -e 10.0

python run.py setup.n_gpus_per_node=4 train.initial_sample=/p/fzv6enresearch/FETA-Pro/exp/dp-feta-pro/mnist_28_eps1.0val_default-2025-08-03-11-06-21/gen/gen.npz -m PE -dn mnist_28 -e 10.0

python run.py setup.n_gpus_per_node=3 -m PrivImage -dn covidx_32 -e 10.0 -ed default_pretraing
python run.py setup.n_gpus_per_node=4 public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/conditional_pretrain_imagenet_32_1000cond/pretrain/checkpoints/final_checkpoint.pth -m PDP-Diffusion -dn covidx_32 -e 10.0 -ed default_pretraing

CUDA_VISIBLE_DEVICES=1 python eval.py -dn cifar10_32 -ep /p/fzv6enresearch/gap/exp/lora-cifar10-eps10

# DP metrics
python run_metric.py -m DPFID -pm stable-diffusion-2-1-base -sd cifar10 --non_DP

python run_metric.py -m DPGAP -pm stable-diffusion-2-1-base -sd cifar10 
python run_metric.py -m DPGAP -pm dpimagebench-ldm -sd celeba

python run_metric.py -m PE-Select -pm stable-diffusion-2-1-base -sd cifar10

# ln -s /bigtemp/fzv6en/gap_data/exp exp

python run.py setup.n_gpus_per_node=3 --method PE-SGD --data_name mnist_28 --epsilon 10.0 eval.mode=val pretrain.mode=time_freq train.pe_freq=[10,20,30,40,50,60] train.contrastive_batch_size=256 train.contrastive_n_epochs=5 train.contrastive_num_samples=1000 train.contrastive_selection_ratio=0.1 -ed pe60

python run.py setup.n_gpus_per_node=4 --method PE-SGD --data_name mnist_28 --epsilon 10.0 train.pe_freq=[] eval.mode=val pretrain.mode=time_freq gen.pe_last=true gen.selection_ratio=0.5 -ed pe_last0.5_new

python run.py setup.n_gpus_per_node=3 --method PE-SGD --data_name mnist_28 --epsilon 10.0 eval.mode=val pretrain.mode=time_freq train.pe_freq=[10,20,30,40,50,60] train.contrastive_batch_size=256 train.contrastive_n_epochs=5 train.contrastive=v1 train.contrastive_alpha=1.0 train.contrastive_num_samples=1000 train.contrastive_selection_ratio=0.1 -ed pe60_contrastive_v1

python run.py setup.n_gpus_per_node=3 --method PE-SGD --data_name mnist_28 --epsilon 10.0 eval.mode=val pretrain.mode=time_freq train.pe_freq=[10,20,30,40,50,60] train.contrastive_batch_size=256 train.contrastive_n_epochs=5 train.contrastive=v2 train.contrastive_alpha=1.0 train.contrastive_num_samples=1000 train.contrastive_selection_ratio=0.1 -ed pe60_contrastive_v2

# variation
python run.py setup.n_gpus_per_node=4 --method PE-SGD --data_name mnist_28 --epsilon 10.0 eval.mode=val pretrain.mode=time_freq train.pe_variation=true

# Noise fixed-SGD   
python run.py setup.n_gpus_per_node=4 --method Noise-SGD --data_name mnist_28 --epsilon 1.0 eval.mode=val train.contrastive_selection_ratio=0.2 train.contrastive_num_samples=60000 pretrain.mode=time_freq model.noise_num=60000 pretrain.n_epochs_time=1 train.pe_freq=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150] train.contrastive_n_epochs=5
