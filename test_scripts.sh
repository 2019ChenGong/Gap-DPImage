python run.py setup.n_gpus_per_node=4 train.cut_noise=20 -m DP-FETA -dn mnist_28 -e 10.0

python run.py setup.n_gpus_per_node=4 train.initial_sample=/p/fzv6enresearch/FETA-Pro/exp/dp-feta-pro/mnist_28_eps1.0val_default-2025-08-03-11-06-21/gen/gen.npz -m PE -dn mnist_28 -e 10.0

python run.py setup.n_gpus_per_node=3 -m PrivImage -dn covidx_32 -e 10.0 -ed default_pretraing
python run.py setup.n_gpus_per_node=4 public_data.name=null model.ckpt=/p/fzv6enresearch/DPImageBench/exp/pdp-diffusion/conditional_pretrain_imagenet_32_1000cond/pretrain/checkpoints/final_checkpoint.pth -m PDP-Diffusion -dn covidx_32 -e 10.0 -ed default_pretraing

CUDA_VISIBLE_DEVICES=1 python eval.py -dn cifar10_32 -ep /p/fzv6enresearch/gap/exp/lora-cifar10-eps10

# ln -s /bigtemp/fzv6en/gap_data/exp exp