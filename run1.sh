CUDA_VISIBLE_DEVICES=0 python eval.py -m PE -dn octmnist_128 -ep /p/fzv6enresearch/gap/exp/base_octmnist_128_noft &
CUDA_VISIBLE_DEVICES=1 python eval.py -m PE -dn octmnist_128 -ep /p/fzv6enresearch/gap/exp/med_octmnist_128_noft &
CUDA_VISIBLE_DEVICES=2 python eval.py -m PE -dn octmnist_128 -ep /p/fzv6enresearch/gap/exp/realv6_octmnist_128_noft &
CUDA_VISIBLE_DEVICES=3 python eval.py -m PE -dn octmnist_128 -ep /p/fzv6enresearch/gap/exp/v2-1-base_octmnist_128_noft 