CUDA_VISIBLE_DEVICES=0 nohup python main_flomo.py --hidden 16 --encoding_type dev >> out1.log &
CUDA_VISIBLE_DEVICES=1 nohup python main_flomo.py --hidden 16 --encoding_type abs >> out2.log &
CUDA_VISIBLE_DEVICES=2 nohup python main_flomo.py --hidden 16 --encoding_type absdev >> out3.log &

CUDA_VISIBLE_DEVICES=0 nohup python main_flomo.py --hidden 32 --encoding_type dev >> out4.log &
CUDA_VISIBLE_DEVICES=1 nohup python main_flomo.py --hidden 32 --encoding_type abs >> out5.log &
CUDA_VISIBLE_DEVICES=2 nohup python main_flomo.py --hidden 32 --encoding_type absdev >> out6.log &

CUDA_VISIBLE_DEVICES=0 nohup python main_flomo.py --hidden 64 --encoding_type dev >> out7.log &
CUDA_VISIBLE_DEVICES=1 nohup python main_flomo.py --hidden 64 --encoding_type abs >> out8.log &
CUDA_VISIBLE_DEVICES=2 nohup python main_flomo.py --hidden 64 --encoding_type absdev >> out9.log &
