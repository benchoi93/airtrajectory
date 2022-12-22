# CUDA_VISIBLE_DEVICES=0 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 1 >> out1.log &
# CUDA_VISIBLE_DEVICES=1 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 2 >> out2.log &
# CUDA_VISIBLE_DEVICES=2 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 3 >> out3.log &

# CUDA_VISIBLE_DEVICES=0 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 7 >> out1.log &
# CUDA_VISIBLE_DEVICES=1 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 10 >> out2.log &
# CUDA_VISIBLE_DEVICES=2 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 15 >> out3.log &

# CUDA_VISIBLE_DEVICES=0 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 20 >> out1.log &
# CUDA_VISIBLE_DEVICES=1 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 30 >> out2.log &
# CUDA_VISIBLE_DEVICES=2 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 50 >> out3.log &

# CUDA_VISIBLE_DEVICES=0 nohup python main_MDN.py --hidden 64 --encoding_type absdev --n_components 100 >> out1.log &



CUDA_VISIBLE_DEVICES=0 nohup python main_flomo.py --hidden 64 --encoding_type absdev >> out3.log &
CUDA_VISIBLE_DEVICES=1 nohup python main_flomo.py --hidden 64 --encoding_type abs >> out2.log &
CUDA_VISIBLE_DEVICES=2 nohup python main_flomo.py --hidden 64 --encoding_type dev >> out1.log &

