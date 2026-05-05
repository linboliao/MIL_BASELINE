cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

#python split_scripts/split_datasets_k_fold_train_val.py --seed 42 --csv_path datasets/Contrast/PFM/train_val_f.csv --save_dir datasets/Contrast/PFM --dataset_name Virchow2 --k 5 #--val_ratio 0.2
CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Contrast/FPM/omiclip.yaml
#CUDA_VISIBLE_DEVICES=2 python train_mil.py --yaml_path configs/Contrast/FPM/mstar.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Contrast/MIL/TRANS_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Contrast/MIL/CLAM_MB_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Contrast/MIL/CLAM_SB_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Contrast/MIL/WIKG_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Contrast/MIL/MAMBA_MIL.yaml
#CUDA_VISIBLE_DEVICES=1 python train_mil.py --yaml_path configs/Contrast/MIL/MAMBA2D_MIL.yaml
#CUDA_VISIBLE_DEVICES=2 python train_mil.py --yaml_path configs/Contrast/MIL/AEM_MIL.yaml
#CUDA_VISIBLE_DEVICES=3 python train_mil.py --yaml_path configs/Contrast/MIL/MICO_MIL.yaml
#CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/Contrast/MIL/MICRO_MIL.yaml
#CUDA_VISIBLE_DEVICES=5 python train_mil.py --yaml_path configs/Contrast/MIL/MSM_MIL.yaml
#CUDA_VISIBLE_DEVICES=6 python train_mil.py --yaml_path configs/Contrast/MIL/TDA_MIL.yaml
#CUDA_VISIBLE_DEVICES=7 python train_mil.py --yaml_path configs/Contrast/MIL/GDF_MIL.yaml
