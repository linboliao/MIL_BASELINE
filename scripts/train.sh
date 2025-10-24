export PYTHONPATH=../../MIL_BASELINE:$PYTHONPATH
export LD_LIBRARY_PATH=/data12/jing/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

#python split_scripts/split_datasets_k_fold_train_val_test.py --seed 42 --csv_path datasets/p-dMMR/data.csv --save_dir datasets/p-dMMR --k 5 --val_ratio 0.2 --dataset_name cls
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/AB_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/CLAM_MB_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/CLAM_SB_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/DS_MIL.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/TRANS_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/DGR_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/DTFD_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/FR_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/ILRA_MIL.yaml

