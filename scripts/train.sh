export PYTHONPATH=../../MIL_BASELINE:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
cd ../
python split_scripts/split_datasets_k_fold_train_val_test.py --seed 42 --csv_path datasets/cancer/0829.csv --save_dir datasets/cancer --dataset_name 0829 --k 5 --val_ratio 0.2
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/cancer/AB_MIL.yaml
#CUDA_VISIBLE_DEVICES=1 python train_mil.py --yaml_path configs/cancer/CLAM_MB_MIL.yaml
#CUDA_VISIBLE_DEVICES=2 python train_mil.py --yaml_path configs/cancer/CLAM_SB_MIL.yaml
#CUDA_VISIBLE_DEVICES=2 python train_mil.py --yaml_path configs/cancer/DS_MIL.yaml
CUDA_VISIBLE_DEVICES=3 python train_mil.py --yaml_path configs/cancer/TRANS_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/DGR_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/DTFD_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/FR_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/ILRA_MIL.yaml

