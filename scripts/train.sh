cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

#python split_scripts/split_datasets_k_fold_train_val_test.py --seed 42 --csv_path datasets/Gastric/data.csv --save_dir datasets/Gastric --dataset_name UNI --k 5 --val_ratio 0.2
#CUDA_VISIBLE_DEVICES=2 python train_mil.py --yaml_path configs/Contrast/MIL/CDP_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Contrast/MIL/CLAM_SB_MIL.yaml
#CUDA_VISIBLE_DEVICES=1 python train_mil.py --yaml_path configs/Contrast/MIL/GATE_AB_MIL.yaml

