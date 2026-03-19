cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

#python split_scripts/split_datasets_k_fold_train_val.py --seed 42 --csv_path datasets/ProstateCls/train_val.csv --save_dir datasets/ProstateCls --dataset_name UNI --k 5 #--val_ratio 0.2
CUDA_VISIBLE_DEVICES=1 python train_mil.py --yaml_path configs/ProstateCls/ABMIL/UNI2.yaml

