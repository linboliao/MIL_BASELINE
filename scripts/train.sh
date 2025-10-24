export PYTHONPATH=../../MIL_BASELINE:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
cd ../

python split_scripts/split_datasets_k_fold_train_val_test.py --seed 42 --csv_path datasets/gleason/0_448/radical.csv --save_dir datasets/gleason/0_448 --dataset_name radical --k 5 --val_ratio 0.2
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/gleason/CLAM_MB_MIL-h-optimus-1.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/gleason/CLAM_SB_MIL-h-optimus-1.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/gleason/AB_MIL-h-optimus-1.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/gleason/DS_MIL-h-optimus-1.yaml
CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/gleason/TRANS_MIL-h-optimus-1.yaml
