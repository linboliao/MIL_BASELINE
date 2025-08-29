export PYTHONPATH=../../MIL_BASELINE:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
cd ../

python test_mil.py --yaml_path configs/xiehe/AB_MIL.yaml --test_dataset_csv datasets/xiehe/test.csv --model_weight_path logs/xiehe/AB_MIL/2025-08-14-16-32_seed_2024/fold_1/Best_EPOCH_91.pth --test_log_dir logs/gleason/test