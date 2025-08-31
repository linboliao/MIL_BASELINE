export PYTHONPATH=../../MIL_BASELINE:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
cd ../

python test_mil.py --yaml_path configs/cancer/AB_MIL.yaml --test_dataset_csv datasets/cancer/test.csv \
--model_weight_path result/cancer/0829/AB_MIL/time_2025-08-29-14-36_0829_AB_MIL_seed_42/fold_1/Best_EPOCH_3.pth \
--test_log_dir result/cancer/test/DS_MIL/fold_1