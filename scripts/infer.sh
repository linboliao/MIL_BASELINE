cd ../

export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

config=configs/cancer/AB_MIL-h-optimus-1.yaml
test_dataset_csv=datasets/cancer/0923/h-optimus-1_test.csv
model_weight_path=result/cancer/h-optimus-1/AB_MIL/seed_42_2025-09-24-06-38/fold_1/Best_EPOCH_2.pth
test_log_dir=result/cancer/h-optimus-1/AB_MIL/seed_42_2025-09-24-06-38/infer
CUDA_VISIBLE_DEVICES=1 python infer_mil.py --yaml_path $config --test_dataset_csv $test_dataset_csv --model_weight_path $model_weight_path --test_log_dir $test_log_dir