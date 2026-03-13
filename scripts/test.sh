cd ../
export PYTHONPATH=../../MIL_BASELINE:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

config=configs/gleason/h-optimus-1/CLAM_MB_MIL.yaml
test_dataset_csv=datasets/gleason/dataset_001/5fold/Total_5-fold_5fold_1fold.csv
base_dir=results/gleason/dataset_001/5fold/CLAM_MB_MIL/time_2026-01-16-17-35_5fold_CLAM_MB_MIL_seed_2024
gpu=0
#cd ../

for fold in {1..5}; do
    model_weight_path="${base_dir}/fold_${fold}/Best_EPOCH_"*.pth

    if [ ! -f ${model_weight_path} ]; then
        echo "警告: 在 fold_${fold} 未找到 Best_EPOCH_*.pth 文件，跳过。"
        continue
    fi

    test_log_dir="${base_dir}/best-1/fold_${fold}"

    mkdir -p ${test_log_dir}

    echo "正在测试 fold_${fold}，模型: ${model_weight_path}"
    CUDA_VISIBLE_DEVICES=$gpu python test_mil.py --yaml_path ${config} --test_dataset_csv ${test_dataset_csv} --model_weight_path ${model_weight_path} --test_log_dir ${test_log_dir}

    if [ $? -eq 0 ]; then
        echo "fold_${fold} 测试完成。"
    else
        echo "错误: fold_${fold} 测试失败！"
    fi
done
echo "所有fold测试完毕。"

for fold in {1..5}; do
    model_weight_path="${base_dir}/fold_${fold}/Last_EPOCH_"*.pth

    if [ ! -f ${model_weight_path} ]; then
        echo "警告: 在 fold_${fold} 未找到 Last_EPOCH_*.pth 文件，跳过。"
        continue
    fi

    test_log_dir="${base_dir}/last-1/fold_${fold}"

    mkdir -p ${test_log_dir}

    echo "正在测试 fold_${fold}，模型: ${model_weight_path}"
    CUDA_VISIBLE_DEVICES=$gpu python test_mil.py --yaml_path ${config} --test_dataset_csv ${test_dataset_csv} --model_weight_path ${model_weight_path} --test_log_dir ${test_log_dir}

    if [ $? -eq 0 ]; then
        echo "fold_${fold} 测试完成。"
    else
        echo "错误: fold_${fold} 测试失败！"
    fi
done
echo "所有fold测试完毕。"

