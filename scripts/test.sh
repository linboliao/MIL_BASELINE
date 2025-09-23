export PYTHONPATH=../../MIL_BASELINE:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

config=configs/cancer/AB_MIL-uni2.yaml
test_dataset_csv=datasets/cancer/uni2_test.csv
base_dir=result/cancer/uni2/AB_MIL/seed_42_2025-09-23-09-47
gpu=2
cd ../

for fold in {1..5}; do
    model_weight_path="${base_dir}/fold_${fold}/Best_EPOCH_"*.pth

    if [ ! -f ${model_weight_path} ]; then
        echo "警告: 在 fold_${fold} 未找到 Best_EPOCH_*.pth 文件，跳过。"
        continue
    fi

    test_log_dir="${base_dir}/best/fold_${fold}"

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

    test_log_dir="${base_dir}/last/fold_${fold}"

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

