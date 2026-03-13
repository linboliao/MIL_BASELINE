cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/data12/jing/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1
python split_scripts/split_datasets_k_fold_train_val_test.py --seed 42 --csv_path 'datasets/gleason/dataset_005/dataset.csv' --dataset_name 5fold --k 5 --save_dir 'datasets/gleason/dataset_005'
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path 'configs/gleason/h-optimus-1/CLAM_MB_MIL-5cls.yaml'
#CUDA_VISIBLE_DEVICES=1 python train_mil.py --yaml_path 'configs/gleason/h-optimus-1/CLAM_MB_MIL-2cls.yaml'
#CUDA_VISIBLE_DEVICES=1 python train_mil.py --yaml_path 'configs/gleason/h-optimus-1/CLAM_MB_MIL-4cls.yaml'
#CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/CLAM_MB_MIL.yaml
#CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/CLAM_SB_MIL.yaml
#CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/DS_MIL.yaml
#CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/xiehe/TRANS_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/DGR_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/DTFD_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/FR_MIL.yaml
#python train_mil.py --yaml_path configs/xiehe/ILRA_MIL.yaml
