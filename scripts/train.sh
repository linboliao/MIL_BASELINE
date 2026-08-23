cd ../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

#python split_scripts/split_datasets_k_fold_train_val.py --seed 42 --csv_path datasets/Diagnosis/PFM/train_val_f.csv --save_dir datasets/Diagnosis/PFM --dataset_name Virchow2 --k 5 #--val_ratio 0.2
#CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/Diagnosis/PFM_Mix/CONCH.yaml
#CUDA_VISIBLE_DEVICES=5 python train_mil.py --yaml_path configs/Diagnosis/PFM_Mix/h-optimus-1.yaml
#CUDA_VISIBLE_DEVICES=6 python train_mil.py --yaml_path configs/Diagnosis/PFM_Mix/mstar.yaml
#CUDA_VISIBLE_DEVICES=7 python train_mil.py --yaml_path configs/Diagnosis/PFM_Mix/omiclip.yaml
#CUDA_VISIBLE_DEVICES=5 python train_mil.py --yaml_path configs/Diagnosis/PFM_Mix/UNI.yaml
#CUDA_VISIBLE_DEVICES=5 python train_mil.py --yaml_path configs/Diagnosis/PFM_Mix/UNI2.yaml
#CUDA_VISIBLE_DEVICES=6 python train_mil.py --yaml_path configs/Diagnosis/PFM_Mix/virchow2.yaml
#CUDA_VISIBLE_DEVICES=6 python train_mil.py --yaml_path configs/Diagnosis/Mag_Mix/5x.yaml
#CUDA_VISIBLE_DEVICES=6 python train_mil.py --yaml_path configs/Diagnosis/Mag_Mix/10x.yaml
#CUDA_VISIBLE_DEVICES=6 python train_mil.py --yaml_path configs/Diagnosis/Stains_Mix/Macenko.yaml
#CUDA_VISIBLE_DEVICES=7 python train_mil.py --yaml_path configs/Diagnosis/Stains_Mix/Reinhard.yaml
#CUDA_VISIBLE_DEVICES=7 python train_mil.py --yaml_path configs/Diagnosis/Stains_Mix/Vahadane.yaml

#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Diagnosis/MIL/TRANS_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Diagnosis/MIL/CLAM_MB_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Diagnosis/MIL/CLAM_SB_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Diagnosis/MIL/WIKG_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python train_mil.py --yaml_path configs/Diagnosis/MIL/MAMBA_MIL.yaml
#CUDA_VISIBLE_DEVICES=1 python train_mil.py --yaml_path configs/Diagnosis/MIL/MAMBA2D_MIL.yaml
#CUDA_VISIBLE_DEVICES=2 python train_mil.py --yaml_path configs/Diagnosis/MIL/AEM_MIL.yaml
#CUDA_VISIBLE_DEVICES=3 python train_mil.py --yaml_path configs/Diagnosis/MIL/MICO_MIL.yaml
#CUDA_VISIBLE_DEVICES=4 python train_mil.py --yaml_path configs/Diagnosis/MIL/MICRO_MIL.yaml
#CUDA_VISIBLE_DEVICES=5 python train_mil.py --yaml_path configs/Diagnosis/MIL/MSM_MIL.yaml
#CUDA_VISIBLE_DEVICES=6 python train_mil.py --yaml_path configs/Diagnosis/MIL/TDA_MIL.yaml
#CUDA_VISIBLE_DEVICES=7 python train_mil.py --yaml_path configs/Diagnosis/MIL/GDF_MIL.yaml
