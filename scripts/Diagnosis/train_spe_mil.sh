cd ../../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH


# Core baselines
#CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/AB_MIL.yaml
#CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/CLAM_SB_MIL.yaml
#CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/CLAM_MB_MIL.yaml
#CUDA_VISIBLE_DEVICES=1 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/TRANS_MIL.yaml
#CUDA_VISIBLE_DEVICES=3 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/WIKG_MIL.yaml

# Architecture-diversity expansion required by the 11-model SPE experiment
CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MAMBA2D_MIL.yaml
#CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/AEM_MIL.yaml
#CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MICO_MIL.yaml

#CUDA_VISIBLE_DEVICES=4 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/MSM_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/TDA_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/GDF_MIL.yaml

#CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/MEAN_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/MAX_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/DS_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/DTFD_MIL.yaml
#CUDA_VISIBLE_DEVICES=0 python -u train_mil.py --yaml_path configs/Diagnosis/MIL/Supplementary/RRT_MIL.yaml

