cd ../../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=3 python -u scripts/Diagnosis/run_spe.py --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml
