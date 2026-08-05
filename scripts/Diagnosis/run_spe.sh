cd ../../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH

python -u scripts/Diagnosis/run_spe.py \
  --spe-config configs/Diagnosis/SPE/hierarchical_spe.yaml \
  --devices cuda:1,cuda:3,cuda:4,cuda:7
