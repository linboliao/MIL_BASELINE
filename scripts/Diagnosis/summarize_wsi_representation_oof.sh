cd ../../
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=/home/lbliao/anaconda3/envs/clam/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
python -u scripts/Diagnosis/summarize_oof.py --input-root result/Diagnosis --output-dir result/Diagnosis/Statistics --bootstrap-iterations 2000 --seed 2024 --threshold 0.5
