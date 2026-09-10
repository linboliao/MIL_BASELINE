import argparse
from utils.yaml_utils import read_yaml,update_config_from_options
from process.process_all import process
import warnings
import os
import re
from utils.general_utils import get_time,merge_k_fold_logs
warnings.filterwarnings('ignore')


def _fold_sort_key(path):
    match = re.search(r'_(\d+)fold\.csv$', os.path.basename(path))
    if match is None:
        return (float('inf'), os.path.basename(path))
    return (int(match.group(1)), os.path.basename(path))


def main(arg):
    yaml_path = arg.yaml_path
    print(f"MIL-yaml path: {yaml_path}")
    args = read_yaml(yaml_path)
    # dinamically update the config file with the options
    if arg.options:
        args = update_config_from_options(args,arg.options)

    if args.Dataset.dataset_root_dir == {} and args.Dataset.dataset_csv_path != None:
        '''
        None-fold split
        '''
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir,exist_ok=True)
        sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
        os.makedirs(sub_dir,exist_ok=True)
        args.Logs.now_log_dir = os.path.join(sub_dir,f'seed_{args.General.seed}_{get_time()}')
        process(args,yaml_path,arg.options)

    else:
        '''
        k-fold split
        '''
        dataset_root_dir = args.Dataset.dataset_root_dir
        if not os.path.isdir(dataset_root_dir):
            raise FileNotFoundError(f'Dataset root directory not found: {dataset_root_dir}')
        k_fold_csv_paths = sorted(
            [
                os.path.join(dataset_root_dir, path)
                for path in os.listdir(dataset_root_dir)
                if path.lower().endswith('.csv')
                and os.path.isfile(os.path.join(dataset_root_dir, path))
            ],
            key=_fold_sort_key,
        )
        if not k_fold_csv_paths:
            raise FileNotFoundError(f'No fold CSV files found in: {dataset_root_dir}')
        # --run_ts lets N parallel per-fold launches share ONE seed_<seed>_<ts>/ dir.
        process_time = getattr(arg, 'run_ts', None) or get_time()
        log_root_dir = args.Logs.log_root_dir
        os.makedirs(log_root_dir,exist_ok=True)
        sub_dir = os.path.join(log_root_dir,args.Dataset.DATASET_NAME,args.General.MODEL_NAME)
        os.makedirs(sub_dir,exist_ok=True)
        fold_total_log_dir = os.path.join(sub_dir,f'seed_{args.General.seed}_{process_time}')

        if getattr(arg, 'merge_only', False):
            merge_k_fold_logs(fold_total_log_dir,args.General.process_pipeline)
            return

        only_fold = getattr(arg, 'only_fold', None)
        for k_idx,k_fold_csv_path in enumerate(k_fold_csv_paths):

            now_fold = k_idx+1
            if only_fold is not None and now_fold != only_fold:
                continue
            args.Dataset.dataset_csv_path = k_fold_csv_path
            args.Dataset.now_fold = now_fold
            args.Logs.now_log_dir = os.path.join(fold_total_log_dir,f'fold_{now_fold}')
            os.makedirs(args.Logs.now_log_dir,exist_ok=True)
            from utils.repro_utils import deterministic_requested, enable_full_determinism, write_run_metadata
            if deterministic_requested():
                enable_full_determinism(args.General.seed)
                write_run_metadata(args.Logs.now_log_dir, args, k_fold_csv_path)
            process(args,yaml_path,arg.options)
            print(f'K-Fold:{now_fold} Done!')

        # a per-fold launch (--only_fold) leaves merging to a final `--merge_only` pass
        if only_fold is None and not getattr(arg, 'no_merge', False):
            merge_k_fold_logs(fold_total_log_dir,args.General.process_pipeline)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path',type=str,default='/path/to/your/yaml',help='path to MIL-yaml file')
    parser.add_argument('--options',nargs='+',help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into the yaml config file')
    parser.add_argument('--only_fold',type=int,default=None,help='k-fold: run only this 1-based fold (for parallel per-fold launches)')
    parser.add_argument('--run_ts',type=str,default=None,help='k-fold: fixed seed_<seed>_<ts> dir name so parallel per-fold launches co-locate')
    parser.add_argument('--no_merge',action='store_true',help='k-fold: skip the final merge_k_fold_logs')
    parser.add_argument('--merge_only',action='store_true',help='k-fold: only run merge_k_fold_logs on seed_<seed>_<run_ts>/, no training')
    arg = parser.parse_args()
    main(arg)
