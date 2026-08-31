import math
import torch
from torch.utils.data import DataLoader
from modules.CENTERADV_AB_MIL.centeradv_ab_mil import CENTERADV_AB_MIL
from utils.process_utils import get_process_pipeline,get_act
from utils.wsi_utils import WSI_Domain_Dataset
from utils.general_utils import set_global_seed,init_epoch_info_log,add_epoch_info_log,early_stop
from utils.model_utils import get_optimizer,get_scheduler,get_criterion,save_last_model,save_log,model_select
from utils.loop_utils import centeradv_train_loop,val_loop
from tqdm import tqdm


def process_CENTERADV_AB_MIL(args):

    train_dataset = WSI_Domain_Dataset(args.Dataset.dataset_csv_path,'train')
    val_dataset = WSI_Domain_Dataset(args.Dataset.dataset_csv_path,'val')
    test_dataset = WSI_Domain_Dataset(args.Dataset.dataset_csv_path,'test')
    process_pipeline = get_process_pipeline(val_dataset,test_dataset)
    args.General.process_pipeline = process_pipeline

    '''
    generator settings
    '''

    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers
    use_balanced_sampler = args.Dataset.balanced_sampler.use
    if use_balanced_sampler:
        sampler = train_dataset.get_balanced_sampler(replacement = args.Dataset.balanced_sampler.replacement)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers = num_workers,generator=generator,sampler=sampler)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers = num_workers,generator=generator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    print('DataLoader Ready!')

    device = torch.device(f'cuda:{args.General.device}')
    num_classes = args.General.num_classes
    in_dim = args.Model.in_dim
    L,D = args.Model.L,args.Model.D
    dropout = args.Model.dropout
    act = get_act(args.Model.act)

    domain_cfg = args.Model.domain_adv
    num_domains = domain_cfg.num_domains
    domain_hidden = domain_cfg.domain_hidden
    lambda_max = domain_cfg.lambda_max
    gamma = domain_cfg.gamma

    mil_model = CENTERADV_AB_MIL(L = L,D = D,num_classes=num_classes,dropout=dropout,act=act,in_dim=in_dim,
                                  num_domains=num_domains,domain_hidden=domain_hidden)
    mil_model.to(device)

    print('Model Ready!')

    optimizer,base_lr = get_optimizer(args,mil_model)
    scheduler,warmup_scheduler = get_scheduler(args,optimizer,base_lr)
    criterion = get_criterion(args.Model.criterion)
    domain_criterion = torch.nn.CrossEntropyLoss()
    warmup_epoch = args.Model.scheduler.warmup

    '''
    begin training
    '''
    epoch_info_log = init_epoch_info_log()
    best_model_metric = args.General.best_model_metric
    REVERSE = False
    best_val_metric = 0
    if best_model_metric == 'val_loss':
        REVERSE = True
        best_val_metric = 9999
    best_epoch = 1
    num_epochs = args.General.num_epochs
    print('Start Process!')
    print('Using Process Pipeline:',process_pipeline)
    for epoch in tqdm(range(num_epochs),colour='GREEN'):
        if epoch+1 <= warmup_epoch:
            now_scheduler = warmup_scheduler
        else:
            now_scheduler = scheduler

        # Standard DANN progressive schedule: ramps the reversal strength
        # from 0 to lambda_max over training so the domain-adversarial signal
        # doesn't destabilize the (still randomly-initialized) feature
        # extractor at the very start of training.
        p = epoch / max(num_epochs - 1, 1)
        grl_lambda = lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)

        train_loss,task_loss,domain_loss,domain_acc,cost_time = centeradv_train_loop(
            device,mil_model,train_dataloader,criterion,domain_criterion,optimizer,now_scheduler,grl_lambda)
        if process_pipeline == 'Train_Val_Test':
            val_loss,val_metrics,_ = val_loop(device,num_classes,mil_model,val_dataloader,criterion)
            test_loss,test_metrics,_ = val_loop(device,num_classes,mil_model,test_dataloader,criterion)
        elif process_pipeline == 'Train_Val':
            val_loss,val_metrics,_ = val_loop(device,num_classes,mil_model,val_dataloader,criterion)
            test_loss,test_metrics = None,None
        elif process_pipeline == 'Train_Test':
            val_loss,val_metrics,test_loss,test_metrics = None,None,None,None
            if epoch+1 == num_epochs:
                test_loss,test_metrics,_ = val_loop(device,num_classes,mil_model,test_dataloader,criterion)


        FAIL = '\033[91m'
        ENDC = '\033[0m'
        print('----------------INFO----------------\n')
        print(f'{FAIL}EPOCH:{ENDC}{epoch+1},  Train_Loss:{train_loss},  Task_Loss:{task_loss},  '
              f'Domain_Loss:{domain_loss},  Domain_Acc:{domain_acc},  GRL_Lambda:{grl_lambda:.4f},  '
              f'Val_Loss:{val_loss},  Test_Loss:{test_loss},  Cost_Time:{cost_time}\n')
        print(f'{FAIL}Val_Metrics:  {ENDC}{val_metrics}\n')
        print(f'{FAIL}Test_Metrics:  {ENDC}{test_metrics}\n')
        add_epoch_info_log(epoch_info_log,epoch,train_loss,val_loss,test_loss,val_metrics,test_metrics)

        # model selection, it only works when process_pipeline is 'Train_Val_Test' or 'Train_Val'
        best_val_metric,best_epoch = model_select(REVERSE,args,mil_model.state_dict(),val_metrics,best_model_metric,best_val_metric,epoch,best_epoch)

        '''
        early stop
        '''
        if early_stop(args,epoch_info_log,process_pipeline,epoch,mil_model.state_dict(),best_epoch):
            break

        if epoch+1 == num_epochs:
            save_last_model(args,mil_model.state_dict(),epoch+1)
            save_log(args,epoch_info_log,best_epoch,process_pipeline)
