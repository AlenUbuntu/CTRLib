import tqdm
import argparse
import torch
import os
import numpy as np 

from configs import cfg 
from data_manager.build import make_dataloader
from sklearn.metrics import roc_auc_score
from copy import deepcopy
from torch.utils.collect_env import get_pretty_env_info

from models.lr import LogisticRegressionModel
from models.fm import FactorizationMachineModel

def get_unique_temp_folder(input_temp_folder_path):
    """
    The function returns the path of a folder in result_experiments
    The function guarantees that the folder is not already existent and it creates it
    """

    if input_temp_folder_path[-1] == "/":
        input_temp_folder_path = input_temp_folder_path[:-1] # remove trailing /

    progressive_temp_folder_name = input_temp_folder_path

    counter_suffix = 0

    # xxx_1_2_3 ...
    while os.path.isdir(progressive_temp_folder_name):
        counter_suffix += 1
        progressive_temp_folder_name = input_temp_folder_path + "_" + str(counter_suffix)
    
    progressive_temp_folder_name += "/" # xxx_1_2_3/
    os.makedirs(progressive_temp_folder_name)

    return progressive_temp_folder_name


def make_lr_scheduler(cfg, optimizer, length):
    if cfg.OPTIMIZER.LR_SCHEDULER == 'cosine_annealing_lr':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.OPTIMIZER.MAX_EPOCH * length
        )
    elif cfg.OPTIMIZER.LR_SCHEDULER == 'step_lr':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            cfg.OPTIMIZER.STEP_SIZE
        )
    else:
        raise NotImplementedError("Learning rate scheduler {} is not supported yet.".format(cfg.OPTIMIZER.LR_SCHEDULER))
    return lr_scheduler

def make_optimizer(cfg, model):
    if cfg.OPTIMIZER.NAME == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.LR, betas=cfg.OPTIMIZER.BETAS, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    elif cfg.OPTIMIZER.NAME == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.OPTIMIZER.LR, momentum=cfg.OPTIMIZER.MOMENTUM, weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY)
    else:
        raise NotImplementedError("{} is not supported yet".format(cfg.OPTIMIZER.NAME))
    
    return optimizer


def get_model(cfg, field_info): 
    name = cfg.MODEL_NAME

    if name == 'lr':
        return LogisticRegressionModel(cfg, field_info)
    if name == 'fm':
        return FactorizationMachineModel(cfg, field_info)


def train(cfg, model, train_loader, valid_loader, save=False):
    best_epoch = ''
    best_score = 0
    best_log_loss = 0.
    best_model = None

    # loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # create optimizer
    optimizer = make_optimizer(cfg, model)

    # create lr scheduler
    lr_scheduler = make_lr_scheduler(cfg, optimizer, len(train_loader))

    # change device
    model = model.to(cfg.DEVICE)

    for epoch in range(cfg.OPTIMIZER.MAX_EPOCH):
        loader = tqdm.tqdm(train_loader)
        epoch_loss = 0.0
        iters = 0

        for i, data in enumerate(loader, 1):
            # data {featx: xxx, label: xxx}
            y = data['label'].to(torch.float).to(cfg.DEVICE)
            del data['label']

            for field_name in data: 
                data[field_name] = data[field_name].to(cfg.DEVICE)

            logits = model(data)
            loss = criterion(logits, y)

            epoch_loss += loss.item()
            iters += 1

            loader.set_description('Epoch: {} Batch Loss={:.2f} Avg Epoch Loss={:.2f}'.format(epoch, loss.item(), epoch_loss/iters)) 

            # backwards
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cfg.OPTIMIZER.LR_SCHEDULER == 'cosine_annealing_lr':
                lr_scheduler.step()
        
        if cfg.OPTIMIZER.LR_SCHEDULER == 'step_lr':
            lr_scheduler.step()
        
        # model evaluation
        print("Evaluating on Validation Dataset")
        model.eval()
        auc, log_loss = test(cfg, model, valid_loader, device=cfg.DEVICE)
        if auc > best_score:
            best_score = auc 
            best_log_loss = log_loss
            best_epoch = epoch 
            best_model = deepcopy(model)
        print("Epoch: {} - AUC Score: {:.6f} - Best Epoch: {} - Best AUC: {:.6f} - Log Loss: {:.6f}".format(epoch, auc, best_epoch, best_score, best_log_loss))
        model.train()

    return best_model


def test(cfg, model, test_loader, device='cpu'):
    model = model.to(device)
    model.eval()

    test_loader = tqdm.tqdm(test_loader)

    # loss function
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    y_true = []
    y_score = []
    label1_prob = []
    label0_prob = []
    log_loss = 0.
    count = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 1):
            y = data['label'].to(device)
            del data['label']

            for field_name in data: 
                data[field_name] = data[field_name].to(device)

            logits = model(data)
            p_prob = torch.sigmoid(logits)

            loss = criterion(logits, y.float())
            log_loss += loss.item()
            count += len(y)

            y_true.extend(y.detach().cpu().numpy().tolist())
            y_score.extend(p_prob.detach().cpu().numpy().tolist())

            label1_prob.extend(p_prob.detach().cpu().numpy()[y.detach().cpu().numpy()==1].tolist())
            label0_prob.extend(p_prob.detach().cpu().numpy()[y.detach().cpu().numpy()==0].tolist())

    # compute scores
    auc = roc_auc_score(y_true, y_score)
    log_loss = log_loss / count
    return auc, log_loss


def main():
    parser = argparse.ArgumentParser(description='PyTorch RecLib')

    parser.add_argument(
        '--config-file',
        default='Configs/default.yaml',
        metavar='FILE',
        help='path to configuration file',
        type=str
    )
    
    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    assert cfg.MODEL_NAME in {'lr', 'fm', 'all'}, "Unexpected model: {}, must be one of 'lr', 'all'.".format(args.model)

    # # create output dir
    # experiment_dir = get_unique_temp_folder(cfg.OUTPUT_DIR)

    print("Collecting env info (may take some time)\n")
    print(get_pretty_env_info())
    print("Loading configuration file from {}".format(args.config_file))
    print('Running with configuration: \n')
    print(cfg)

    # set random seed for pytorch and numpy 
    if cfg.SEED != 0:
        print("Using manual seed: {}".format(cfg.SEED))
        torch.manual_seed(cfg.SEED)
        torch.cuda.manual_seed(cfg.SEED)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        np.random.seed(cfg.SEED)
    else:
        print("Using random seed")
        torch.backends.cudnn.benchmark = True

    # create dataloader
    train_loader, field_info = make_dataloader(cfg, split='train')
    valid_loader, _ = make_dataloader(cfg, split='valid')
    test_loader, _ = make_dataloader(cfg, split='test')

    # create model 
    model = get_model(cfg, field_info)

    best_model = train(cfg, model, train_loader, valid_loader, save=False)
    auc, log_loss = test(cfg, best_model, test_loader, device=cfg.DEVICE)
    print("*"*20)
    print("* Test AUC: {:.5f} *".format(auc))
    print("* Test Log Loss: {:.5f} *".format(log_loss))
    print("*"*20)

if __name__ == '__main__':
    main()
