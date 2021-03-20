import tqdm
import argparse
import torch
import os

from configs import cfg 
from data_manager.build import make_dataloader


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


def make_lr_scheduler(cfg, optimizer):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.OPTIMIZER.MAX_EPOCH * cfg.DATALOADER.NUM_BATCH
    )
    return lr_scheduler


def train(cfg, train_loader, valid_loader, save=False):
    best_epoch = ''
    best_score = 0

    # loss function
    criterion = torch.nn.CrossEntropyLoss()

    # # create learning rate scheduler 
    # lr_scheduler = make_lr_scheduler(cfg, optimizer)

    from pprint import pprint 

    for epoch in range(cfg.OPTIMIZER.MAX_EPOCH):
        loader = tqdm.tqdm(train_loader)

        for i, data in enumerate(loader, 1):
            # data {featx: xxx, label: xxx}
            y = data['label']
            exit()

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

    # # create output dir
    # experiment_dir = get_unique_temp_folder(cfg.OUTPUT_DIR)

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

    train_loader = make_dataloader(cfg, split='train')
    valid_loader = make_dataloader(cfg, split='valid')
    test_loader = make_dataloader(cfg, split='test')

    train(cfg, train_loader, valid_loader, save=False)

if __name__ == '__main__':
    main()
