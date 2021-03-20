import torch 
import os
import configs.dataset_catalog as dataset_catalog

from data_manager.criteo import CriteoDataset

DATASETS = {
    'Criteo': CriteoDataset,
}

def build_dataset(cfg, name, dataset_catalog, split):
    """
    Arguments:
        name (str): name of the dataset
        dataset_catalog (DatasetCatalog): contains the information on how to construct a dataset.
    """
    data_config = dataset_catalog.get(cfg, name, split)

    factory = DATASETS[name]
    dataset = factory(
        cfg=cfg,
        cache_path=data_config['cache_path'] 
    )
    dataset.set_split_fold(data_config['split'], data_config['fold'])
    return dataset 


def make_data_sampler(cfg, dataset, shuffle):
    label_counts = cfg.DATASET.LABEL_COUNTS 
    label_weights = [1./c for c in label_counts]
    sample_weights = dataset.get_sample_weights(label_weights)
    if shuffle:
        return torch.utils.data.WeightedRandomSampler(
            weights=sample_weights, 
            num_samples=cfg.DATALOADER.BATCH_SIZE,
            replacement=False)
    else:
        return torch.utils.data.sampler.SequentialSampler(dataset)


def make_dataloader(cfg, split):
    
    assert split in ('train', 'valid', 'test'), "split can only take values from 'train', 'valid', or 'test'"
    
    DatasetCatalog = dataset_catalog.DatasetCatalog

    if split == 'train':
        shuffle = True
    else:
        shuffle = False

    dataset = build_dataset(cfg, cfg.DATASET.NAME, DatasetCatalog, split)
    sampler = make_data_sampler(cfg, dataset, shuffle)

    if split == 'train':
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            drop_last=True
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=cfg.DATALOADER.BATCH_SIZE,
            sampler=sampler,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=cfg.DATALOADER.PIN_MEMORY,
            drop_last=False
        )

    return dataloader 
