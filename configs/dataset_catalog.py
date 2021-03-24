import os 

class DatasetCatalog(object):
    DATASETS = (
        'Criteo'
    )
    @staticmethod
    def get(cfg, name, split):
        if name not in DatasetCatalog.DATASETS:
            raise RuntimeError("Dataset not available: {}".format(name))

        data_dir = cfg.DATASET.DIR 
        cache_path = cfg.DATASET.CACHE_PATH 
        rebuild_cache = cfg.DATASET.REBUILD_CACHE 
        
        return dict(
            data_dir=data_dir,
            cache_path=cache_path,
            rebuild_cache=rebuild_cache,
            split=split,
        )