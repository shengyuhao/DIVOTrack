# encoding: utf-8

from .divo import DIVO
from .partial_ilids import PartialILIDS
from .partial_reid import PartialREID
from .dataset_loader import ImageDataset

__factory = {
    'divo':DIVO,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
