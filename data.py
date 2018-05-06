import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms


def map_to_int(label):
    return int(label)


class LoaderConfig(object):
    batch_size = 1
    shuffle = False
    truncate_final_batch = False

    def __init__(self, map_labels=map_to_int):
        self.map_labels = map_to_int


class DirectoryLoaderConfig(LoaderConfig):

    @staticmethod
    def build_with(arch):

        class Identity(nn.Module):
            def forward(self, x):
                return x

        if arch == "resnet18":
            model = models.resnet18(pretrained=True)
            model.fc = Identity()
            model.eval()
        elif arch == "resnet34":
            model = models.resnet34(pretrained=True)
            model.fc = Identity()
            model.eval()
        else:
            raise ValueError("Incompatible architecture '{}'.".format(arch))

        return DirectoryLoaderConfig(model)


    def __init__(self, model, *args, **kwargs):
        super(DirectoryLoaderConfig, self).__init__(*args, **kwargs)
        self.model = model


class DataLoader(object):

    @staticmethod
    def build_with(path, source, config):
        if source == "directory":
            loader = DirectoryLoader(path, config)
        elif source == "hdf5":
            loader = HDF5Loader(path, config)
        else:
            raise ValueError("Incompatible source '{}'.".format(source))
        return loader

    def __init__(self):
        super(DataLoader, self).__init__()
        
    def iterator(self):
        raise NotImplementedError


class DirectoryLoader(object):
    def __init__(self, path, config):
        super(DirectoryLoader, self).__init__()
        self.path = path
        self.config = config

        # Load dataset and transform
        dataset = dset.ImageFolder(root=path,
                                   transform=transforms.Compose([
                                   transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])
                                  )

        # Read images
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=config.shuffle, drop_last=not config.truncate_final_batch)

        self.map_labels = config.map_labels

        self.model = config.model
        self.dataset = dataset
        self.dataloader = dataloader

    def iterator(self):
        model = self.model
        dataset = self.dataset
        dataloader = self.dataloader

        map_labels = self.map_labels

        for i, imgs in enumerate(dataloader):
            tensor, target = imgs

            batch = {}

            batch['target'] = torch.LongTensor(list(map(map_labels, target.tolist())))
            batch['example_ids'] = None
            batch['layer4_2'] = None
            batch['fc'] = None
            batch['avgpool_512'] = model(tensor).detach()

            yield batch
        

class HDF5Loader(object):
    def __init__(self, path, config):
        super(HDF5Loader, self).__init__()
        self.path = path
        self.config = config
