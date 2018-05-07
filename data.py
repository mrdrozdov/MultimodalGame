# import os

import torch
import torch.nn as nn
# from torch.autograd import Variable
import torchvision.models as models
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torch.utils.data.sampler import Sampler


def map_to_int(label):
    return int(label)


class RandomSampler(Sampler):
    r"""Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.randperm = None

    def __iter__(self):
        self.randperm = randperm = torch.randperm(len(self.data_source)).tolist()
        return iter(randperm)

    def __len__(self):
        return len(self.data_source)


class LoaderConfig(object):
    batch_size = 1
    shuffle = False
    truncate_final_batch = False
    cuda = False

    def __init__(self, map_labels=map_to_int):
        self.map_labels = map_to_int


class DirectoryLoaderConfig(LoaderConfig):

    @staticmethod
    def build_with(arch, pretrained=True):

        class Identity(nn.Module):
            def forward(self, x):
                return x

        if arch == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = Identity()
            model.eval()
        elif arch == "resnet34":
            model = models.resnet34(pretrained=pretrained)
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


class DirectoryLoader(DataLoader):
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

        # Init Sampler
        self.sampler = sampler = RandomSampler(dataset) if config.shuffle else None

        # Read images
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size,
            sampler=sampler,
            drop_last=not config.truncate_final_batch)

        self.map_labels = config.map_labels

        self.model = config.model
        self.dataset = dataset
        self.dataloader = dataloader

        self.nfeatures = 512
        self.ndata = len(dataset)
        self.cache = torch.FloatTensor(self.ndata, self.nfeatures).fill_(0)
        self.cache_keys = set()

        if config.cuda:
            self.model.cuda()

    def get_batch_indices(self, i, batch_size):
        if self.config.shuffle:
            randperm = self.sampler.randperm
            indices = [randperm[ii] for ii in range(i, i+batch_size)]
        else:
            indices = [ii for ii in range(i, i+batch_size)]
        return indices

    def get_cached_output(self, tensor, example_ids):
        model = self.model

        # Check cache
        newids = [ii for ii, key in enumerate(example_ids) if key not in self.cache_keys]
        newkeys = [key for ii, key in enumerate(example_ids) if key not in self.cache_keys]
        oldids = [ii for ii, key in enumerate(example_ids) if key in self.cache_keys]
        oldkeys = [key for ii, key in enumerate(example_ids) if key in self.cache_keys]

        # Get output
        output = torch.FloatTensor(tensor.size(0), self.nfeatures)

        if self.config.cuda:
            output = output.cuda()
        if len(newids) > 0:
            output[newids] = model(tensor[newids]).detach()
        if len(oldids) > 0:
            toload = self.cache[oldkeys]
            if self.config.cuda:
                toload = toload.cuda()
            output[oldids] = toload

        # Update cache
        if len(newids) > 0:
            tocache = output[newids]
            if self.config.cuda:
                tocache = tocache.cpu()
            self.cache[newkeys] = tocache
            self.cache_keys.update(newkeys)

        return output

    def iterator(self):
        # TODO: Add random seed.

        dataloader = self.dataloader

        map_labels = self.map_labels

        it = iter(dataloader)

        for i, imgs in enumerate(it):
            tensor, target = imgs

            if self.config.cuda:
                tensor = tensor.cuda()

            batch = {}

            batch['target'] = torch.LongTensor(list(map(map_labels, target.tolist())))
            batch['example_ids'] = example_ids = self.get_batch_indices(i, target.size(0))
            batch['layer4_2'] = None
            batch['fc'] = None
            batch['avgpool_512'] = self.get_cached_output(tensor, example_ids)

            yield batch


class HDF5Loader(object):
    def __init__(self, path, config):
        super(HDF5Loader, self).__init__()
        self.path = path
        self.config = config
