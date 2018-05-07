import os
import unittest
from collections import Counter

import torch

from data import DirectoryLoaderConfig, DataLoader, DirectoryLoader


dir_path = os.path.dirname(os.path.realpath(__file__))


class TestDirectoryLoader(unittest.TestCase):
    def setUp(self):
        self.path = os.path.abspath(os.path.join(dir_path, "resources", "directory_loader_imgs"))
        self.source = "directory"
        self.config = DirectoryLoaderConfig.build_with("resnet34")

    def test_init_loader(self):
        loader = DataLoader.build_with(self.path, self.source, self.config)
        self.assertTrue(isinstance(loader, DirectoryLoader))

    def test_size_perclass(self):
        loader = DataLoader.build_with(self.path, self.source, self.config)

        count = Counter()
        for batch in loader.iterator():
            target = batch["target"]
            count[target[0].item()] += 1

        self.assertEqual(len(count.keys()), 2)
        self.assertEqual(count[loader.dataset.class_to_idx["goldfinch"]], 2)
        self.assertEqual(count[loader.dataset.class_to_idx["terrapin"]], 1)

    def test_load_once(self):
        loader = DataLoader.build_with(self.path, self.source, self.config)

        size = 0
        for batch in loader.iterator():
            target = batch["target"]
            size += target.size(0)

        self.assertEqual(size, 3)

    def test_load_twice(self):
        loader = DataLoader.build_with(self.path, self.source, self.config)

        size = 0
        for batch in loader.iterator():
            target = batch["target"]
            size += target.size(0)
        for batch in loader.iterator():
            target = batch["target"]
            size += target.size(0)

        self.assertEqual(size, 6)

    def test_feature_size_resnet18(self):
        config = DirectoryLoaderConfig.build_with("resnet18")
        loader = DataLoader.build_with(self.path, self.source, config)

        batch = next(loader.iterator())
        tensor = batch["avgpool_512"]

        self.assertEqual(tensor.size(1), 512)

    def test_feature_size_resnet34(self):
        config = DirectoryLoaderConfig.build_with("resnet34")
        loader = DataLoader.build_with(self.path, self.source, config)

        batch = next(loader.iterator())
        tensor = batch["avgpool_512"]

        self.assertEqual(tensor.size(1), 512)

    def test_feature_detached(self):
        loader = DataLoader.build_with(self.path, self.source, self.config)

        batch = next(loader.iterator())
        tensor = batch["avgpool_512"]

        self.assertFalse(tensor.requires_grad)

    def test_sequential(self):
        config = DirectoryLoaderConfig.build_with("resnet18")
        loader = DataLoader.build_with(self.path, self.source, config)

        perm = [0, 1, 2]
        for i, batch in enumerate(loader.iterator()):
            example_id = batch["example_ids"][0]
            self.assertEqual(perm[i], example_id)
        for i, batch in enumerate(loader.iterator()):
            example_id = batch["example_ids"][0]
            self.assertEqual(perm[i], example_id)

    def test_shuffle(self):
        config = DirectoryLoaderConfig.build_with("resnet18")
        config.shuffle = True
        loader = DataLoader.build_with(self.path, self.source, config)

        randperm0 = None
        for i, batch in enumerate(loader.iterator()):
            if randperm0 is None:
                randperm0 = loader.sampler.randperm
            example_id = batch["example_ids"][0]
            self.assertEqual(randperm0[i], example_id)

        randperm1 = None
        for i, batch in enumerate(loader.iterator()):
            if randperm1 is None:
                randperm1 = loader.sampler.randperm
            example_id = batch["example_ids"][0]
            self.assertEqual(randperm1[i], example_id)

        # 1/6 chance of these being equal with dataset size = 3
        # If necessary, then fix the seed.
        self.assertEqual(all(i0 == i1 for i0, i1 in zip(randperm0, randperm1)), False)

    def test_cache_hit(self):
        config = DirectoryLoaderConfig.build_with("resnet18")
        loader = DataLoader.build_with(self.path, self.source, config)

        # Fill cache
        loader.cache[0] = 1
        loader.cache_keys.add(0)

        batch = next(loader.iterator())
        tensor = batch["avgpool_512"]
        index = batch["example_ids"][0]

        self.assertEqual(index, 0)
        self.assertTrue(tensor.eq(1).all())

    @unittest.skipIf(not torch.cuda.is_available(), "skipping cuda test")
    def test_cuda(self):
        config = DirectoryLoaderConfig.build_with("resnet18")
        config.cuda = True
        loader = DataLoader.build_with(self.path, self.source, config)

        batch = next(loader.iterator())
        tensor = batch["avgpool_512"]
        target = batch["target"]

        self.assertEqual(tensor.is_cuda, True)
        self.assertEqual(target.is_cuda, False)


if __name__ == '__main__':
    unittest.main()
