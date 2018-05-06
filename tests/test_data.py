import unittest

from collections import Counter

from data import *


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



if __name__ == '__main__':
    unittest.main()
