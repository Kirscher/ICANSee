import unittest
import tempfile
from pathlib import Path
from PIL import Image
import torch

from dataset import RareDataset, RareTestSet


class TestRareDataset(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure
        self.temp_dir = tempfile.mkdtemp()
        self.center_dir = Path(self.temp_dir) / "center_1"
        self.center_dir.mkdir()

        # Create neo and ndbe subdirs
        neo_dir = self.center_dir / "neo"
        ndbe_dir = self.center_dir / "ndbe"
        neo_dir.mkdir()
        ndbe_dir.mkdir()

        # Create dummy images
        self.create_dummy_image(neo_dir / "neo1.png")
        self.create_dummy_image(neo_dir / "neo2.png")
        self.create_dummy_image(ndbe_dir / "ndbe1.png")
        self.create_dummy_image(ndbe_dir / "ndbe2.png")
        self.create_dummy_image(ndbe_dir / "ndbe3.png")

    def create_dummy_image(self, path):
        # Create a simple 224x224 RGB image
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img.save(path)

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_dataset_loading(self):
        dataset = RareDataset(self.temp_dir)
        self.assertEqual(len(dataset), 5)
        self.assertEqual(dataset.class_counts["neoplasia"], 2)
        self.assertEqual(dataset.class_counts["nondysplastic"], 3)

    def test_dataset_getitem(self):
        dataset = RareDataset(self.temp_dir)
        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])

    def test_testset_loading(self):
        testset = RareTestSet(self.temp_dir)
        self.assertEqual(len(testset), 5)
        self.assertEqual(testset.class_counts["neoplasia"], 2)
        self.assertEqual(testset.class_counts["nondysplastic"], 3)

    def test_testset_getitem_with_paths(self):
        testset = RareTestSet(self.temp_dir, return_paths=True)
        image, label, path = testset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])
        self.assertIsInstance(path, str)

    def test_testset_getitem_without_paths(self):
        testset = RareTestSet(self.temp_dir, return_paths=False)
        image, label = testset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertEqual(image.shape, (3, 224, 224))
        self.assertIn(label, [0, 1])


if __name__ == "__main__":
    unittest.main()
