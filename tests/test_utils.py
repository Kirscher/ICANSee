import unittest
import tempfile
from pathlib import Path
from PIL import Image

from utils import split_dataset
from dataset import RareDataset


class TestUtils(unittest.TestCase):
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
        self.create_dummy_image(ndbe_dir / "ndbe4.png")

    def create_dummy_image(self, path):
        # Create a simple 224x224 RGB image
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img.save(path)

    def tearDown(self):
        # Clean up temporary directory
        import shutil

        shutil.rmtree(self.temp_dir)

    def test_split_dataset(self):
        dataset = RareDataset(self.temp_dir)
        train_dataset, val_dataset = split_dataset(dataset, val_split=0.5, seed=42)

        # Check that datasets are split correctly
        total_samples = len(dataset)
        train_samples = len(train_dataset)
        val_samples = len(val_dataset)

        self.assertEqual(train_samples + val_samples, total_samples)

        # Check that split is approximately correct
        expected_val = int(total_samples * 0.5)
        # Allow some variance
        self.assertAlmostEqual(val_samples, expected_val, delta=2)

    def test_split_dataset_class_balance(self):
        dataset = RareDataset(self.temp_dir)
        train_dataset, val_dataset = split_dataset(dataset, val_split=0.5, seed=42)

        # Count classes in original dataset
        original_neo = sum(1 for _, label in dataset.samples if label == "neoplasia")
        original_ndbe = sum(
            1 for _, label in dataset.samples if label == "nondysplastic"
        )

        # Count classes in train dataset
        train_neo = 0
        train_ndbe = 0
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            if label == 1:  # neoplasia
                train_neo += 1
            else:
                train_ndbe += 1

        # Count classes in val dataset
        val_neo = 0
        val_ndbe = 0
        for i in range(len(val_dataset)):
            _, label = val_dataset[i]
            if label == 1:  # neoplasia
                val_neo += 1
            else:
                val_ndbe += 1

        # Check that class counts are preserved
        self.assertEqual(train_neo + val_neo, original_neo)
        self.assertEqual(train_ndbe + val_ndbe, original_ndbe)

    def test_split_dataset_deterministic(self):
        dataset = RareDataset(self.temp_dir)
        train1, val1 = split_dataset(dataset, val_split=0.3, seed=123)
        train2, val2 = split_dataset(dataset, val_split=0.3, seed=123)

        # Same seed should give same split
        self.assertEqual(len(train1), len(train2))
        self.assertEqual(len(val1), len(val2))

        # Check that indices are the same
        train_indices1 = train1.indices
        train_indices2 = train2.indices
        self.assertEqual(train_indices1, train_indices2)


if __name__ == "__main__":
    unittest.main()
