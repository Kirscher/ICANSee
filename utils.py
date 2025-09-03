import random
from torch.utils.data import Subset


def split_dataset(dataset, val_split=0.2, seed=42):
    """Split dataset into train and validation sets by class.

    Args:
        dataset (RareDataset): Dataset instance.
        val_split (float): Fraction of data to use for validation.
        seed (int): Random seed for reproducibility.

    Returns:
        train_subset (Subset), val_subset (Subset)
    """
    random.seed(seed)

    # Separate indices by string label
    neoplasia_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == "neoplasia"
    ]
    ndbe_indices = [
        i for i, (_, label) in enumerate(dataset.samples) if label == "nondysplastic"
    ]

    # Shuffle both lists
    random.shuffle(neoplasia_indices)
    random.shuffle(ndbe_indices)

    # Split function
    def split(indices):
        val_size = int(len(indices) * val_split)
        return indices[val_size:], indices[:val_size]  # train, val

    # Split each class
    train_neo, val_neo = split(neoplasia_indices)
    train_ndbe, val_ndbe = split(ndbe_indices)

    # Combine splits
    train_indices = train_neo + train_ndbe
    val_indices = val_neo + val_ndbe

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
