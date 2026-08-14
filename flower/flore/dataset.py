"""Dataset loading and partitioning."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from typing import Tuple, List
from datasets import DatasetDict, load_dataset
from easydict import EasyDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DistributionPartitioner
from flwr_datasets.preprocessor import Preprocessor
from torchvision.transforms import Compose, Normalize, ToTensor


FDS = None  # Cache FederatedDataset

MNIST_TRANSFORMS = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

TINY_MEAN, TINY_STD = (0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)


class TinyImageNet(torch.utils.data.Dataset):
    """Tiny-ImageNet-200 served from preconverted npy arrays (CIFAR-like API:
    .data uint8 NHWC, .targets list). Run probe/convert_tinyimagenet.py once to
    build the npy files from the official zip."""

    def __init__(self, root="./data", train=True, download=False, transform=None):
        import os
        from PIL import Image
        self._Image = Image
        split = "train" if train else "val"
        base = os.path.join(root, "tiny-imagenet-200")
        self.data = np.load(os.path.join(base, f"{split}_images.npy"))
        self.targets = np.load(os.path.join(base, f"{split}_labels.npy")).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self._Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def get_transforms(dataset_name: str):
    """Get transforms for dataset."""
    if dataset_name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
    elif dataset_name == "mnist":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif dataset_name in ["tinyimagenet", "tiny-imagenet"]:
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(TINY_MEAN, TINY_STD),
        ])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return transform_train, transform_test


def partition_dirichlet(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42
) -> List[List[int]]:
    """Partition dataset using Dirichlet distribution."""
    targets = np.array(dataset.targets)
    num_classes = len(np.unique(targets))
    
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        rng.shuffle(idx_k)
        
        # Sample proportions from Dirichlet
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        idx_k_split = np.split(idx_k, proportions)
        for i, idx in enumerate(idx_k_split):
            client_indices[i].extend(idx.tolist())
    
    return client_indices


# def load_partition_data(partition_id: int, config: dict) -> Tuple[DataLoader, DataLoader]:
#     """Load data for a specific partition."""
#     dataset_name = config["dataset"]["name"]
#     num_clients = config["num_clients"]
#     batch_size = config["dataset"]["batch_size"]
#     alpha = config.get("dataset", {}).get("alpha", 0.5)
#     seed = config.get("dataset", {}).get("seed", 42)

#     transform_train, _ = get_transforms(dataset_name)
    
#     if dataset_name == "cifar10":
#         trainset = datasets.CIFAR10(
#             root="./data", train=True, download=True, transform=transform_train
#         )
#     elif dataset_name == "cifar100":
#         trainset = datasets.CIFAR100(
#             root="./data", train=True, download=True, transform=transform_train
#         )
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")
    
#     # Partition
#     partitioning = config["dataset"].get("partitioning", "dirichlet")
#     if partitioning == "iid":
#         # IID: Split data evenly across clients
#         total_samples = len(trainset)
#         samples_per_client = total_samples // num_clients
#         start_idx = partition_id * samples_per_client
#         end_idx = start_idx + samples_per_client if partition_id < num_clients - 1 else total_samples
#         indices = list(range(start_idx, end_idx))
#     elif partitioning == "dirichlet":
#         client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
#         indices = client_indices[partition_id]
#     else:
#         raise ValueError(f"Unknown partitioning: {partitioning}")
    
#     # Get this client's data
#     client_dataset = Subset(trainset, indices)
    
#     # Split into train/val
#     val_split = 0.2
#     n_val = int(len(client_dataset) * val_split)
#     n_train = len(client_dataset) - n_val
    
#     train_subset, val_subset = torch.utils.data.random_split( # type: ignore
#         client_dataset, [n_train, n_val]
#     )
    
#     trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
#     valloader = DataLoader(val_subset, batch_size=batch_size)
    
#     return trainloader, valloader

from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch

def partition_mnist_shards(
    trainset,
    num_clients: int,
    num_shards: int = 400,
    shards_per_client: int = 2,
    seed: int = 42,
):
    """SCALLION-style partition: 400 single-class shards, 2 per client."""
    assert num_shards == num_clients * shards_per_client, \
        "num_shards must equal num_clients * shards_per_client"

    rng = np.random.RandomState(seed)

    # get labels
    targets = np.array(trainset.targets if hasattr(trainset, "targets") else trainset.labels)
    num_classes = len(np.unique(targets))
    shards_per_class = num_shards // num_classes

    # build shards: list of index arrays
    shards = []
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        rng.shuffle(idx_c)
        shard_size = len(idx_c) // shards_per_class
        for s in range(shards_per_class):
            start = s * shard_size
            end = (s + 1) * shard_size if s < shards_per_class - 1 else len(idx_c)
            shards.append(idx_c[start:end])

    rng.shuffle(shards)

    # assign shards to clients
    client_indices = []
    for cid in range(num_clients):
        s_start = cid * shards_per_client
        s_end = s_start + shards_per_client
        client_shards = shards[s_start:s_end]
        idx = np.concatenate(client_shards).astype(int)
        rng.shuffle(idx)
        client_indices.append(idx.tolist())

    return client_indices


def load_partition_data(partition_id: int, config: dict) -> Tuple[DataLoader, DataLoader]:
    """Load data for a specific partition (CIFAR + MNIST/F-MNIST)."""
    dataset_name = config["dataset"]["name"].lower()
    num_clients = config["num_clients"]
    batch_size = config["dataset"]["batch_size"]
    alpha = config.get("dataset", {}).get("alpha", 0.5)
    seed = config.get("dataset", {}).get("seed", 42)

    transform_train, _ = get_transforms(dataset_name)

    # ---- load training set ----
    if dataset_name == "cifar10":
        trainset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset_name == "cifar100":
        trainset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset_name == "mnist":
        trainset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
    elif dataset_name in ["fashion-mnist", "fmnist"]:
        trainset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        dataset_name = "fashion-mnist"  # normalize
    elif dataset_name in ["tinyimagenet", "tiny-imagenet"]:
        trainset = TinyImageNet(root="./data", train=True, transform=transform_train)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # ---- partition training set across clients ----
    partitioning = config["dataset"].get("partitioning", "dirichlet")

    if partitioning == "iid":
        # IID: equal contiguous chunks
        total_samples = len(trainset)
        samples_per_client = total_samples // num_clients
        start_idx = partition_id * samples_per_client
        end_idx = start_idx + samples_per_client if partition_id < num_clients - 1 else total_samples
        indices = list(range(start_idx, end_idx))

    elif partitioning == "dirichlet":
        client_indices = partition_dirichlet(trainset, num_clients, alpha, seed)
        indices = client_indices[partition_id]

    elif partitioning in ["shards", "scallion"]:
        # SCALLION-style for MNIST/F-MNIST
        if dataset_name not in ["mnist", "fashion-mnist"]:
            raise ValueError("Shards partitioning is only intended for (F)MNIST here.")
        client_indices = partition_mnist_shards(
            trainset,
            num_clients=num_clients,
            num_shards=400,
            shards_per_client=2,
            seed=seed,
        )
        indices = client_indices[partition_id]

    else:
        raise ValueError(f"Unknown partitioning: {partitioning}")

    # ---- client subset + local train/val split ----
    client_dataset = Subset(trainset, indices)

    val_split = 0.2
    n_val = int(len(client_dataset) * val_split)
    n_train = len(client_dataset) - n_val

    train_subset, val_subset = torch.utils.data.random_split(
        client_dataset, [n_train, n_val]
    )

    trainloader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )
    valloader = DataLoader(
        val_subset, batch_size=batch_size,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    return trainloader, valloader

def load_test_data(run_config):
    dataset_name = run_config["dataset"]["name"].lower()
    _, transform_test = get_transforms(dataset_name)
    batch_size_test = run_config["dataset"].get("batch_size", 256)

    if dataset_name == "cifar10":
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    elif dataset_name == "cifar100":
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform_test)
    elif dataset_name == "mnist":
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
    elif dataset_name in ["fashion-mnist", "fmnist"]:
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)
    elif dataset_name in ["tinyimagenet", "tiny-imagenet"]:
        testset = TinyImageNet(root="./data", train=False, transform=transform_test)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(testset, batch_size=batch_size_test, shuffle=False,
                      num_workers=2, pin_memory=True, persistent_workers=True)


def load_calibration_data(run_config, num_samples: int = 5000):
    """Load a subset of training data for BN calibration.

    According to HeteroFL paper Section 3.2, sBN layers need to be calibrated
    before evaluation by running a forward pass on training data.

    Args:
        run_config: Run configuration
        num_samples: Number of samples for calibration (default 5000)

    Returns:
        DataLoader for BN calibration
    """
    dataset_name = run_config["dataset"]["name"].lower()
    # Calibration should use deterministic inputs (no train-time augmentation),
    # otherwise BN running stats become noisy/inconsistent across rounds.
    _, transform_calib = get_transforms(dataset_name)
    batch_size = run_config["dataset"].get("batch_size", 64)

    if dataset_name == "cifar10":
        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_calib)
    elif dataset_name == "cifar100":
        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform_calib)
    elif dataset_name == "mnist":
        trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform_calib)
    elif dataset_name in ["fashion-mnist", "fmnist"]:
        trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_calib)
    elif dataset_name in ["tinyimagenet", "tiny-imagenet"]:
        trainset = TinyImageNet(root="./data", train=True, transform=transform_calib)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Use a random subset of training data for calibration
    num_samples = min(num_samples, len(trainset))
    indices = torch.randperm(len(trainset))[:num_samples].tolist()
    calibration_subset = torch.utils.data.Subset(trainset, indices)

    return DataLoader(calibration_subset, batch_size=batch_size, shuffle=True)


# def load_test_data(config: dict) -> DataLoader:
#     """Load test dataset."""
#     dataset_name = config["dataset"]["name"]
#     batch_size = config["dataset"]["batch_size"]
    
#     _, transform_test = get_transforms(dataset_name)
    
#     if dataset_name == "cifar10":
#         testset = datasets.CIFAR10(
#             root="./data", train=False, download=True, transform=transform_test
#         )
#     elif dataset_name == "cifar100":
#         testset = datasets.CIFAR100(
#             root="./data", train=False, download=True, transform=transform_test
#         )
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")
#     print(f"[Info] Loaded {dataset_name} test dataset with {len(testset)} samples.")
#     return DataLoader(testset, batch_size=batch_size)


class FEMNISTFilter(Preprocessor): #type: ignore
    """A Preprocessor class that filter the FEMNIST data.

    It filters data with label 0 to 9 (lower case letters 'a'-'j')
    """

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        """."""
        allowed_labels = list(range(10))  # mapping to 'a'-'j'
        filtered_dataset = dataset.filter(
            lambda example: example["character"] in allowed_labels
        )
        return filtered_dataset


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [MNIST_TRANSFORMS(img) for img in batch["image"]]

    return batch


def process_femnist(dataset):
    """Process FEMNIST when setting up centralised test data."""
    return dataset.filter(lambda example: example["character"] in list(range(10)))


def load_data(
    dataset_config: EasyDict,
    partition_id: int,
    num_partitions: int,
):
    """Load and partition data."""
    # Only initialize `FederatedDataset` once
    global FDS  # pylint: disable=global-statement
    if FDS is None:
        # Generate a vector from a log-normal probability distribution
        rng = np.random.default_rng(dataset_config.seed) #type: ignore
        distribution_array = rng.lognormal(
            dataset_config.mu, #type: ignore
            dataset_config.sigma, #type: ignore
            (num_partitions * dataset_config.num_unique_labels_per_partition), #type: ignore
        )
        distribution_array = distribution_array.reshape(
            (dataset_config.num_unique_labels, -1) #type: ignore
        )
        labels_per_partition = dataset_config.num_unique_labels_per_partition #type: ignore
        samples_per_label = dataset_config.preassigned_num_samples_per_label #type: ignore
        label_key = "character" if "femnist" in dataset_config.path else "label" #type: ignore
        partitioner = DistributionPartitioner(
            distribution_array=distribution_array,
            num_partitions=num_partitions,
            num_unique_labels_per_partition=labels_per_partition,
            partition_by=label_key,  # target column `label` ("character" for FEMNIST)
            preassigned_num_samples_per_label=samples_per_label,
        )
        if "femnist" in dataset_config.path: #type: ignore
            FDS = FederatedDataset(
                dataset=dataset_config.path, #type: ignore
                partitioners={"train": partitioner},
                preprocessor=FEMNISTFilter(),  # Add the Preprocessor class for FEMNIST
            )
        else:
            FDS = FederatedDataset(
                dataset=dataset_config.path, #type: ignore
                partitioners={"train": partitioner},
            )

    partition = FDS.load_partition(partition_id)

    # Divide data on each node: 90% train, 10% test
    partition_train_test = partition.train_test_split(
        test_size=dataset_config.val_ratio, seed=dataset_config.seed #type: ignore
    )
    # The validation set is never used because we do centralized evaluation
    # on the server on the held-out test dataset.
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    return (
        DataLoader(
            partition_train_test["train"], #type: ignore
            batch_size=dataset_config.batch_size, #type: ignore
            shuffle=True,
        ),
        DataLoader(
            partition_train_test["test"], #type: ignore
            batch_size=dataset_config.batch_size, #type: ignore
        ),
    )


def prepare_test_loader(dataset_config: EasyDict):
    """Generate the dataloader for the test set.

    Args:
        dataset_config (dict): The dataset configuration.

    Note: FEMNIST does not have a test data, so we need to manually process the
    training data to create test data.

    Returns
    -------
        DataLoader: The MNIST test set dataloader.
    """
    if "femnist" in dataset_config.path: #type: ignore
        dataset = load_dataset(path=dataset_config.path)["train"] #type: ignore
        split_dataset = dataset.train_test_split(  #type: ignore
            test_size=dataset_config.val_ratio, seed=dataset_config.seed #type: ignore
        )
        test_dataset = process_femnist(split_dataset["test"])
        test_dataset = test_dataset.with_transform(apply_transforms)
    else:
        test_dataset = load_dataset(path=dataset_config.path)["test"].with_transform( #type: ignore
            apply_transforms
        )
    return DataLoader(test_dataset, batch_size=dataset_config.batch_size) #type: ignore

import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from torch.utils.data import DataLoader, Subset
import numpy as np

def create_heterogeneous_partition(dataset, num_clients=200, classes_per_client=2):
    """
    Create heterogeneous data partition following Li & Li 2023.
    Split into 400 shards (each containing samples from one class).
    Each client gets 2 shards from at most 2 classes.
    """
    # Group indices by class
    targets = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(targets))
    
    class_indices = {c: np.where(targets == c)[0] for c in range(num_classes)}
    
    # Create 400 shards (40 shards per class)
    shards = []
    shards_per_class = 40
    
    for c in range(num_classes):
        indices = class_indices[c]
        np.random.shuffle(indices)
        
        # Split class indices into shards
        shard_size = len(indices) // shards_per_class
        for i in range(shards_per_class):
            start = i * shard_size
            end = start + shard_size if i < shards_per_class - 1 else len(indices)
            shards.append({
                'class': c,
                'indices': indices[start:end]
            })
    
    # Assign shards to clients
    client_indices = []
    np.random.shuffle(shards)
    
    for client_id in range(num_clients):
        # Each client gets 2 shards
        client_shards = shards[client_id * 2:(client_id + 1) * 2]
        
        # Combine indices from both shards
        indices = []
        for shard in client_shards:
            indices.extend(shard['indices'])
        
        np.random.shuffle(indices)
        client_indices.append(indices)
    
    return client_indices

def load_mnist_data(dataset_name="mnist", num_clients=200, batch_size=32):
    """Load MNIST or Fashion-MNIST with heterogeneous partitioning."""
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) if dataset_name == "mnist" 
                          else transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load dataset
    if dataset_name == "mnist":
        train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    else:  # fashion-mnist
        train_dataset = FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create heterogeneous partition
    client_indices = create_heterogeneous_partition(train_dataset, num_clients)
    
    # Create data loaders for each client
    client_loaders = []
    for indices in client_indices:
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
        client_loaders.append(loader)
    
    # Test loader
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return client_loaders, test_loader
