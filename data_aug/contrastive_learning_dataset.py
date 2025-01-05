'''
Roy modified:
Add to support Roboflow components dataset

'''

# from torchvision.transforms import transforms
# from data_aug.gaussian_blur import GaussianBlur
# from torchvision import transforms, datasets
# from data_aug.view_generator import ContrastiveLearningViewGenerator
# from exceptions.exceptions import InvalidDatasetSelection
#
#
# class ContrastiveLearningDataset:
#     def __init__(self, root_folder):
#         self.root_folder = root_folder
#
#     @staticmethod
#     def get_simclr_pipeline_transform(size, s=1):
#         """Return a set of data augmentation transformations as described in the SimCLR paper."""
#         color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
#         data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
#                                               transforms.RandomHorizontalFlip(),
#                                               transforms.RandomApply([color_jitter], p=0.8),
#                                               transforms.RandomGrayscale(p=0.2),
#                                               GaussianBlur(kernel_size=int(0.1 * size)),
#                                               transforms.ToTensor()])
#         return data_transforms
#
#     def get_dataset(self, name, n_views):
#         valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
#                                                               transform=ContrastiveLearningViewGenerator(
#                                                                   self.get_simclr_pipeline_transform(32),
#                                                                   n_views),
#                                                               download=True),
#
#                           'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
#                                                           transform=ContrastiveLearningViewGenerator(
#                                                               self.get_simclr_pipeline_transform(96),
#                                                               n_views),
#                                                           download=True)}
#
#         try:
#             dataset_fn = valid_datasets[name]
#         except KeyError:
#             raise InvalidDatasetSelection()
#         else:
#             return dataset_fn()


import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision import transforms, datasets
import torch

class CustomImageFolder(Dataset):
    """Custom dataset for loading images from a folder with no labels, specifically for contrastive learning."""

    def __init__(self, root_folder, transform=None):
        if transform is None:
            raise ValueError("Transform must be provided for contrastive learning.")
        self.root_folder = root_folder
        self.transform = transform
        self.image_paths = [os.path.join(root_folder, f) for f in os.listdir(root_folder) if
                            f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')

        # Apply the transformation to generate multiple views
        views = self.transform(image)
        labels = torch.tensor(-1) * views[0].size()[0]  ## same as datasets.stl10's argument of split = "unlabeled"
        return views, labels  # views is a list of tensors, and None for target

class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(kernel_size=int(0.1 * size)),
            transforms.ToTensor()
        ])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {
            'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                transform=ContrastiveLearningViewGenerator(
                                                    self.get_simclr_pipeline_transform(32),
                                                    n_views),
                                                download=True),

            'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                            transform=ContrastiveLearningViewGenerator(
                                                self.get_simclr_pipeline_transform(96),
                                                n_views),
                                            download=True),
            ## new added
            'roboflow': lambda: CustomImageFolder(
                os.path.join(self.root_folder, './Roboflow/components'),
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(64),
                    n_views)
            )
        }

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection(f"Invalid dataset selection: {name}")
        else:
            return dataset_fn()


# Example usage:
if __name__ == "__main__":
    root_folder = "../datasets/"
    dataset = ContrastiveLearningDataset(root_folder)
    components_dataset = dataset.get_dataset('roboflow', n_views=2) ## stl10， roboflow

    # Create a DataLoader for the components dataset
    dataloader = DataLoader(components_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Iterate over the DataLoader to access the augmented views
    for batch in dataloader:
        # Each batch contains a list of n_views tensors, each tensor is a batch of images
        images, _ = batch  # For n_views=2
        # You can now use these views for training your SimCLR model
        view1, view2 = images
        print(view1.shape, view2.shape)