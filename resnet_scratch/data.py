# put the dataloader related functionalities here
# as we may use different image folders at different points, it seems like a good idea

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from PIL import Image
import random
import os
from pprint import pprint
import matplotlib.pyplot as plt
import torchvision.transforms as T
from time import time
import numpy as np

"""
--------------------------------------------------------------------------------
3 CLASS DATASET
--------------------------------------------------------------------------------
This dataset contains 3236 microscopic images taken from 258 patients. The
objective lens is 20x.

The training set consists of 1644 images from randomly selected 129 patients.
It includes 510 normal, 859 low-grade cancerous, and 275 high-grade cancerous
tissue images.

The test set consists of 1592 images from the remaining 129 patients. It includes 
491 normal, 844 low-grade cancerous, and 257 high-grade cancerous tissue images.

The first 1644 images (s1.jpg ... s1644.jpg) belong to the training set. The
remaining 1592 images (s1645.jpg ... s3236.jpg) belong to the test set. 

labels file includes the image labels:
1 --> Normal
2 --> Low-grade
3 --> High-grade
"""

"""
--------------------------------------------------------------------------------
5 CLASS DATASET
--------------------------------------------------------------------------------
COLON-PRIMARY-RETRIEVAL

This dataset contains 1574 microscopic images taken from 209 patients. The
objective lens is 20x.

The training set consists of 791 images from randomly selected 104 patients.
The test set consists of 784 images from the remaining 105 patients. 

The first 791 images (r1.jpg ... r791.jpg) belong to the training set. The
remaining 784 images (r792.jpg ... r1574.jpg) belong to the test set. 

There are six classes in this dataset. These are labeled as
1 --> Normal
2 --> Grade 1
3 --> Grade 1-2
4 --> Grade 2
5 --> High-grade
6 --> Medullary

The class distributions in the training and test sets are as follows:
              Training     Test
Normal        182          178
Grade 1       188          179
Grade 1-2     121          117
Grade 2       123          124
High-grade    104          105
Medullary      73           80

labels file contains the patient and class information of each image. 
In this file, each row corresponds to an image. The first column contains 
the image id. The second column contains the patient id (there are more
than one image from each patient). The last column contains the image class.

In this file, patients ids are NOT consecutively given. The ids are in the
range of 1 and 258, whereas there are only the images of 209 patients.
"""

# { CHANGE
def seed_everything(seed: int):    
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
# }

# this function creates 3 datasets for each split (train, val, test)
def create_splits(dataset_type="3class", percentage:float = 0.95):
  assert dataset_type in ["3class", "5class"]

  if dataset_type == "3class":
    means, stds = [0.7107, 0.6803, 0.7247], [0.1503, 0.1741, 0.1078]
  elif dataset_type == "5class":
    means, stds = [0.7134, 0.6744, 0.7245], [0.1470, 0.1725, 0.1062]

  train_transform = T.Compose([
    T.Resize(224),
    T.RandomCrop(size=(224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(means, stds),
  ])

  val_transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize(means, stds)
  ])

  # create train and test split ids
  if dataset_type == "3class":
    train_ids = list(range(1, 1645))
    test_ids = list(range(1645, 3237))
  elif dataset_type == "5class":
    train_ids = list(range(1, 792))
    test_ids = list(range(792, 1575))

  # { CHANGE
  # randomly sample data from train set with the given percentage
  prev_train_count = len(train_ids)
  random.shuffle(train_ids)
  train_count = int((len(train_ids)+1) * percentage)

  if percentage != 0.95:
    if dataset_type == "3class":
      val_count = int(120 * percentage) # percentage -> count, 0.1 -> 12, 0.25 -> 30
    elif dataset_type == "5class":
      val_count = int(200 * percentage) # percentage -> count, 0.1 -> 20, 0.25 -> 50
  else:
      val_count = len(train_ids) - train_count

  val_ids = train_ids[train_count:train_count+val_count]
  train_ids = train_ids[:train_count]
  
  print(f"Prev train example count: {prev_train_count}, cur train example count: {len(train_ids)}, cur val example count: {len(val_ids)}")
  # }

  # create datasets
  train_set = HEDataset(train_ids, dataset_type=dataset_type, transform=train_transform)
  val_set = HEDataset(val_ids, dataset_type=dataset_type, transform=val_transform)
  test_set = HEDataset(test_ids, dataset_type=dataset_type, transform=val_transform)

  return train_set, val_set, test_set


class HEDataset(Dataset):
  def __init__(self, ids: list, dataset_type="3class", transform=None, target_transform=None):
    # ensure that dataset types and split names are restricted 
    assert dataset_type in ["3class", "5class"]

    suffix = ".jpg"
    # prefixes are different for each dataset type
    if dataset_type == "3class":
      prefix = "t"
    elif dataset_type == "5class":
      prefix = "rd"
    
    # root independent folder and labels file names
    folder = f"../HE-dataset/{dataset_type}"
    data_folder = f"{folder}/480x640/"
    if dataset_type == "3class":
      labels_file = f"{folder}/labels.txt"
    elif dataset_type == "5class":
      labels_file = f"{folder}/labels"

    # read labels
    with open(labels_file, "r") as f:
        text = f.read()
        lines = text.split("\n")
        if dataset_type == "3class":
          lines = [int(s) - 1 for s in lines] # as we will compare the index, we need to start from 0
        elif dataset_type == "5class":
          cur_lines = []
          for line in lines:
            if not line.strip(): # check if the line is empty
              continue
            idx, patient, label = line.split()
            cur_lines.append(int(label) - 1)
          lines = cur_lines


    # form image names and labels respectively 
    img_dir, img_labels, img_ids = [], [], []
    for id in ids:
      if dataset_type == "5class":
        if lines[id - 1] != 5: # remember that we decremented labels
          img_dir.append(f"{data_folder}{prefix}{id}{suffix}")
          img_labels.append(lines[id - 1]) # needed as id in {1, 2, ..., 3266}
          img_ids.append(id)
      elif dataset_type == "3class":
        img_dir.append(f"{data_folder}{prefix}{id}{suffix}")
        img_labels.append(lines[id - 1]) # needed as id in {1, 2, ..., 3266}
        img_ids.append(id)

    self.img_dir = img_dir
    self.img_labels = img_labels
    self.img_ids = img_ids
    self.transform = transform
    self.target_transform = target_transform

    print(f"Image dir: {len(self.img_dir)}, Image labels: {len(self.img_labels)}")

  def __len__(self):
    return len(self.img_labels)

  def __getitem__(self, idx):
    img_path = self.img_dir[idx]
    image = Image.open(img_path)
    label = self.img_labels[idx]
    index = self.img_ids[idx]

    if self.transform:
      image = self.transform(image)
    if self.target_transform:
      label = self.target_transform(label)

    return image, label, index
    

if __name__ == "__main__":
  for data_type in ["3class", "5class"]:
    train_set, val_set, test_set = create_splits(data_type)

    train_dataloader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)
    
    begin = time()
    train_features, train_labels, train_idxs = next(iter(train_dataloader))
    print(f"Elapsed time: {time() - begin}")




"""
Channel-wise mean and std calculation

total_means, total_stds = torch.zeros(3), torch.zeros(3)
for train_feature, train_label in train_dataloader:
  train_feature = train_feature.squeeze(0)
  r_c = train_feature[0, :, :].float() / 255.0
  g_c = train_feature[1, :, :].float() / 255.0
  b_c = train_feature[2, :, :].float() / 255.0

  means = [torch.mean(r_c), torch.mean(g_c), torch.mean(b_c)]
  stds = [torch.std(r_c), torch.std(g_c), torch.std(b_c)]
  total_means += torch.FloatTensor(means)
  total_stds += torch.FloatTensor(stds)

total_means /= len(train_dataloader.dataset)
total_stds /= len(train_dataloader.dataset)
print(f"Channel-wise mean for {data_type}: {total_means}")
print(f"Channel-wise std for {data_type}: {total_stds}")
"""
