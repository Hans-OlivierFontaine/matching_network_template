import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os.path
from pathlib import Path
import csv

import numpy as np

np.random.seed(2191)


class MatchingDataset(data.Dataset):
    def __init__(self, dataroot: Path, version='train', n_episodes=1000, n_ways=10, k_shots=1, img_size: int = 256,
                 channels: int = 3):

        self.nEpisodes: int = n_episodes
        self.n_ways: int = n_ways
        self.k_shots: int = k_shots
        self.n_samples: int = self.k_shots * self.n_ways
        self.n_samplesNShot: int = 5
        self.filenameToPILImage = lambda x: Image.open(x.__str__())
        self.PiLImageResize = lambda x: x.resize((img_size, img_size))
        self.transform = transforms.Compose([self.filenameToPILImage,
                                             self.PiLImageResize,
                                             transforms.ToTensor()
                                             ])
        self.img_size: int = img_size
        self.channels: int = channels

        def load_split(split_file: Path) -> dict:
            dict_labels = {}
            with split_file.open("r") as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader, None)   # Ignore first line (explanation: Filename, label)
                for row in csvreader:
                    filename = row[0]
                    label = row[1]
                    if label in dict_labels.keys():
                        dict_labels[label].append(filename)
                    else:
                        dict_labels[label] = [filename]
            return dict_labels
        self.dataset_dir: Path = (dataroot / 'images').absolute()
        self.csv_data = load_split(split_file=(dataroot / (version + '.csv')).absolute())
        self.data = []
        self.img2label = {}
        for i, (k, v) in enumerate(self.csv_data.items()):
            self.data.append(v)
            for img in v:
                self.img2label[img] = k

        self.support_set_x_batch = []
        self.target_x_batch = []
        for b in np.arange(self.nEpisodes):
            selected_classes = np.random.choice(list(self.csv_data.keys()), self.n_ways, False)
            selected_class_meta_test = np.random.choice(selected_classes)
            support_set_x = []
            target_x = []
            for c in selected_classes:
                number_of_samples = self.k_shots
                if c == selected_class_meta_test:
                    number_of_samples += self.n_samplesNShot
                selected_samples = np.random.choice(len(self.csv_data[c]), number_of_samples, False)
                index_d_train = np.array(selected_samples[:self.k_shots])
                support_set_x.append(np.array(self.csv_data[c])[index_d_train].tolist())
                # if c == selected_class_meta_test:
                index_d_test = np.array(selected_samples[self.k_shots:])
                target_x.append(np.array(self.csv_data[c])[index_d_test].tolist())
            self.support_set_x_batch.append(support_set_x)
            self.target_x_batch.append(target_x)

    def __getitem__(self, index):

        support_set_x = torch.FloatTensor(self.n_samples, self.channels, self.img_size, self.img_size)
        # support_set_y = np.zeros(self.n_samples, dtype=np.int)
        target_x = torch.FloatTensor(self.n_samplesNShot, self.channels, self.img_size, self.img_size)
        # target_y = np.zeros(self.n_samplesNShot, dtype=np.int)

        flatten_support_set_x_batch = [(self.dataset_dir / item)
                                       for sublist in self.support_set_x_batch[index] for item in sublist]
        support_set_y = np.array([self.img2label[item]
                                  for sublist in self.support_set_x_batch[index] for item in sublist])
        flatten_target_x = [(self.dataset_dir / item)
                            for sublist in self.target_x_batch[index] for item in sublist]
        target_y = np.array([self.img2label[item]
                             for sublist in self.target_x_batch[index] for item in sublist])

        for i, path in enumerate(flatten_support_set_x_batch):
            if self.transform is not None:
                support_set_x[i] = self.transform(path)

        for i, path in enumerate(flatten_target_x):
            if self.transform is not None:
                target_x[i] = self.transform(path)

        # convert the targets number between [0, self.classes_per_set)
        classes_dict_temp = {np.unique(support_set_y)[i]: i for i in np.arange(len(np.unique(support_set_y)))}
        support_set_y = np.array([classes_dict_temp[i] for i in support_set_y])
        target_y = np.array([classes_dict_temp[i] for i in target_y])

        return support_set_x, torch.IntTensor(support_set_y), target_x, torch.IntTensor(target_y)

    def __len__(self):
        return self.nEpisodes
