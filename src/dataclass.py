import glob
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import random
import torch

from sklearn.feature_extraction.text import CountVectorizer
from utils import CTFIDFVectorizer
from utils import id2label, label2id
import torch.nn.functional as F

import re


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TextDataSet(Dataset):
    def _clean_text(self, text):
        text = text.lower()  # making text lowercase
        text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text)  # regular expression for deleting
        punctuations = "@#!?+&*[]-%.:/();$=><|{}^" + "'`" + "_"
        for p in punctuations:
            text = text.replace(p, "")  # punctuations removal
        return text

    def __init__(self, dataset_path, labels_path):
        self.texts = pd.read_pickle(dataset_path)
        self.labels = pd.read_pickle(labels_path)
        self.labels = {k: id2label[v] for k, v in self.labels.items()}

        # making it unified with labels from images
        self.texts = {os.path.basename(k): v for k, v in self.texts.items()}
        self.labels = {os.path.basename(k): v for k, v in self.labels.items()}

        # some keys are missing
        keys_to_del = []
        for k, v in self.texts.items():
            if k not in self.labels.keys():
                keys_to_del.append(k)

        for k in keys_to_del:
            del self.texts[k]

        self.preprocess_text()
        self.images = list(self.labels.keys())

    def preprocess_text(self):
        # preprocessing text
        self.texts = {k: self._clean_text(v) for k, v in self.texts.items()}
        dic = {
            text: label
            for (k_2, text), (k_1, label) in zip(
                self.texts.items(), self.labels.items()
            )
            if k_1 == k_2
        }
        docs = pd.DataFrame({"texts": list(dic.keys()), "labels": list(dic.values())})
        docs_per_class = docs.groupby(["labels"], as_index=False).agg(
            {"texts": " ".join}
        )
        count_vectorizer = CountVectorizer(stop_words="english", max_features=128).fit(
            docs_per_class.texts
        )
        count = count_vectorizer.transform(docs_per_class.texts)
        ctfidf_vectorizer = CTFIDFVectorizer().fit(count, n_samples=len(docs))
        self.texts = {
            k: ctfidf_vectorizer.transform(count_vectorizer.transform([v]))
            for k, v in self.texts.items()
        }
        # shape of text [1, 512]

        # maybe at the end we can also check normal tf-idf?
        # though it worked worse

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[image]
        text = self.texts[image]
        return torch.tensor(text.todense(), dtype=torch.float32), label


class ImageDataSet(Dataset):
    def __init__(self, img_dir, extractor=None, test=False):
        self.img_dir = img_dir
        self.labels = {}
        self.images = []
        self.test = test
        if test:
            for img_name in glob.glob(img_dir + "/*"):
                self.images.append(os.path.basename(img_name))
        else:
            for dir_name in glob.glob(img_dir + "/*"):
                label = os.path.basename(dir_name)
                for file_name in glob.glob(dir_name + "/*"):
                    file_name = os.path.basename(file_name)
                    self.labels[file_name] = id2label[label]
        self.extractor = extractor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        if self.test:
            img_path = os.path.join(self.img_dir, image_name)
        else:
            img_path = os.path.join(
                self.img_dir, label2id[self.labels[image_name]], image_name
            )
        image = Image.open(img_path)
        image = image.convert("RGB")
        image = self.extractor(images=image, return_tensors="pt")["pixel_values"]
        image = torch.squeeze(image)
        label = image_name
        if not self.test:
            label = F.one_hot(torch.tensor(self.labels[image_name]), 21)

        return image.to(device), image_name


class EnsembleDataset(Dataset):
    def __init__(self, img_dir, text_data_path, text_labels_path, extractor=None):
        self.textdataset = TextDataSet(text_data_path, text_labels_path)
        self.imgdataset = ImageDataSet(img_dir, extractor)

        self.images = list(
            set(self.textdataset.images).intersection(self.imgdataset.images)
        )
        # unifying images
        self.textdataset.images = self.images
        self.imgdataset.images = self.images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.imgdataset[idx]
        text, _ = self.textdataset[idx]

        return (img, text), label


text_dataset_train = TextDataSet(
    "../hackathon/train_set_ocr.pkl", "../hackathon/train_labels_final.pkl"
)
image_dataset_train = ImageDataSet("../datasets/train_set")


if __name__ == "__main__":
    print("Number of examples in text dataset: ", len(text_dataset_train))
    print(
        "Some random example",
        text_dataset_train[random.randint(0, len(text_dataset_train))],
    )

    # print(
    #     "Some random example",
    #     image_dataset_train[random.randint(0, len(text_dataset_train))],
    # )
