import glob
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import id2label, label2id


class TextDataSet(Dataset):
    def _clean_text(self, text):
        text = text.lower()  # making text lowercase
        return text

    def __init__(self, dataset_path, labels_path):
        self.texts = pd.read_pickle(dataset_path)
        self.labels = pd.read_pickle(labels_path)
        self.labels = {k: id2label[v] for k, v in self.labels.items()}

        # some keys are missing
        keys_to_del = []
        for k, v in self.texts.items():
            if k not in self.labels.keys():
                keys_to_del.append(k)

        for k in keys_to_del:
            del self.texts[k]
        self.images = list(self.texts.keys())

    def preprocess_text(self):
        # preprocessing text
        self.texts = {k: self._clean_text(v) for k, v in self.texts.items()}
        tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        tfidf_vectorizer.fit(list(self.texts.values()))
        self.vectorizer = tfidf_vectorizer
        self.texts = {k: tfidf_vectorizer.transform([v]) for k, v in self.texts.items()}

        # todo: check for classes distributions on words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[image]
        text = self.texts[image]
        return text, label


class ImageDataSet(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.labels = {}
        for dir_name in glob.glob(img_dir + "/*"):
            label = os.path.basename(dir_name)
            for file_name in glob.glob(dir_name + "/*"):
                file_name = os.path.basename(file_name)
                self.labels[file_name] = id2label[label]
        self.transform = transform
        self.images = list(self.labels.keys())
        self.use_rgb = False

    def __len__(self):
        return len(self.labels)

    def use_rgb(self):
        self.use_rgb = True

    def __getitem__(self, idx):
        image_name = self.images[idx]
        img_path = os.path.join(
            self.img_dir, label2id[self.labels[image_name]], image_name
        )
        image = Image.open(img_path)
        if self.use_rgb:
            image = image.convert("RGB")
        return np.array(image), self.labels[image_name]


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
