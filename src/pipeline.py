import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from dataclass import text_dataset_train
from dataclass import TextDataSet, ImageDataSet
from transformers import ViTImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms, ToPILImage, ToTensor
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Lambda


def cross_validation_training(model, dataset, folds=10):
    kf = KFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        print(f"Fold {i}:")
        X_train, y_train = [], []
        X_test, y_test = [], []
        for idx in train_index:
            value, target = dataset[idx]
            X_train.append(value.toarray().flatten())
            y_train.append(target)

        for idx in test_index:
            value, target = dataset[idx]
            X_test.append(value.toarray().flatten())
            y_test.append(target)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        model.fit(X_train, y_train)

        predicted = model.predict(np.array(X_test))
        print(f"Accuracy score: {accuracy_score(np.array(y_test), predicted)}")


def img_pipeline():
    extractor = ViTImageProcessor.from_pretrained(
        "DunnBC22/dit-base-Business_Documents_Classified_v2"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "DunnBC22/dit-base-Business_Documents_Classified_v2"
    )
    image_data_set = ImageDataSet("../datasets/train_set", extractor)

    dataloader = DataLoader(image_data_set, batch_size=8, shuffle=True, num_workers=10)

    model.classifier = torch.nn.Linear(
        in_features=model.classifier.in_features, out_features=21
    )
    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )
    num_epochs = 10
    losses = []
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0
        for train_features, train_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(train_features).logits
            train_labels = train_labels.type(torch.float32)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        al = running_loss / len(image_data_set)
        losses.append(al)
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    img_pipeline()
