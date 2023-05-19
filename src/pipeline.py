import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from dataclass import ImageDataSet, EnsembleDataset
from transformers import ViTImageProcessor, AutoModelForImageClassification
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from ensemble import Ensemble


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
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    extractor = ViTImageProcessor.from_pretrained(
        "DunnBC22/dit-base-Business_Documents_Classified_v2"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "DunnBC22/dit-base-Business_Documents_Classified_v2"
    )
    image_data_set = ImageDataSet("../datasets/train_set", extractor)
    dataset_eval, dataset_train = torch.utils.data.random_split(
        image_data_set, [0.2, 0.8]
    )

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=32, shuffle=True)

    model.classifier = torch.nn.Linear(
        in_features=model.classifier.in_features, out_features=21
    )
    if torch.cuda.is_available():
        model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0
        for train_features, train_labels in tqdm(dataloader_train):
            optimizer.zero_grad()
            train_features.to(device)
            outputs = model(train_features).logits
            train_labels = train_labels.type(torch.float32)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = 0
        length = 0
        for test_features, test_labels in tqdm(dataloader_eval):
            with torch.no_grad():
                test_features.to(device)
                outputs = model(test_features).logits
                acc += outputs == test_labels
                length += len(outputs)
        print(f"acc for batch: ", acc / length)

        al = running_loss / len(dataloader_train)
        print(f"Running loss: {al}")
        torch.save(model, f"../model_vanilla{epoch}")


def ensemble_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    extractor = ViTImageProcessor.from_pretrained(
        "DunnBC22/dit-base-Business_Documents_Classified_v2"
    )

    ensemble_dataset = EnsembleDataset(
        "../datasets/train_set",
        "../hackathon/train_set_ocr.pkl",
        "../hackathon/train_labels_final.pkl",
        extractor,
    )

    model = AutoModelForImageClassification.from_pretrained(
        "DunnBC22/dit-base-Business_Documents_Classified_v2"
    )

    dataset_eval, dataset_train = torch.utils.data.random_split(dataloader, [0.2, 0.8])

    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_eval = DataLoader(dataset_eval, batch_size=32, shuffle=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])
    ensemble = Ensemble(768, 128)

    if torch.cuda.is_available():
        model.cuda()
        ensemble.cuda()

    for param in model.parameters():
        param.requires_grad = False

    for param in ensemble.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, ensemble.parameters()), lr=0.001
    )
    num_epochs = 5

    for epoch in range(num_epochs):
        running_loss = 0
        for (img_features, text_features), train_labels in tqdm(dataloader_train):
            optimizer.zero_grad()
            img_features = img_features.to(device)
            text_features = text_features.to(device)
            outputs = model(img_features)["pooler_output"].squeeze()
            outputs = ensemble(outputs, text_features.squeeze())
            train_labels = train_labels.type(torch.float32)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        acc = 0
        length = 0
        for (img_features, text_features), test_labels in dataloader_eval:
            with torch.no_grad():
                img_features = img_features.to(device)
                text_features = text_features.to(device)
                outputs = model(img_features)["pooler_output"].squeeze()
                outputs = ensemble(outputs, text_features.squeeze())
                acc += outputs == test_labels
                length += len(outputs)
        print(f"acc for batch: ", acc / length)

        al = running_loss / len(dataloader_train)
        print(f"Running loss: {al}")
        torch.save(ensemble, f"../ensemble{epoch}")


if __name__ == "__main__":
    img_pipeline()
    ensemble_pipeline()
