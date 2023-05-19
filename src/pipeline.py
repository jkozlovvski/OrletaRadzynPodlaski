import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from dataclass import ImageDataSet, EnsembleDataset
from transformers import ViTImageProcessor, AutoModelForImageClassification
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


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
    dataloader = DataLoader(image_data_set, batch_size=32, shuffle=True)

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
    num_epochs = 10
    losses = []
    for epoch in range(num_epochs):
        print("epoch: ", epoch)
        running_loss = 0
        for train_features, train_labels in tqdm(dataloader):
            optimizer.zero_grad()
            train_features = train_features.to(device)
            train_labels = train_labels.to(device)
            outputs = model(train_features).logits
            train_labels = train_labels.type(torch.float32)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        al = running_loss / len(image_data_set)
        losses.append(al)
        print("running_loss: ", running_loss)
    torch.save(model.state_dict(), "model")


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

    print(f"Length of ensemble dataset: {len(ensemble_dataset.images)}")
    return


if __name__ == "__main__":
    ensemble_pipeline()
