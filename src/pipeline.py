import numpy as np
from utils import id2label, label2id
from dataclass import text_dataset_train, image_dataset_train
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# simple pipeline for models
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


if __name__ == "__main__":
    clf = RandomForestClassifier()
    cross_validation_training(clf, text_dataset_train)
