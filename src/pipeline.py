import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from dataclass import text_dataset_train


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


# works like shit
def bertopic_pipeline(dataset):
    # as bertopic can take whole strings as input
    # there is a separate pipeline for it
    empty_dimensionality_model = BaseDimensionalityReduction()
    clf = LogisticRegression()
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    # Create a fully supervised BERTopic instance
    topic_model = BERTopic(
        umap_model=empty_dimensionality_model,
        hdbscan_model=clf,
        ctfidf_model=ctfidf_model,
    )
    kf = KFold(n_splits=10)
    for i, (train_index, test_index) in enumerate(kf.split(dataset)):
        print(f"Fold {i}:")
        X_train, y_train = [], []
        X_test, y_test = [], []
        for idx in train_index:
            value, target = dataset[idx]
            X_train.append(value)
            y_train.append(target)

        for idx in test_index:
            value, target = dataset[idx]
            X_test.append(value)
            y_test.append(target)
        topic_model.fit(X_train, y=y_train)
        predicted = topic_model.transform(X_test)[1]
        print(f"Accuracy score: {accuracy_score(y_test, predicted)}")


if __name__ == "__main__":
    clf = RandomForestClassifier()
    dataset = text_dataset_train
    dataset.preprocess_text()
    cross_validation_training(clf, dataset)
