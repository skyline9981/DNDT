import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn import tree
from connect4_loader import get_connect4_data
from german_loader import get_german_data
import time
import random

if __name__ == "__main__":
    np.random.seed(1943)
    torch.manual_seed(1943)

    seed = random.seed(1990)
    _x, _y = get_german_data()
    # _x, _y = get_connect4_data()
    _y = _y.reshape(-1, 1)
    print(_x.shape, _y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        _x, _y, train_size=0.70, random_state=seed
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    clf = tree.DecisionTreeClassifier()

    start_time = time.time()

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("acc= ", acc)
    print("--- %s seconds ---" % (time.time() - start_time))
    print("====================================================================")
    print("DT: ", classification_report(y_test, y_pred))

    # 画图
    plt.figure(figsize=(8, 8))

    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=np.argmax(y_test, axis=1),
        marker="o",
        s=50,
        cmap="summer",
        edgecolors="black",
    )
    cm = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

    plt.show()
