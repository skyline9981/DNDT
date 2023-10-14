import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import OneHotEncoder
from connect4_loader import get_connect4_data
from german_loader import get_german_data
import random
import warnings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DNDT_Classifier(nn.Module):
    def __init__(self, num_cut, num_class, epoch, temperature):
        super(DNDT_Classifier, self).__init__()
        self.num_cut = num_cut
        self.num_leaf = np.prod(np.array(num_cut) + 1)
        self.num_class = num_class
        self.temperature = torch.tensor(temperature)
        self.cut_points_list = [
            torch.rand([i], requires_grad=True, device=device) for i in num_cut
        ]
        self.leaf_score = torch.rand(
            [self.num_leaf, self.num_class], requires_grad=True, device=device
        )
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.cut_points_list + [self.leaf_score], lr=0.01
        )
        self.epoch = epoch
        self.temperature = self.temperature.to(device)
        self.cut_points_list = [cp.to(device) for cp in self.cut_points_list]
        self.leaf_score = self.leaf_score.to(device)

    # 计算克罗内克积
    def torch_kron_prod(self, a, b):
        res = torch.einsum("ij,ik->ijk", [a, b])
        res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
        return res.to(device)

    # 软分箱算法
    def torch_bin(self, x, cut_points, temperature):
        # x is a N-by-1 matrix (column vector)
        # cut_points is a D-dim vector (D is the number of cut-points)
        # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
        D = cut_points.shape[0]
        W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1]).to(device)
        # make sure cut_points is monotonically increasing
        cut_points, _ = torch.sort(cut_points)
        b = torch.cumsum(
            torch.cat([torch.zeros([1], device=device), -cut_points], 0), 0
        )
        h = torch.matmul(x, W) + b
        h = h / self.temperature

        res = F.softmax(h, dim=1)
        return res

    # 建树
    def nn_decision_tree(self, x):
        # cut_points_list contains the cut_points for each dimension of feature
        leaf = reduce(
            self.torch_kron_prod,
            map(
                lambda z: self.torch_bin(x[:, z[0] : z[0] + 1], z[1], self.temperature),
                enumerate(self.cut_points_list),
            ),
        )
        return torch.matmul(leaf, self.leaf_score)

    def fit(self, X_train, y_train):
        x = torch.from_numpy(X_train.astype(np.float32)).to(device)
        y = torch.from_numpy(y_train.astype(np.int64)).squeeze().to(device)
        for i in range(1000):
            self.optimizer.zero_grad()
            y_pred = self.nn_decision_tree(x)
            loss = self.loss_function(y_pred, y)
            loss.backward()
            self.optimizer.step()
            if i % 200 == 0:
                print("epoch %d loss= %f" % (i, loss.cpu().detach().numpy()))
        print(
            "error rate %.2f"
            % (
                1
                - np.mean(
                    np.argmax(y_pred.cpu().detach().numpy(), axis=1)
                    == y_train.astype(np.int64)
                )
            )
        )
        return y_pred

    def predict(self, X_test):
        x = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_pred = self.nn_decision_tree(x)
        return y_pred

    def get_params(self, deep=True):
        out = dict()
        for key in self._get_param_names():
            try:
                value = getattr(self, key)
            except AttributeError:
                warnings.warn(
                    "From version 0.24, get_params will raise an "
                    "AttributeError if a parameter cannot be "
                    "retrieved as an instance attribute. Previously "
                    "it would return None.",
                    FutureWarning,
                )
                value = None
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out


class DNDT_Ensemble:
    def __init__(self, n_trees, num_cut, num_class, epoch, temperature):
        self.n_trees = n_trees
        self.trees = [
            DNDT_Classifier(num_cut, num_class, epoch, temperature)
            for _ in range(n_trees)
        ]
        self.trees = [tree.to(device) for tree in self.trees]
        self.feature_indices = []

    def fit(self, X_train, y_train):
        n_features = X_train.shape[1]
        for tree in self.trees:
            # Randomly select 10 features for each tree
            indices = np.random.choice(n_features, 10, replace=False)
            self.feature_indices.append(indices)
            print("indices: ", indices)
            print("Tree index: ", self.trees.index(tree))
            tree.fit(X_train[:, indices], y_train)

        y_train_pred = self.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        print("Train Accuracy:", train_acc)

    def predict(self, X_test):
        predictions = []
        for tree, indices in zip(self.trees, self.feature_indices):
            y_pred = tree.predict(X_test[:, indices])
            predictions.append(np.argmax(y_pred.cpu().detach().numpy(), axis=1))
        # Majority voting
        predictions = np.array(predictions)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )
        return final_predictions


if __name__ == "__main__":
    # np.random.seed(1943)
    # torch.manual_seed(1943)

    # seed = random.seed(1990)
    # _x, _y = get_german_data()
    _x, _y = get_connect4_data()

    X_train, X_test, y_train, y_test = train_test_split(_x, _y, train_size=0.70)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    d = X_train.shape[1]
    num_cut = [1, 1, 1, 1, 1, 1, 1, 1]  # "Petal length" and "Petal width"
    num_class = 4
    epoch = 1000
    temperature = 0.1

    # 1. 初始化模型
    n_trees = 10
    ensemble_model = DNDT_Ensemble(n_trees, num_cut, num_class, epoch, temperature)

    # 2. 擬合數據
    ensemble_model.fit(X_train, y_train)

    # 3. 預測
    y_pred_ensemble = ensemble_model.predict(X_test)

    # 計算精確度
    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    print("Ensemble Accuracy:", acc_ensemble)

    # cm = ConfusionMatrixDisplay.from_predictions(y_test[:, 0], y_pred_ensemble)
    # plt.show()
