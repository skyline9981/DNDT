import pandas as pd
import torch


def get_connect4_data():
    # Load the dataset
    df = pd.read_csv("./connect4/connect-4.data", header=None)

    # Split features and labels
    x = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values  # Only the last column

    # Convert the data type to string
    x = x.astype(str)
    y = y.astype(str)

    # Replace the values in x
    x[x == "b"] = 0
    x[x == "x"] = 1
    x[x == "o"] = 2

    # Convert x back to integer type after replacement
    x = x.astype(int)

    # Replace the values in y
    y[y == "win"] = 1
    y[y == "loss"] = 2
    y[y == "draw"] = 0

    # Convert y back to integer type after replacement
    y = y.astype(int)
    # Convert them to PyTorch tensors
    # x = torch.tensor(x, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.long)  # Assuming labels are integers

    return x, y


if __name__ == "__main__":
    x, y = get_connect4_data()
