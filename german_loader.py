import pandas as pd
import torch


def process_row(row):
    for i in range(len(row)):
        row[i] = row[i].replace("A", "")
        if not row[i].isdigit():
            row[i] = int(row[i])


def get_german_data():
    # Load the dataset
    df = pd.read_csv("./german/german.data", header=None)
    df = df[0].str.split(expand=True)

    # Split features and labels
    x = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values  # Only the last column

    # Convert the data type to string
    x = x.astype(str)

    # Replace the values in x
    for row in x:
        process_row(row)

    # Convert x back to integer type after replacement
    x = x.astype(int)

    # Convert y back to integer type after replacement
    y = y.astype(int)
    # Convert them to PyTorch tensors
    # x = torch.tensor(x, dtype=torch.float32)
    # y = torch.tensor(y, dtype=torch.long)  # Assuming labels are integers

    return x, y


if __name__ == "__main__":
    x, y = get_german_data()
    print(x.shape, y.shape)
