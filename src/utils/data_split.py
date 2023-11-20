import pandas as pd
import os


def save_split_files(X_, y_, test=False):
    y = pd.DataFrame(data=y_, columns=["y"])

    X_.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)

    df = pd.concat([X_, y], axis=1, sort=False)
    if test:
        ext = "_test.csv"
    else:
        ext = "_train.csv"

    file_path = "./" + ext
    df.to_csv(file_path, index=False)

    print(f"Saved file to: {os.path.abspath(file_path)}")
