import pandas as pd
from utils import save_split_files
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
import GA

# This is a standard machine learning dataset from the UCI Machine Learning repository
df = datasets.load_iris()
X = df.data
y_ = df.target.reshape(-1, 1)  # Convert data to a single column
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# sepal length, sepal width, petal length, petal width and species
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)


# Splitting data in 10-Fold
X_ = pd.DataFrame(data=X, columns=["p1", "p2", "p3", "p4"])

# Instantiating the K-Fold cross validation object with 5 folds
k_folds = KFold(n_splits=10, shuffle=True, random_state=42)

filename = 0
# Iterating through each of the folds in K-Fold
for train_index, val_index in k_folds.split(X_):
    # Splitting the training set from the validation set for this specific fold
    X_train, X_val = X_.iloc[train_index, :], X_.iloc[val_index, :]
    y_train, y_val = y_[train_index], y_[val_index]
    save_split_files(X_train, y_train, str(filename))
    save_split_files(X_val, y_val, str(filename), test=True)
    filename += 1

# Actual Execution

df_X_train = pd.DataFrame(X_train)
n_bundles = 200
n_rules = 30
generations = 100
r_cross = 0.1
r_mut = 0
n_class = 3
n_train = 20
n_parents = 20
n_children = 10
input = pd.DataFrame(X_test)
obj = y_test
interval = 3
rule_evolve = True
two_best = False

after = 0
d_rate = 1
every = 0

dampening = (after, d_rate, every)  # (Start pos of Damepning, damepning rate, steps)
test = GA(
    n_bundles,
    n_rules,
    generations,
    r_cross,
    r_mut,
    n_class,
    n_train,
    n_parents,
    n_children,
    interval,
    rule_evolve,
    two_best,
    dampening=dampening,
)
model, score = test.run_model(df_X_train, y_train, input, obj)
