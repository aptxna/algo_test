import os, sys
import numpy as np
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log


class DecisionTreeID3:
    def __init__(self, df):
        self.df = df
        self.label = df.keys()[-1]
        self.tree = self._build_tree()

    def dataset_entropy(self, df):
        _, counts = np.unique(df[self.label], return_counts=True)
        probabilities = counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))
        return entropy

    def conditional_entropy(self, feature, df):
        result = 0
        variables = df[feature].unique()
        for variable in variables:
            _, counts = np.unique(df[self.label][df[feature]==variable], return_counts=True)
            probabilities = counts / counts.sum()
            entropy = sum(probabilities * -np.log2(probabilities))
            fraction = len(df[feature][df[feature]==variable]) / len(df)
            result += fraction * entropy
        return result

    def find_maximum_infogain(self, df):
        infogain = []
        for key in df.keys()[:-1]:
            infogain.append(self.dataset_entropy(df) - self.conditional_entropy(key, df))
        return df.keys()[:-1][np.argmax(infogain)]

    def get_subset(self, df, node, value):
        return df[df[node]==value].reset_index(drop=True)

    def _build_tree(self, df=None):
        if df is None:
            df = self.df
        tree = {}
        node = self.find_maximum_infogain(df)
        feature_values = np.unique(df[node])
        for value in feature_values:
            subset = self.get_subset(df, node, value)
            label_values, label_counts = np.unique(subset[self.label], return_counts=True)
            if len(label_counts) == 1:
                tree[node+'=='+value] = label_values[0]
            else:
                tree[node+'=='+value] = self._build_tree(subset)
        return tree

    def print_all_paths_root2leaf(self):
        def helper(tree, cur=()):
            if isinstance(tree, dict):
                for n, s in tree.items():
                    for path in helper(s, cur+(n,)):
                        yield path
            else:
                yield cur+(tree,)
        return helper(self.tree)

    def test_one_sample(self, sample, tree=None):
        if tree is None:
            tree = self.tree
        feature_name = list(tree.keys())[0].split('==')[0]
        values = [e.split('==')[1] for e in tree.keys()]
        subtree_index = values.index(sample[feature_name])
        subtree_value = tree[list(tree.keys())[subtree_index]]
        if isinstance(subtree_value, str):
            return subtree_value
        else:
            return self.test_one_sample(sample, subtree_value)


if __name__ == "__main__":
    outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(',')
    temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(',')
    humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(',')
    windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(',')
    play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')

    dataset ={'outlook':outlook,'temp':temp,'humidity':humidity,'windy':windy,'play':play}
    df = pd.DataFrame(dataset,columns=['outlook','temp','humidity','windy','play'])
    print(df), print("\n")

    id3 = DecisionTreeID3(df)
    print(id3.tree), print("\n")
    print(list(id3.print_all_paths_root2leaf())), print("\n")

    sample = df.iloc[0]
    prediction = id3.test_one_sample(sample)
    print(prediction), print("\n")