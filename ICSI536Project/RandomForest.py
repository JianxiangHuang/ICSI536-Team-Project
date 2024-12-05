from statistics import mode

import numpy

from ProcessDataSet import ProcessDataSet
from GiniDecisionTree import GiniDecisionTree
from TreeNode import TreeNode


# this class is the randomForest class
class RandomForest:
    # this class need to initialize
    # dateset is the training set
    # tree_amount is the scale of the forest, or amount of tree, default as 100
    # max_deep is how deep the tree is allowed to grow, default is 5
    # feature_fraction is the amount of feature selected for each tree to train, default as square root of feature amount
    # row_fraction is the proportion of data selected for each tree to train
    def __init__(self, dataset, tree_amount=100, max_deep=5, feature_fraction=None, row_fraction=0.8, random_seed=None):
        self.tree_amount = tree_amount
        self.dataset = dataset
        self.max_deep = max_deep
        self.feature_fraction = feature_fraction
        self.row_fraction = row_fraction
        self.random_seed = random_seed

    # this method is sued to train the forest
    def generate_forest(self):
        # list to contain all the trained trees
        forest = []
        # a loop run tree_amount time to train that amount of tree
        for i in range(self.tree_amount):
            # this is a method that return the random selected sub set of data and the feature info relate to what feature selected
            sample, feature = ProcessDataSet.random_select_rows_and_features(self.dataset, self.feature_fraction,
                                                                             self.row_fraction, self.random_seed)
            # train the tree use GiniDecisionTree class's method
            model = GiniDecisionTree(sample, self.max_deep)
            node = model.generate_model(sample)
            # check the tree's performance, if the accuracy >0.5, then add it to the forest
            test_result = self.test_tree_performance(node, self.dataset, feature)
            if test_result:
                forest.append((node, feature))
                # nodes.print_tree(nodes)
                print(i)
        return forest

    # this method is used to check a tree's performance on the training dataset
    def test_tree_performance(self, node, dataset, feature):
        score = 0
        total = dataset.shape[0]
        # test each row of data in the test set
        for unit in dataset:
            actual_label = unit[-1]
            predicted_label = self.test_unit(node, unit[:-1], feature)
            # print(f"actual label is {actual_label} and the predicted label is {predicted_label}")
            if predicted_label == actual_label:
                score += 1
        accuracy = score / total
        return accuracy > 0.5

    # this method is used to predict the result on test set and return the accuracy
    def predict(self, forest, testset):
        score = 0
        total = testset.shape[0]
        # each row of data
        for unit in testset:
            results = []
            actual_label = unit[-1]
            # loop over the forest, append the result of each tree into a list
            for j in forest:
                node, feature = j
                result = self.test_unit(node, unit[:-1], feature)
                results.append(result)
            # find the mode number within the list
            predict_label = mode(results)
            # print the result
            print(f"actual label is {actual_label} and the predicted label is {predict_label}")
            print(results)

            if predict_label == actual_label:
                score += 1
        accuracy = score / total*100
        print(f"My algorithm's accuracy is : {accuracy:.2f}%")

    # this method is used by method predict and test_tree_performance to check if the prediction is correct or not
    # it returns the prediction or Recursion until get the result
    def test_unit(self, model, test_unit, features):

        # if reach the leaf of the tree, return the result
        if model.position is None:
            return model.label

        # if it is not the leaf, decide go left or right
        feature_index, feature_value = model.position
        if test_unit[features[feature_index]] < feature_value:
            return self.test_unit(model.leftnode, test_unit, features)
        else:
            return self.test_unit(model.rightnode, test_unit, features)
