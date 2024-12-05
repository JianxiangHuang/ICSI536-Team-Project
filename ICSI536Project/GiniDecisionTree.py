import numpy

from TreeNode import TreeNode


# this class is the Gini version decision tree algorithm class
class GiniDecisionTree:
    # this class need to initialize
    # dateset is the training set
    # max_deep is how deep the tree is allowed to grow, default is 5
    def __init__(self, dataset, max_deep=5):
        self.max_deep = max_deep
        self.dataset = dataset
        self.unique_labels = numpy.unique(dataset[:, -1])

    # this method is used to calculate ginivalue
    def calculate_ginivalue(self, dataset):
        # separate the labels part
        labels = dataset[:, -1]
        # calculate the number of data exist in this dataset
        total = len(labels)
        # use numpy.unique to calculate the number of each kind of label within the dataset
        unique_labels, label_counts = numpy.unique(labels, return_counts=True)
        # calculate the proportion of each labels
        probabilities = label_counts / total
        # calculate the Gini value
        gini_value = 1 - numpy.sum(numpy.square(probabilities))
        # print(f"total: {total}\nlabel_counts: {label_counts}\nprobabilities: {probabilities}\ngini: {gini_value}")
        return gini_value

    # this method split the current node into two child nodes
    # this method try every possible way to split the current dataset and find the one with smallest gini value
    # and then check if this split result a gini gain compair with its parent's gini value
    # gini_base is require to input, use the default value 1 only for root of the tree
    def spliting(self, dataset):
        # timer = 0;
        # used to store every possible split's position and gini value
        # use position as key and gini as value
        # position is a tuple, first element is the feature used to split and the second element is the value to compare
        dict_ = {}
        # the number of data existed in the dataset
        total = dataset.shape[0]

        # for a 4 features dataset, this loop will run 100 times to find the best way to split the dataset
        # try every feature, exclude the label column
        for j in range(dataset.shape[1] - 1):
            # get the range list that will use to split and get it's gini value
            # use getRange() method
            range_ = self.getRange(dataset[:, j])
            # try every value in the range list
            for i in range_:
                # get the left and right array that depand on the value
                left_array = dataset[dataset[:, j] < i]
                right_array = dataset[dataset[:, j] >= i]
                # print(f"left_array for j {j} and i {i} is :{left_array}")
                # print(f"right_array for j {j} and i {i} is :{right_array}")
                # timer = timer + 1
                # print(timer)
                # calculate the proportion ratio of the two array
                left_ratio = left_array.shape[0] / total
                right_ratio = right_array.shape[0] / total
                # calculate the gini total which is the weighted total gini value of the two array
                gini_total = left_ratio * self.calculate_ginivalue(left_array) + right_ratio * self.calculate_ginivalue(
                    right_array)
                # record it into the dictionary
                dict_[(j, i)] = gini_total

        # find the smallest gini value and it's position in this dictionary
        min_gini_position = min(dict_, key=dict_.get)
        min_gini = dict_[min_gini_position]
        # print(f"min gini is {min_gini} at position {min_gini_position}")

        # calculate the gini gain
        gini_current = min_gini
        gini_base = self.calculate_ginivalue(dataset)
        gini_gain = gini_base - gini_current

        # if it is beneficial with a positive gini gain, then return the split info
        if gini_gain > 0:
            j, i = min_gini_position[0], min_gini_position[1]
            left_array = dataset[dataset[:, j] < i]
            right_array = dataset[dataset[:, j] >= i]
            return min_gini_position, left_array, right_array
        else:
            return None, None, None

    # this method return a list that used to split by a feature
    # the range_ parameter is the feature column
    def getRange(self, range_):
        range_ = sorted(set(range_))
        # find the max and min value within the ordered list
        min_value = range_[0]
        max_value = range_[-1]
        # use linspace to generate 25 equal spacing split point
        split_points = numpy.linspace(min_value, max_value, 25)
        # round it to 1 decimal and drop duplicate
        split_points_rounded = numpy.unique(numpy.round(split_points, 1))
        return split_points_rounded

    # recursively generate the tree until the tree reach the max deep or complete separated
    # dataset is the only required input, and it is the training dataset
    def generate_model(self, dataset, current_deep=1, nodes=None):
        if nodes is None:
            nodes = TreeNode()
        # record the median label in this node
        labels = dataset[:, -1]
        median_label_value = numpy.median(labels)
        # print("label is "+str(median_label_value))
        nodes.label = median_label_value
        nodes.proportion = self.calculate_label_distribution(labels)
        # check if reached the max deep
        if current_deep <= self.max_deep:
            # split the current dataset
            min_gini_position, left_array, right_array = self.spliting(dataset)
            # print(f"dataset is \n{dataset}\n")
            # print((f"left_array is \n{left_array}\n"))
            # print(f"right_array is \n{right_array}\n")
            # if it is not None, record position into this Node and generate left and right next level nodes
            if min_gini_position is not None:
                nodes.position = min_gini_position
                leftnode = TreeNode()
                nodes.set_leftnode(leftnode)
                self.generate_model(left_array, current_deep + 1, leftnode)
                rightnode = TreeNode()
                nodes.set_rightnode(rightnode)
                self.generate_model(right_array, current_deep + 1, rightnode)
            else:
                return
        return nodes

    # this method is used by generate_model to calculate the label distribution within this node
    def calculate_label_distribution(self, dataset):
        label_distribution = {label: 0 for label in self.unique_labels}
        unique_labels, label_counts = numpy.unique(dataset, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            label_distribution[label] = count
        return label_distribution

    # this method is used to test the test set
    # def test_model(self, model, dataset):
    #     score = 0;
    #     total = dataset.shape[0]
    #     # test each row of data in the test set
    #     for unit in dataset:
    #         actual_label = unit[-1]
    #         predicted_label = self.test_unit(model, unit[:-1])
    #         if (predicted_label == actual_label):
    #             # print(f"predicted label is {predicted_label} and actual label is {actual_label}")
    #             score += 1
    #     # calculate the accuracy and
    #     accuracy = score / total
    #     print(f"My accuracy is : {accuracy * 100:.2f}%")

    def test_model(self, model, dataset):
        score = 0
        total = dataset.shape[0]
        # test each row of data in the test set
        for unit in dataset:
            actual_label = unit[-1]
            predicted_label = self.test_unit(model, unit[:-1])
            print(f"actual label is {actual_label} and the predicted label is {predicted_label}")
            if predicted_label == actual_label:
                score += 1
        # calculate the accuracy and
        accuracy = score / total*100
        print(f"My accuracy is : {accuracy:.2f}%")

    # this method is used by method test_model
    # it returns the prediction or Recursion until get the result
    def test_unit(self, model, test_unit):

        # if reach the leaf of the tree, return the result
        if model.position is None:
            return model.label

        # if it is not the leaf, decide go left or right
        feature_index, feature_value = model.position
        # print(f"index is {feature_index}, value is {feature_value}")
        # print(test_unit)
        if test_unit[feature_index] < feature_value:
            return self.test_unit(model.leftnode, test_unit)
        else:
            return self.test_unit(model.rightnode, test_unit)
