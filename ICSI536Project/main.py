import numpy
from sklearn.ensemble import RandomForestClassifier

from GiniDecisionTree import GiniDecisionTree
from ProcessDataSet import ProcessDataSet

from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report

from graphviz import Digraph

from RandomForest import RandomForest


# use sklearn to verify the dataset
# print the accuracy of sklearn decision tree algorithm performance on the dataset
def sklearn_verify_decisiontree():
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_set[:, :-1], train_set[:, -1])
    tree.plot_tree(clf)
    # 获取决策树的特征和阈值
    features = clf.tree_.feature  # 节点使用的特征索引
    thresholds = clf.tree_.threshold  # 节点使用的阈值
    # 初始化一个列表来存储特征和对应的阈值
    split_structure = []
    # 遍历每个节点并记录特征索引和划分阈值
    for i in range(len(features)):
        if features[i] != -2:  # -2 表示这是叶子节点，不再进行划分
            split_structure.append((features[i], thresholds[i]))
    # 打印结构 [(特征索引, 阈值), ...]
    # print(split_structure)
    y_pred = clf.predict(test_set[:, :-1])
    accuracy = accuracy_score(test_set[:, -1], y_pred)
    print(f"Sklearn Accuracy: {accuracy * 100:.2f}%")


# use sklearn to verify the random forest
# print the accuracy of sklearn random forest algorithm performance on the dataset
def sklearn_verify_randomforest():
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(train_set[:, :-1], train_set[:, -1])
    y_pred = rf_classifier.predict(test_set[:, :-1])
    accuracy = accuracy_score(test_set[:, -1], y_pred)*100
    print(f"Sklearn Accuracy: {accuracy:.2f}%")


# this method is used to draw the decision tree, not important for this project's two models
def add_nodes_edges(dot, node):
    if node is not None:
        dot.node(str(id(node)), f"Label: {node.label}, \nPosition: {node.position}, \nProportion: {node.proportion})",
                 shape='box')

        if node.leftnode:
            dot.node(str(id(node.leftnode)),
                     f"Label: {node.leftnode.label}, \nPosition: {node.leftnode.position}, \nProportion: {node.leftnode.proportion})",
                     shape='box')
            dot.edge(str(id(node)), str(id(node.leftnode)))
            add_nodes_edges(dot, node.leftnode)

        if node.rightnode:
            dot.node(str(id(node.rightnode)),
                     f"Label: {node.rightnode.label}, \nPosition: {node.rightnode.position}, \nProportion: {node.rightnode.proportion})",
                     shape='box')
            dot.edge(str(id(node)), str(id(node.rightnode)))
            add_nodes_edges(dot, node.rightnode)


# run the main method to verify the algorithm
# I used two dataset in this project, iris_dataset.csv and creditcard2.csv
# unmark the code section to run the program
if __name__ == '__main__':

    # this part is used to generate the decision tree for the dataset inputted

    # split the dataset into train set and test set
    train_set, test_set = ProcessDataSet.split_dataset_to_traindata_and_testdata("datasets/iris_dataset.csv", 0.2)
    # train the model
    model = GiniDecisionTree(train_set, 5)
    nodes = model.generate_model(train_set)
    # print the trained model
    # nodes.print_tree(nodes)
    # test the model with the test set
    model.test_model(nodes, test_set)
    # verify the model by train the dataset with ML module and verify if get the similar result
    sklearn_verify_decisiontree()





    # this part is used to generate the random forest for the dataset inputted

    # # split the dataset into train set and test set
    # train_set, test_set = ProcessDataSet.split_dataset_to_traindata_and_testdata("datasets/iris_dataset.csv", 0.2)
    # # train the model
    # model = RandomForest(train_set, 100, 5)
    # forest = model.generate_forest()
    # # test the model with the test set
    # model.predict(forest, test_set)
    # # verify the model by train the dataset with ML module and verify if get the similar result
    # sklearn_verify_randomforest()




    # Not important

    # draw the tree
    # dot = Digraph()
    # add_nodes_edges(dot, nodes)
    # dot.render('tree2', view=True,format='png')
