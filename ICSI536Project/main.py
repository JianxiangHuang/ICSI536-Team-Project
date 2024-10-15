import numpy

from GiniDecisionTree import GiniDecisionTree
from ProcessDataSet import ProcessDataSet


from sklearn import tree
from sklearn.metrics import accuracy_score
def verify():
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
    print(split_structure)
    y_pred = clf.predict(test_set[:,:-1])
    accuracy = accuracy_score(test_set[:,-1], y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    train_set, test_set= ProcessDataSet.split_dataset_to_traindata_and_testdata("datasets/iris_dataset.csv",0.1,10)

    # ProcessDataSet.print_dataset(train_set,"Train Dataset")
    #
    # ProcessDataSet.print_dataset(test_set,"Test Dataset")

    model=GiniDecisionTree(train_set, 4)
    nodes=model.generate_model(model.dataset)
    nodes.print_tree(nodes)
    model.test_model(nodes, test_set)
    # model.calculate_ginivalue(train_set)
    # model.spliting(train_set)

    verify()


# 模拟 iris 数据集
# 随机生成数据集的函数
def generate_noisy_data(num_samples=100):
    numpy.random.seed(42)  # 固定随机种子以便复现结果

    # 特征1: 随机生成 [0, 10) 之间的浮点数
    feature1 = numpy.random.uniform(0, 10, num_samples)

    # 特征2: 随机生成 [0, 100) 之间的浮点数
    feature2 = numpy.random.uniform(0, 100, num_samples)

    # 标签: 基于特征1的值进行分类，若 feature1 > 5，则标签为1，否则为0
    labels = (feature1 > 5).astype(int)

    # 为10%的样本标签添加噪声，翻转标签（1变0，0变1）
    num_noisy_samples = int(0.1 * num_samples)
    noise_indices = numpy.random.choice(num_samples, num_noisy_samples, replace=False)
    labels[noise_indices] = 1 - labels[noise_indices]  # 标签翻转

    # 将数据合并成一个数组 (特征1, 特征2, 标签)
    dataset = numpy.column_stack((feature1, feature2, labels))

    return dataset


# 生成数据集
dataset = generate_noisy_data(100)

# 将数据集拆分为训练集和测试集 (80%训练, 20%测试)
train_set = dataset[:80]  # 前80个样本用于训练
test_set = dataset[80:]  # 剩余20个样本用于测试

# 使用 GiniDecisionTree 进行训练和测试
model = GiniDecisionTree(train_set, 2)  # 假设第3列是标签列
nodes = model.generate_model(train_set)
nodes.print_tree(nodes)  # 打印生成的决策树结构
model.test_model(nodes, test_set)  # 测试模型
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_set[:, :-1], train_set[:, -1])
y_pred = clf.predict(test_set[:, :-1])
accuracy = accuracy_score(test_set[:, -1], y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")



