import numpy

from TreeNode import TreeNode


class GiniDecisionTree:
    def __init__(self, dataset, label_index, max_deep=5):
        self.dataset = dataset
        self.label = label_index
        self.max_deep=max_deep

    def calculate_ginivalue(self, dataset):
        labels=dataset[:,self.label]
        total=len(labels)
        # 使用 numpy.unique 计算每个标签的计数
        unique_labels, label_counts = numpy.unique(labels, return_counts=True)

        probabilities=label_counts/total
        gini_value=1-numpy.sum(numpy.square(probabilities))
        print(f"total: {total}\nlabel_counts: {label_counts}\nprobabilities: {probabilities}\ngini: {gini_value}")
        return gini_value

    def spliting(self, dataset, gini_base=1):
        timer=0;
        dict_={}
        total=dataset.shape[0]
        for j in range(dataset.shape[1]-1):
            #  获取特征 dataset[:, j] 中所有唯一值并按升序排列
            range_=self.getRange(sorted(set(dataset[:,j])))
            for i in range_:
                left_array = dataset[dataset[:, j] < i]
                right_array = dataset[dataset[:, j] >= i]
                print(f"left_array for j {j} and i {i} is :{left_array}")
                print(f"right_array for j {j} and i {i} is :{right_array}")
                timer=timer+1
                print(timer)
                left_ratio=left_array.shape[0]/total
                right_ratio=right_array.shape[0]/total
                gini_total= left_ratio * self.calculate_ginivalue(left_array) +right_ratio* self.calculate_ginivalue(right_array)
                dict_[(j,i)]=gini_total

        min_gini_position=min(dict_, key=dict_.get)
        min_gini=dict_[min_gini_position]
        print(f"min gini is {min_gini} at position {min_gini_position}")

        gini_current=min_gini
        gini_gain=gini_base-gini_current
        gini_base=gini_current

        if gini_gain>0:
            j, i=min_gini_position[0], min_gini_position[1]
            left_array = dataset[dataset[:, j] < i]
            right_array = dataset[dataset[:, j] >= i]
            return min_gini_position, gini_base,left_array, right_array
        else:
            return None, None, None, None


    def getRange(self, range_):
        # 计算 range_ 的最小值和最大值
        min_value = range_[0]
        max_value = range_[-1]
        # 使用 np.linspace 生成 25 份等间隔的数列
        split_points = numpy.linspace(min_value, max_value, 25)
        # 将数列的值保留到小数点后一位，并去重
        split_points_rounded = numpy.unique(numpy.round(split_points, 1))
        return split_points_rounded

    def generate_model(self, dataset, gini_base=1, current_deep=1, nodes=TreeNode()):
        min_gini_position, gini_base,left_array, right_array =self.spliting(dataset, gini_base)
        if min_gini_position is not None:
            left_median_label_value=numpy.median(left_array[:,-1])
            nodes.set_leftnode(TreeNode(left_median_label_value,min_gini_position))
            right_median_label_value=numpy.median(right_array[:,-1])
            nodes.set_rightnode(TreeNode(right_median_label_value,min_gini_position))
        else:
            return

        if current_deep<=self.max_deep and len(set(left_array.flatten()))!=1 and len(set(left_array[:,self.label]))!=1:
            self.generate_model(left_array, gini_base, current_deep+1,nodes.get_leftnode())
        if current_deep<=self.max_deep and len(set(right_array.flatten()))!=1 and len(set(right_array[:,self.label]))!=1:
            self.generate_model(right_array, gini_base, current_deep+1,nodes.get_rightnode())
        return nodes


    def test_model(self, model,dataset):
        score=0;
        total=dataset.shape[0]

        for unit in dataset:
            actual_label=unit[-1]
            print("test unit is ")
            predicted_label=self.test_unit(model,unit[:-1])
            if(predicted_label == actual_label):
                print(f"predicted lable is {predicted_label} and actual label is {actual_label}")
                score+=1

        accuarcy=score/total
        print(f"accuracy is : {accuarcy * 100:.2f}%")


    def test_unit(self, model, test_unit):

        # 如果当前节点是叶子节点，返回该节点的标签
        if model.leftnode is None and model.rightnode is None:
            return model.label

        left_node=model.get_leftnode()

        feature_index, feature_value=left_node.position
        print(f"index is {feature_index}, value is {feature_value}")
        print(test_unit)
        if test_unit[feature_index] < feature_value:
            return self.test_unit(model.leftnode, test_unit)
        else:
            return self.test_unit(model.rightnode, test_unit)