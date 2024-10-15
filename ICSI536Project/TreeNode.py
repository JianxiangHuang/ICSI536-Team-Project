
class TreeNode:
    def __init__(self, label=None,position=None,leftnode=None, rightnode=None, ):
        self.label=label
        self.position = position
        self.leftnode=leftnode
        self.rightnode=rightnode

    def set_leftnode(self, leftnode):
        self.leftnode=leftnode

    def set_rightnode(self, rightnode):
        self.rightnode=rightnode

    def get_leftnode(self):
        return self.leftnode

    def get_rightnode(self):
        return self.rightnode

    def print_tree(self,node, level=0):
        # 基础终止条件：当节点为空时，返回
        if node is None:
            return

        # 打印当前节点的内容，显示深度和节点的标签或位置
        indent = " " * (level * 4)  # 根据当前深度添加缩进
        print(f"{indent}Node (Label: {node.label}, Position: {node.position})")

        # 打印左子树
        if node.leftnode is not None:
            print(f"{indent}Left:")
            self.print_tree(node.leftnode, level + 1)

        # 打印右子树
        if node.rightnode is not None:
            print(f"{indent}Right:")
            self.print_tree(node.rightnode, level + 1)