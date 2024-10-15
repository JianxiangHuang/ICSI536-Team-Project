# this class is used to store the tree structure as the decision tree
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
        if node is None:
            return
        indent = " " * (level * 4)
        print(f"{indent}Node (Label: {node.label}, Position: {node.position})")
        if node.leftnode is not None:
            print(f"{indent}Left:")
            self.print_tree(node.leftnode, level + 1)
        if node.rightnode is not None:
            print(f"{indent}Right:")
            self.print_tree(node.rightnode, level + 1)