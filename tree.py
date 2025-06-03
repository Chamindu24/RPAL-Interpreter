class Tree:
    def __init__(self, value: str = "", node_type: str = ""):
        self.val = value
        self.type = node_type
        self.left_node = None  # type: Tree | None
        self.right_node = None  # type: Tree | None

    # Set the type of the node
    def set_type(self, node_type: str):
        self.type = node_type

    # Set the value of the node
    def set_value(self, value: str):
        self.val = value

    # Get the type of the node
    def get_type(self) -> str:
        return self.type

    # Get the value of the node
    def get_value(self) -> str:
        return self.val

    # Create a new node with a given value and type
    @staticmethod
    def build_node(value: str, node_type: str):
        return Tree(value, node_type)

    # Create a new node based on another node (copy value, type, left child)
    @staticmethod
    def build_node_from_existing(x: 'Tree'):
        new_tree = Tree(x.get_value(), x.get_type())
        new_tree.left_node = x.left_node
        new_tree.right_node = None
        return new_tree

    # Recursively display the syntax tree with indentation
    def display_syntax_tree(self, indentation_level: int = 0):
        print("." * indentation_level, end="")

        if self.type in {"ID", "STR", "INT"}:
            print(f"<{self.type}:{self.val}>")
        elif self.type in {"BOOL", "NIL", "DUMMY"}:
            print(f"<{self.val}>")
        else:
            print(self.val)

        if self.left_node:
            self.left_node.display_syntax_tree(indentation_level + 1)
        if self.right_node:
            self.right_node.display_syntax_tree(indentation_level)
