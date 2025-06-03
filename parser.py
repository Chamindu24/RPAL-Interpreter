from collections import deque
import math
# Standard libraries
import sys
import math
import os
import re
from collections import deque
from typing import List, Dict, Optional
from lexer import Token    
from tree import Tree         
from cse_machine import Environment

# Stack for syntax tree (LIFO)
st = deque()

# Array of single-character operators
operators = ['+', '-', '*', '<', '>', '&', '.', '@', '/', ':', '=', '~', '|', '$', '!', '#', '%',
             '^', '_', '[', ']', '{', '}', '"', '`', '?']

# Array of binary operators
binary_operators = [
    "+", "-", "*", "/", "**", "gr", "ge", "<", "<=", ">", ">=", "ls", "le", "eq", "ne", "&", "or", "><"
]

# Array of keywords
keywords = [
    "let", "fn", "in", "where", "aug", "or", "not", "true", "false", "nil", "dummy", "within",
    "and", "rec", "gr", "ge", "ls", "le", "eq", "ne"
]

class Parser:
    def __init__(self, read_array: str, row: int, size: int, af: int):
        self.next_token: Token = None  # Placeholder, to be set properly in actual parsing logic
        
        self.index = row
        self.tree = Tree()
        self.tree.read_line = read_array[:10000]  # Store up to first 10,000 characters
        self.size_of_file = size
        self.ast_flag = af
        self.parsing_complete = False
        self.cse_flag = 0  # Default value
        self.st = deque()

        
        # Control structure state
        self.control_structures_index = 1
        self.control_structures_column = 0
        self.control_structures_row = 0
        self.control_structures_beta = 1
        self.controlNodeArray = [[None for _ in range(200)] for _ in range(200)]
        self.cse_flag = 0
        

        

    # Check if the given string is a keyword
    def is_reserved_key(self, string: str) -> bool:
        return string in keywords

    # Check if the given character is an operator
    def is_operator(self, ch: str) -> bool:
        return ch in operators

    # Check if the given character is an alphabet letter
    def is_alpha(self, ch: str) -> bool:
        return ch.isalpha()

    # Check if the given character is a digit
    def is_digit(self, ch: str) -> bool:
        return ch.isdigit()

    # Check if the given string is a binary operator
    def is_binary_operator(self, op: str) -> bool:
        return op in binary_operators

    # Check if the given string is a number
    def is_number(self, s: str) -> bool:
        return s.is_digit()
    
    def read(self, val, type):
        #print(f"read: Expecting value={val}, type={type}")
        if val != self.next_token.get_value() or type != self.next_token.get_type():
            #print(f'Parse error: Expected "{val}", but "{self.next_token.get_value()}" was found')
            sys.exit(0)
        
        if type in ("ID", "INT", "STR"):
            self.buildTree(val, type, 0)
            #print(f"read: Pushed node with value={val}, type={type} onto stack")
        
        self.next_token = self.get_token()
        #print(f"read: Next token is now {self.next_token.get_value()} ({self.next_token.get_type()})")
        
        while self.next_token.get_type() == "DELETE":
            self.next_token = self.get_token()
        #print(f"read: Skipped DELETE token, next token is now {self.next_token.get_value()} ({self.next_token.get_type()})")

    def buildTree(self, val, type, child):
        #print(f"buildTree: Building node value={val}, type={type}, children={child}, stack size before: {len(self.st)}")
        if child == 0:
            temp = Tree.build_node(val, type)
            self.st.append(temp)
        elif child > 0:
            temp = []
            no_of_pops = child
            while self.st and no_of_pops > 0:
                temp.append(self.st.pop())
                no_of_pops -= 1
            temp.reverse()  # reverse to maintain order since stack pop reverses order
            
            tempLeft = temp[0]
            child -= 1
            
            if len(temp) > 1 and child > 0:
                rightNode = temp[1]
                tempLeft.right_node = rightNode  # ✅ dot notation
                child -= 1
                
                i = 2
                while i < len(temp) and child > 0:
                    addRight = temp[i]
                    rightNode.right_node = addRight  # ✅ dot notation
                    rightNode = rightNode.right_node
                    child -= 1
                    i += 1
            
            toPush = Tree.build_node(val, type)
            toPush.left_node = tempLeft  # ✅ dot notation
            self.st.append(toPush)
        #print(f"buildTree: Stack size after building: {len(self.st)}")

    def get_token(self):
        row = self.index
        Tree.read = self.tree.read_line  # Assuming this is initialized properly elsewhere
        
        # Debug: show current reading index and character
        current_char = Tree.read[row] if row < self.size_of_file else 'EOF'
        #print(f"get_token: Called with self.index={self.index}, current char: '{current_char}'")
        
        # End of file or null char check
        if row >= self.size_of_file or Tree.read[row] == '\0':
            t = Token("EOF", "EOF")
            #print("get_token: Reached EOF")
            return t
        
        # Main tokenization loop
        while row < self.size_of_file and row < 10000 and Tree.read[row] != '\0':
            c = Tree.read[row]
            #print(f"get_token: Processing char '{c}' at index {row}")
            
            # Integer token
            if self.is_digit(c):
                num = ""
                while row < self.size_of_file and self.is_digit(Tree.read[row]):
                    num += Tree.read[row]
                    row += 1
                self.index = row
                t = Token("INT", num)
                #print(f"get_token: Created INT token: value={t.get_value()}, type={t.get_type()}")
                with open("output_token_sequence.txt", "a") as outputFile:
                    outputFile.write(f"<INTEGER> {num}\n")
                return t
            
            # Identifier or keyword token
            elif self.is_alpha(c):
                id_str = ""
                while row < self.size_of_file and (self.is_alpha(Tree.read[row]) or self.is_digit(Tree.read[row]) or Tree.read[row] == '_'):
                    id_str += Tree.read[row]
                    row += 1
                
                if self.is_reserved_key(id_str):
                    self.index = row
                    t = Token("KEYWORD", id_str)
                    #print(f"get_token: Created KEYWORD token: value={t.get_value()}, type={t.get_type()}")
                    with open("output_token_sequence.txt", "a") as outputFile:
                        outputFile.write(f"<KEYWORD> {id_str}\n")
                    return t
                else:
                    self.index = row
                    t = Token("ID", id_str)
                    #print(f"get_token: Created ID token: value={t.get_value()}, type={t.get_type()}")
                    with open("output_token_sequence.txt", "a") as outputFile:
                        outputFile.write(f"<IDENTIFIER> {id_str}\n")
                    return t
            
            # Comment token (//)
            elif c == '/' and row + 1 < self.size_of_file and Tree.read[row + 1] == '/':
                comment = ""
                while row < self.size_of_file and Tree.read[row] != '\n':
                    comment += Tree.read[row]
                    row += 1
                if row < self.size_of_file and Tree.read[row] == '\n':
                    row += 1
                self.index = row
                t = Token("COMMENT", comment)
                #print(f"get_token: Created COMMENT token: value={t.get_value()}, type={t.get_type()}")
                with open("output_token_sequence.txt", "a") as outputFile:
                    outputFile.write(f"<COMMENT> {comment}\n")
                return t
            
            # Operator token
            elif self.is_operator(c):
                op = ""
                while row < self.size_of_file and self.is_operator(Tree.read[row]):
                    op += Tree.read[row]
                    row += 1
                self.index = row
                t = Token("OPERATOR", op)
                #print(f"get_token: Created OPERATOR token: value={t.get_value()}, type={t.get_type()}")
                with open("output_token_sequence.txt", "a") as outputFile:
                    outputFile.write(f"<OPERATOR> {op}\n")
                return t
            
            # String literal token (single quotes)
            elif c == '\'':
                string_lit = "'"
                row += 1
                while row < self.size_of_file:
                    ch = Tree.read[row]
                    string_lit += ch
                    if ch == '\'':
                        row += 1
                        break
                    elif ch == '\\':
                        row += 1
                        if row < self.size_of_file:
                            string_lit += Tree.read[row]
                            row += 1
                    else:
                        row += 1
                self.index = row
                t = Token("STR", string_lit)
                #print(f"get_token: Created STR token: value={t.get_value()}, type={t.get_type()}")
                with open("output_token_sequence.txt", "a") as outputFile:
                    outputFile.write(f"<STRING> {string_lit}\n")
                return t
            
            # Punctuation token
            elif c in (')', '(', ';', ','):
                self.index = row + 1
                t = Token("PUNCTUATION", c)
                #print(f"get_token: Created PUNCTUATION token: value={t.get_value()}, type={t.get_type()}")
                with open("output_token_sequence.txt", "a") as outputFile:
                    outputFile.write(f"<PUNCTUATION> {c}\n")
                return t
            
            # Whitespace token (skip and do not output)
            elif c.isspace():
                while row < self.size_of_file and Tree.read[row].isspace():
                    row += 1
                self.index = row
                #print(f"get_token: Skipped whitespace, updated self.index to {self.index}")
                # Return a special token or continue loop to get next token
                continue
            
            # Unknown token (single char)
            else:
                self.index = row + 1
                t = Token("UNKNOWN", c)
                #print(f"get_token: Created UNKNOWN token: value={t.get_value()}, type={t.get_type()}")
                return t
        
        #print(f"get_token: Reached end of input at index {row}")
        return Token("EOF", "EOF")

    
    def parse(self):
        print("Starting parsing process...")

        self.next_token = self.get_token()  # Initialize the first token
        #print(f"Initial token: {self.next_token.get_value()} ({self.next_token.get_type()})")

        while self.next_token.get_type() == "DELETE":
            self.next_token = self.get_token()
            #print(f"Skipping DELETE token: {self.next_token.get_value()} ({self.next_token.get_type()})")

        self.procedure_E()

        while self.next_token.get_type() == "DELETE":
            self.next_token = self.get_token()
            #print(f"Final token after DELETE checks: {self.next_token.get_value()} ({self.next_token.get_type()})")

        if self.index >= self.size_of_file - 1:
            #print("Reached end of file, processing syntax tree...")

            t = self.st[-1] if self.st else None
            #print(f"Top of stack: {t.get_value() if t else 'None'}")

            if self.ast_flag == 1 and t:
                t.display_syntax_tree(0)

            if t:
                self.MST(t)
                print("Made standard tree (MST) from syntax tree.")

                if self.ast_flag == 2:
                    t.display_syntax_tree(0)

                controlStructureArray = [[None] * 200 for _ in range(200)]
                self.build_control_structures(t, controlStructureArray)

                size = 0
                while size < 200 and controlStructureArray[size][0] is not None:
                    size += 1

                controlNodeArray = []
                for row in range(size):
                    temp = [node for node in controlStructureArray[row] if node is not None]
                    controlNodeArray.append(temp)

                self.cse_machine(controlNodeArray)

        self.parsing_complete = True



    def cse_parse(self):
        print("Starting CSE parsing process...")
        self.cse_flag = 1
        self.next_token = self.get_token()

        while self.next_token.get_type() == "DELETE":
            self.next_token = self.get_token()

        self.procedure_E()

        while self.next_token.get_type() == "DELETE":
            self.next_token = self.get_token()

        if self.index >= self.size_of_file - 1:
            if not self.st:
                #print("Error: Parsing completed but syntax tree stack is empty. Check input file or parsing logic.")
                self.parsing_complete = True
                return

            t = self.st[-1]

            if self.ast_flag == 1:
                t.display_syntax_tree(0)

            self.MST(t)

            if self.ast_flag == 2:
                t.display_syntax_tree(0)

            controlStructureArray = [[None] * 200 for _ in range(200)]
            self.build_control_structures(t, controlStructureArray)

            size = 0
            while size < 200 and controlStructureArray[size][0] is not None:
                size += 1

            controlNodeArray = []
            for row in range(size):
                temp = [node for node in controlStructureArray[row] if node is not None]
                controlNodeArray.append(temp)

            self.cse_machine(controlNodeArray)

        self.parsing_complete = True

    def isParsingComplete(self):
        return self.parsing_complete


    def MST(self,t):
        self.makeStandardTree(t)


    def makeStandardTree(self, t):
        if t is None:
            return None
        
        # Debug: Trace the node being processed
        # print(f"makeStandardTree: Processing node with value={t.get_value()}, type={t.get_type()}")

        # Recursively standardize left and right subtrees
        self.makeStandardTree(t.left_node)
        self.makeStandardTree(t.right_node)

        val = t.get_value()
        # print(f"makeStandardTree: Transforming node: value={val}")

        if val == "let":
            if t.left_node and t.left_node.get_value() == "=":
                # Transform let X = E in P to gamma(lambda(X, P), E)
                t.set_value("gamma")
                t.set_type("KEYWORD")
                P = t.left_node.right_node  # Entire P subtree
                X = t.left_node.left_node   # Entire X subtree
                E = t.left_node.left_node.right_node  # Entire E subtree
                t.left_node = Tree.build_node("lambda", "KEYWORD")
                t.left_node.right_node = P
                t.left_node.left_node = X
                t.left_node.left_node.right_node = E
                # print(f"makeStandardTree: Transformed 'let' to gamma-lambda structure")

        elif val == "and" and t.left_node and t.left_node.get_value() == "=":
            # Transform and (X1=E1, X2=E2, ...) to =(, (X1, X2, ...), tau(E1, E2, ...))
            equal = t.left_node
            t.set_value("=")
            t.set_type("KEYWORD")
            t.left_node = Tree.build_node(",", "PUNCTUATION")
            comma = t.left_node
            comma.left_node = equal.left_node  # First variable
            t.left_node.right_node = Tree.build_node("tau", "KEYWORD")
            tau = t.left_node.right_node
            tau.left_node = equal.left_node.right_node  # First expression
            tau_current = tau.left_node
            comma_current = comma.left_node
            equal = equal.right_node

            while equal is not None:
                comma_current.right_node = equal.left_node
                comma_current = comma_current.right_node
                tau_current.right_node = equal.left_node.right_node
                tau_current = tau_current.right_node
                equal = equal.right_node
            # print(f"makeStandardTree: Transformed 'and' to =-comma-tau structure")

        elif val == "where":
            if t.left_node and t.left_node.right_node and t.left_node.right_node.get_value() == "=":
                # Transform P where X = E to gamma(lambda(X, P), E)
                t.set_value("gamma")
                t.set_type("KEYWORD")
                P = t.left_node  # Entire P subtree
                X = t.left_node.right_node.left_node
                E = t.left_node.right_node.left_node.right_node
                t.left_node = Tree.build_node("lambda", "KEYWORD")
                t.left_node.right_node = P
                t.left_node.left_node = X
                t.left_node.left_node.right_node = E
                # print(f"makeStandardTree: Transformed 'where' to gamma-lambda structure")

        elif val == "within":
            if t.left_node and t.left_node.get_value() == "=" and t.left_node.right_node and t.left_node.right_node.get_value() == "=":
                # Transform X1 = E1 within X2 = E2 to =(X2, gamma(lambda(X1, E2), E1))
                X1 = t.left_node.left_node
                E1 = t.left_node.left_node.right_node
                X2 = t.left_node.right_node.left_node
                E2 = t.left_node.right_node.left_node.right_node
                t.set_value("=")
                t.set_type("KEYWORD")
                t.left_node = X2
                t.left_node.right_node = Tree.build_node("gamma", "KEYWORD")
                temp = t.left_node.right_node
                temp.left_node = Tree.build_node("lambda", "KEYWORD")
                temp.left_node.right_node = E2
                temp.left_node.left_node = X1
                temp.left_node.left_node.right_node = E1
                # print(f"makeStandardTree: Transformed 'within' to =-gamma-lambda structure")

        elif val == "rec" and t.left_node and t.left_node.get_value() == "=":
            # Transform rec X = E to =(X, gamma(YSTAR, lambda(X, E)))
            X = t.left_node.left_node
            E = t.left_node.left_node.right_node
            t.set_value("=")
            t.set_type("KEYWORD")
            t.left_node = X
            t.left_node.right_node = Tree.build_node("gamma", "KEYWORD")
            gamma = t.left_node.right_node
            gamma.left_node = Tree.build_node("YSTAR", "KEYWORD")
            ystar = gamma.left_node
            ystar.right_node = Tree.build_node("lambda", "KEYWORD")
            ystar.right_node.left_node = X
            ystar.right_node.left_node.right_node = E
            # print(f"makeStandardTree: Transformed 'rec' to =-gamma-YSTAR-lambda structure")

        elif val == "function_form":
            # Transform function_form P V1 V2 ... Vn E to =(P, lambda(V1, lambda(V2, ... lambda(Vn, E))))
            P = t.left_node
            V = t.left_node.right_node
            t.set_value("=")
            t.set_type("KEYWORD")
            t.left_node = P
            temp = t
            while V and V.right_node and V.right_node.right_node is not None:
                temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
                temp = temp.left_node.right_node
                temp.left_node = V
                V = V.right_node
            temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
            temp = temp.left_node.right_node
            temp.left_node = V
            temp.left_node.right_node = V.right_node
            # print(f"makeStandardTree: Transformed 'function_form' to =-lambda chain")

        elif val == "lambda":
            if t.left_node is not None:
                # Transform lambda V1 V2 ... Vn E to lambda(V1, lambda(V2, ... lambda(Vn, E)))
                V = t.left_node
                temp = t
                while V.right_node and V.right_node.right_node is not None:
                    temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
                    temp = temp.left_node.right_node
                    temp.left_node = V
                    V = V.right_node
                temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
                temp = temp.left_node.right_node
                temp.left_node = V
                temp.left_node.right_node = V.right_node
                # print(f"makeStandardTree: Transformed 'lambda' with multiple parameters")

        elif val == "@":
            if t.left_node and t.left_node.right_node and t.left_node.right_node.right_node:
                # Transform E1 @ N E2 to gamma(gamma(N, E1), E2)
                E1 = t.left_node
                N = t.left_node.right_node
                E2 = t.left_node.right_node.right_node
                t.set_value("gamma")
                t.set_type("KEYWORD")
                t.left_node = Tree.build_node("gamma", "KEYWORD")
                t.left_node.right_node = E2
                t.left_node.left_node = N
                t.left_node.left_node.right_node = E1
                # print(f"makeStandardTree: Transformed '@' to gamma-gamma structure")

        return None
    
    def build_control_structures(self, x, controlNodeArray):
        """Recursively builds control structures from AST nodes"""
        if x is None:
            return

        row = self.control_structures_row
        column = self.control_structures_column

        # Handle column overflow
        if column >= 200:
            row += 1
            column = 0
            self.control_structures_row = row

        if row >= 200:  # Prevent row overflow
            return

        # Create node and add to control structure
        controlNodeArray[row][column] = Tree.build_node(x.get_value(), x.get_type())
        column += 1

        # Update position
        self.control_structures_column = column

        # Process children recursively
        if hasattr(x, 'left_node') and x.left_node:
            self.build_control_structures(x.left_node, controlNodeArray)
        if hasattr(x, 'right_node') and x.right_node:
            self.build_control_structures(x.right_node, controlNodeArray)

    def build_node(self, x, value, node_type):
        """Builds control structures for special node types"""
        print(f"build_node: Processing {x.get_value()} with value={value}, type={node_type}")

        # Handle lambda nodes
        if x.get_value() == "lambda":
            row = self.control_structures_row
            column = self.control_structures_column
            index = self.control_structures_index

            # Count existing rows to determine delta number
            counter = 0
            temp_row = 0
            while temp_row < len(self.controlNodeArray) and self.controlNodeArray[temp_row][0] is not None:
                temp_row += 1
                counter += 1

            # Create delta number node
            delta_num = Tree.build_node(str(counter), "deltaNumber")
            self.controlNodeArray[row][column] = delta_num
            column += 1

            # Add bound variable
            if hasattr(x, 'left_node') and x.left_node:
                self.controlNodeArray[row][column] = Tree.build_node(x.left_node.get_value(), x.left_node.get_type())
                column += 1

            # Add lambda node
            self.controlNodeArray[row][column] = Tree.build_node("lambda", "lambda")
            column += 1

            # Save current position
            saved_row = row
            saved_column = column

            # Move to next available row for lambda body
            while row < len(self.controlNodeArray) and self.controlNodeArray[row][0] is not None:
                row += 1

            # Update position for recursion
            self.control_structures_row = row
            self.control_structures_column = 0
            self.control_structures_index = index + 1

            # Process lambda body
            if hasattr(x, 'left_node') and x.left_node and hasattr(x.left_node, 'right_node'):
                self.build_control_structures(x.left_node.right_node, self.controlNodeArray)

            # Restore position
            self.control_structures_row = saved_row
            self.control_structures_column = saved_column
            return

        # Handle conditional nodes "->"
        elif x.get_value() == "->":
            row = self.control_structures_row
            column = self.control_structures_column
            index = self.control_structures_index

            # Create delta numbers for then and else branches
            then_delta = Tree.build_node(str(index), "deltaNumber")
            else_delta = Tree.build_node(str(index + 1), "deltaNumber")
            beta_node = Tree.build_node("beta", "beta")

            self.controlNodeArray[row][column] = then_delta
            column += 1
            self.controlNodeArray[row][column] = else_delta
            column += 1
            self.controlNodeArray[row][column] = beta_node
            column += 1

            # Save current state
            saved_row = row
            saved_column = column

            # Process condition first
            if hasattr(x, 'left_node') and x.left_node:
                self.build_control_structures(x.left_node, self.controlNodeArray)

            # Move to next row for then branch
            while row < len(self.controlNodeArray) and self.controlNodeArray[row][0] is not None:
                row += 1
            
            then_row = row
            self.control_structures_row = row
            self.control_structures_column = 0
            self.control_structures_index = index + 2

            # Process then branch
            if (hasattr(x, 'left_node') and x.left_node and 
                hasattr(x.left_node, 'right_node') and x.left_node.right_node):
                self.build_control_structures(x.left_node.right_node, self.controlNodeArray)

            # Move to next row for else branch
            while row < len(self.controlNodeArray) and self.controlNodeArray[row][0] is not None:
                row += 1

            else_row = row
            self.control_structures_row = row
            self.control_structures_column = 0

            # Process else branch
            if (hasattr(x, 'left_node') and x.left_node and 
                hasattr(x.left_node, 'right_node') and x.left_node.right_node and
                hasattr(x.left_node.right_node, 'right_node') and x.left_node.right_node.right_node):
                self.build_control_structures(x.left_node.right_node.right_node, self.controlNodeArray)

            # Update delta numbers with actual row indices
            self.controlNodeArray[saved_row][saved_column - 3].set_value(str(then_row))
            self.controlNodeArray[saved_row][saved_column - 2].set_value(str(else_row))

            # Restore position
            self.control_structures_row = saved_row
            self.control_structures_column = saved_column
            self.control_structures_beta += 2
            return

        # Handle tau nodes (tuples)
        elif x.get_value() == "tau":
            row = self.control_structures_row
            column = self.control_structures_column

            # Count children
            child_count = 0
            current = x.left_node if hasattr(x, 'left_node') else None
            while current:
                child_count += 1
                current = current.right_node if hasattr(current, 'right_node') else None

            # Add count and tau node
            count_node = Tree.build_node(str(child_count), "CHILDCOUNT")
            tau_node = Tree.build_node("tau", "tau")
            
            self.controlNodeArray[row][column] = count_node
            column += 1
            self.controlNodeArray[row][column] = tau_node
            column += 1

            # Update position
            self.control_structures_row = row
            self.control_structures_column = column

            # Process all children
            current = x.left_node if hasattr(x, 'left_node') else None
            while current:
                self.build_control_structures(current, self.controlNodeArray)
                current = current.right_node if hasattr(current, 'right_node') else None

            return

        # Handle other nodes
        else:
            row = self.control_structures_row
            column = self.control_structures_column

            if column >= 200:
                row += 1
                column = 0
                self.control_structures_row = row

            if row < 200:
                self.controlNodeArray[row][column] = Tree.build_node(x.get_value(), x.get_type())
                column += 1
                self.control_structures_column = column

                # Process children
                if hasattr(x, 'left_node') and x.left_node:
                    self.build_control_structures(x.left_node, self.controlNodeArray)
                if hasattr(x, 'right_node') and x.right_node:
                    self.build_control_structures(x.right_node, self.controlNodeArray)

    def cse_machine(self, control_struct):
        """Main CSE machine execution engine"""
        print("Starting CSE machine execution...")
        
        if not control_struct or len(control_struct) == 0:
            print("Error: Empty control structure")
            return

        # Initialize stacks and environment
        control = deque()
        machine_stack = deque()
        environment_stack = deque()

        curr_env_index = 0
        curr_env = Environment(name="env0")
        
        curr_env_index += 1
        machine_stack.append(Tree.build_node(curr_env.name, "ENV"))
        control.append(Tree.build_node(curr_env.name, "ENV"))
        environment_stack.append(curr_env)

        # Load initial control structure
        if control_struct and len(control_struct) > 0:
            initial_control = control_struct[0]
            for i in range(len(initial_control) - 1, -1, -1):  # Reverse order
                if initial_control[i] is not None:
                    control.append(initial_control[i])

        print(f"Initial control stack: {[node.get_value() if node else 'None' for node in control]}")
        print(f"Initial machine stack: {[node.get_value() if node else 'None' for node in machine_stack]}")

        # Main execution loop
        while control:
            try:
                next_token = control.pop()
                if not next_token:
                    continue
                    
                print(f"\nProcessing token: {next_token.get_value()} ({next_token.get_type()})")
                print(f"Control stack: {[node.get_value() if node else 'None' for node in list(control)[-5:]]}")  # Show last 5
                print(f"Machine stack: {[node.get_value() if node else 'None' for node in list(machine_stack)[-5:]]}")  # Show last 5

                # Handle nil tokens
                if next_token.get_value() == "nil":
                    next_token.set_type("tau")

                # Handle operands and built-in functions
                if (next_token.get_type() in ["INT", "STR", "BOOL", "NIL", "DUMMY"] or 
                    next_token.get_value() in ["lambda", "YSTAR", "Print", "Isinteger", "Istruthvalue", 
                                            "Isstring", "Istuple", "Isfunction", "Isdummy", "Stem", 
                                            "Stern", "Conc", "Order", "nil"]):
                    
                    if next_token.get_value() == "lambda":
                        # Lambda requires environment, variable, and delta index
                        if len(control) < 2:
                            print(f"Warning: Insufficient tokens for lambda")
                            machine_stack.append(next_token)
                            continue
                        
                        bound_variable = control.pop()
                        delta_index = control.pop()
                        env_node = Tree.build_node(curr_env.name, "ENV")
                        
                        # Push in correct order: delta, variable, env, lambda
                        machine_stack.append(delta_index)
                        machine_stack.append(bound_variable)
                        machine_stack.append(env_node)
                        machine_stack.append(next_token)
                    else:
                        machine_stack.append(next_token)

                # Handle gamma instruction
                elif next_token.get_value() == "gamma":
                    if not machine_stack:
                        print("Error: Empty machine stack for gamma")
                        continue
                        
                    rator = machine_stack[-1]  # Don't pop yet
                    
                    if rator.get_value() == "lambda":
                        # Lambda application
                        machine_stack.pop()  # Remove lambda
                        if len(machine_stack) < 3:
                            print("Error: Insufficient elements for lambda application")
                            continue
                            
                        env_node = machine_stack.pop()
                        bound_var = machine_stack.pop()
                        delta_index = machine_stack.pop()
                        
                        if not machine_stack:
                            print("Error: No argument for lambda application")
                            continue
                        rand = machine_stack.pop()  # The argument

                        # Create new environment
                        new_env = Environment(name=f"env{curr_env_index}")
                        
                        # Find parent environment
                        for env in reversed(environment_stack):
                            if env.name == env_node.get_value():
                                new_env.prev = env
                                break

                        # Bind the parameter
                        if bound_var.get_value() == "," and rand.get_value() == "tau":
                            # Handle tuple unpacking for multiple parameters
                            params = self.extract_comma_list(bound_var)
                            values = self.extract_tau_elements(rand)
                            
                            for param, value in zip(params, values):
                                param_key = Tree.build_node(param.get_value(), param.get_type())
                                new_env.bound_variable[param_key] = [value]
                        else:
                            # Simple parameter binding
                            param_key = Tree.build_node(bound_var.get_value(), bound_var.get_type())
                            new_env.bound_variable[param_key] = [rand]

                        # Update environment
                        curr_env = new_env
                        environment_stack.append(curr_env)
                        curr_env_index += 1

                        # Push new environment marker
                        machine_stack.append(Tree.build_node(curr_env.name, "ENV"))
                        control.append(Tree.build_node(curr_env.name, "ENV"))

                        # Load the lambda body
                        try:
                            delta_idx = int(delta_index.get_value())
                            if 0 <= delta_idx < len(control_struct):
                                lambda_body = control_struct[delta_idx]
                                for i in range(len(lambda_body) - 1, -1, -1):
                                    if lambda_body[i] is not None:
                                        control.append(lambda_body[i])
                        except (ValueError, IndexError) as e:
                            print(f"Error loading lambda body: {e}")

                    elif rator.get_value() == "tau":
                        # Tuple selection
                        tau_node = machine_stack.pop()
                        if not machine_stack:
                            print("Error: No index for tuple selection")
                            continue
                        index_node = machine_stack.pop()
                        
                        try:
                            index = int(index_node.get_value())
                            elements = self.extract_tau_elements(tau_node)
                            if 1 <= index <= len(elements):
                                machine_stack.append(elements[index - 1])
                            else:
                                print(f"Error: Tuple index {index} out of range")
                        except (ValueError, IndexError) as e:
                            print(f"Error in tuple selection: {e}")

                    elif rator.get_value() in ["Print", "Order", "Isinteger", "Istruthvalue", 
                                             "Isstring", "Istuple", "Isfunction", "Isdummy"]:
                        # Built-in function application
                        func = machine_stack.pop()
                        if machine_stack:
                            arg = machine_stack.pop()
                            result = self.apply_builtin_function(func.get_value(), arg)
                            if result:
                                machine_stack.append(result)

                # Handle environment restoration
                elif next_token.get_value().startswith("env"):
                    # Save the current result
                    result_stack = []
                    while machine_stack and not machine_stack[-1].get_value().startswith("env"):
                        result_stack.append(machine_stack.pop())
                    
                    # Remove environment marker
                    if machine_stack and machine_stack[-1].get_value() == next_token.get_value():
                        machine_stack.pop()
                        if environment_stack:
                            environment_stack.pop()
                            curr_env = environment_stack[-1] if environment_stack else Environment()
                    
                    # Restore results
                    for item in reversed(result_stack):
                        machine_stack.append(item)

                # Handle variable lookup
                elif (next_token.get_type() == "ID" and 
                    next_token.get_value() not in ["Print", "Order", "Isinteger", "Istruthvalue", 
                                                "Isstring", "Istuple", "Isfunction", "Isdummy", 
                                                "Stem", "Stern", "Conc"]):
                    
                    found = False
                    temp_env = curr_env
                    
                    while temp_env and not found:
                        print(f"Searching in env '{temp_env.name}': {[var.get_value() for var in temp_env.bound_variable.keys()]}")
                        
                        for var_key, values in temp_env.bound_variable.items():
                            if var_key.get_value() == next_token.get_value():
                                print(f"Found variable '{next_token.get_value()}' in env '{temp_env.name}'")
                                for value in values:
                                    machine_stack.append(value)
                                found = True
                                break
                        
                        temp_env = temp_env.prev
                    
                    if not found:
                        print(f"Error: Variable '{next_token.get_value()}' not found")

                # Handle operators
                elif self.is_operator(next_token.get_value()):
                    result = self.apply_operator(next_token.get_value(), machine_stack)
                    if result:
                        machine_stack.append(result)

                # Handle conditional (beta)
                elif next_token.get_value() == "beta":
                    if len(machine_stack) < 1 or len(control) < 2:
                        print("Error: Insufficient data for conditional")
                        continue
                        
                    condition = machine_stack.pop()
                    else_delta = control.pop()
                    then_delta = control.pop()
                    
                    try:
                        # Choose branch based on condition
                        if condition.get_value() in ["true", "True"] or condition.get_value() == True:
                            chosen_delta = then_delta
                        else:
                            chosen_delta = else_delta
                            
                        delta_idx = int(chosen_delta.get_value())
                        if 0 <= delta_idx < len(control_struct):
                            branch_control = control_struct[delta_idx]
                            for i in range(len(branch_control) - 1, -1, -1):
                                if branch_control[i] is not None:
                                    control.append(branch_control[i])
                    except (ValueError, IndexError) as e:
                        print(f"Error in conditional: {e}")

                # Handle tuple creation (tau)
                elif next_token.get_value() == "tau":
                    if not control:
                        print("Error: No count for tau")
                        continue
                        
                    count_node = control.pop()
                    try:
                        count = int(count_node.get_value())
                        if len(machine_stack) < count:
                            print(f"Error: Need {count} elements, have {len(machine_stack)}")
                            continue
                            
                        # Create tau node
                        tau_node = Tree.build_node("tau", "tau")
                        if count > 0:
                            # Pop elements and chain them
                            elements = []
                            for _ in range(count):
                                elements.append(machine_stack.pop())
                            elements.reverse()  # Restore original order
                            
                            # Chain elements using right_node
                            tau_node.left_node = elements[0]
                            current = tau_node.left_node
                            for i in range(1, len(elements)):
                                current.right_node = elements[i]
                                current = current.right_node
                                
                        machine_stack.append(tau_node)
                    except ValueError as e:
                        print(f"Error in tau creation: {e}")

                else:
                    print(f"Unhandled token: {next_token.get_value()} ({next_token.get_type()})")

            except Exception as e:
                print(f"Error during execution: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\nCSE machine execution completed")
        
        # Output final result
        if machine_stack:
            # Find the actual result (skip environment markers)
            result = None
            for item in reversed(machine_stack):
                if not item.get_value().startswith("env"):
                    result = item
                    break
            
            if result:
                print("Output of the above program is:")
                if result.get_type() == "STR":
                    # Remove quotes from string output
                    output = result.get_value()
                    if output.startswith("'") and output.endswith("'"):
                        output = output[1:-1]
                    print(output)
                else:
                    print(result.get_value())
            else:
                print("No result found")
        else:
            print("Warning: Empty machine stack")


    def arrangeTuple(self,tauNode, res):
        if tauNode is None or tauNode.get_value() == "lamdaTuple":
            return
        if tauNode.get_value() not in ["tau", "nil"]:
            res.append(tauNode)
        self.arrangeTuple(tauNode.left_node, res)
        self.arrangeTuple(tauNode.right_node, res)

    def addSpaces(temp):
            result = temp
            row = 1
            while row < len(result):
                if result[row - 1] == '\\' and result[row] == 'n':
                    result = result[:row - 1] + '\\n' + result[row + 1:]
                    row += 1
                elif result[row - 1] == '\\' and result[row] == 't':
                    result = result[:row - 1] + '\\t' + result[row + 1:]
                    row += 1
                row += 1
            result = result.replace('\\', '').replace("'", '')
            return result

    def procedure_E(self):
        #print(f"procedure_E: Current token: value={self.next_token.get_value()}, type={self.next_token.get_type()}")
        if self.next_token.get_value() == "let":
            #print("procedure_E: Processing 'let'")
            self.read("let", "KEYWORD")
            #print("procedure_E: Calling procedure_D")
            self.procedure_D()
            #print("procedure_E: Reading 'in'")
            self.read("in", "KEYWORD")
            #print("procedure_E: Calling procedure_E recursively")
            self.procedure_E()
            #print("procedure_E: Building 'let' tree")
            self.buildTree("let", "KEYWORD", 2)
        elif self.next_token.get_value() == "fn":
            #print("procedure_E: Processing 'fn'")
            n = 0
            self.read("fn", "KEYWORD")
            #print("procedure_E: Reading 'fn' variables")
            while self.next_token.get_type() == "ID" or self.next_token.get_value() == "(":
                self.procedure_Vb()
                n += 1
            self.read(".", "OPERATOR")
            #print("procedure_E: Calling procedure_E after 'fn'")
            self.procedure_E()
            #print("procedure_E: Building 'lambda' tree with {n+1} children")
            self.buildTree("lambda", "KEYWORD", n + 1)
        else:
            #print("procedure_E: Falling back to procedure_Ew")
            self.procedure_Ew()
    def procedure_Ew(self):
        self.procedure_T()
        if self.next_token.get_value() == "where":
            self.read("where", "KEYWORD")
            self.procedure_Dr()
            self.buildTree("where", "KEYWORD", 2)

    def procedure_T(self):
        self.procedure_Ta()
        n = 1
        while self.next_token.get_value() == ",":
            n += 1
            self.read(",", "PUNCTUATION")
            self.procedure_Ta()
        if n > 1:
            self.buildTree("tau", "KEYWORD", n)

    def procedure_Ta(self):
        self.procedure_Tc()
        while self.next_token.get_value() == "aug":
            self.read("aug", "KEYWORD")
            self.procedure_Tc()
            self.buildTree("aug", "KEYWORD", 2)

    def procedure_Tc(self):
        self.procedure_B()
        if self.next_token.get_value() == "->":
            self.read("->", "OPERATOR")
            self.procedure_Tc()
            self.read("|", "OPERATOR")
            self.procedure_Tc()
            self.buildTree("->", "KEYWORD", 3)

    def procedure_B(self):
        self.procedure_Bt()
        while self.next_token.get_value() == "or":
            self.read("or", "KEYWORD")
            self.procedure_Bt()
            self.buildTree("or", "KEYWORD", 2)

    def procedure_Bt(self):
        self.procedure_Bs()
        while self.next_token.get_value() == "&":
            self.read("&", "OPERATOR")
            self.procedure_Bs()
            self.buildTree("&", "KEYWORD", 2)

    def procedure_Bs(self):
        if self.next_token.get_value() == "not":
            self.read("not", "KEYWORD")
            self.procedure_Bp()
            self.buildTree("not", "KEYWORD", 1)
        else:
            self.procedure_Bp()

    def procedure_Bp(self):
        self.procedure_A()
        temp = self.next_token.get_value()
        temp2 = self.next_token.get_type()
        if temp in ["gr", ">"]:
            self.read(temp, temp2)
            self.procedure_A()
            self.buildTree("gr", "KEYWORD", 2)
        elif temp in ["ge", ">="]:
            self.read(temp, temp2)
            self.procedure_A()
            self.buildTree("ge", "KEYWORD", 2)
        elif temp in ["ls", "<"]:
            self.read(temp, temp2)
            self.procedure_A()
            self.buildTree("ls", "KEYWORD", 2)
        elif temp in ["le", "<="]:
            self.read(temp, temp2)
            self.procedure_A()
            self.buildTree("le", "KEYWORD", 2)
        elif temp == "eq":
            self.read(temp, temp2)
            self.procedure_A()
            self.buildTree("eq", "KEYWORD", 2)
        elif temp == "ne":
            self.read(temp, temp2)
            self.procedure_A()
            self.buildTree("ne", "KEYWORD", 2)

    def procedure_A(self):
        if self.next_token.get_value() == "+":
            self.read("+", "OPERATOR")
            self.procedure_At()
        elif self.next_token.get_value() == "-":
            self.read("-", "OPERATOR")
            self.procedure_At()
            self.buildTree("neg", "KEYWORD", 1)
        else:
            self.procedure_At()
        while self.next_token.get_value() in ["+", "-"]:
            temp = self.next_token.get_value()
            self.read(temp, "OPERATOR")
            self.procedure_At()
            self.buildTree(temp, "OPERATOR", 2)

    def procedure_At(self):
        self.procedure_Af()
        while self.next_token.get_value() in ["*", "/"]:
            temp = self.next_token.get_value()
            self.read(temp, "OPERATOR")
            self.procedure_Af()
            self.buildTree(temp, "OPERATOR", 2)

    def procedure_Af(self):
        self.procedure_Ap()
        if self.next_token.get_value() == "**":
            self.read("**", "OPERATOR")
            self.procedure_Af()
            self.buildTree("**", "KEYWORD", 2)

    def procedure_Ap(self):
        self.procedure_R()
        while self.next_token.get_value() == "@":
            self.read("@", "OPERATOR")
            if self.next_token.get_type() != "ID":
                print("Exception: UNEXPECTED_TOKEN")
            else:
                self.read(self.next_token.get_value(), "ID")
                self.procedure_R()
                self.buildTree("@", "KEYWORD", 3)

    def procedure_R(self):
        self.procedure_Rn()
        while self.next_token.get_type() in ["ID", "INT", "STR"] or self.next_token.get_value() in ["true", "false", "nil", "(", "dummy"]:
            self.procedure_Rn()
            self.buildTree("gamma", "KEYWORD", 2)

    def procedure_Rn(self):
        if self.next_token.get_type() in ["ID", "INT", "STR"]:
            self.read(self.next_token.get_value(), self.next_token.get_type())
        elif self.next_token.get_value() == "true":
            self.read("true", "KEYWORD")
            self.buildTree("true", "BOOL", 0)
        elif self.next_token.get_value() == "false":
            self.read("false", "KEYWORD")
            self.buildTree("false", "BOOL", 0)
        elif self.next_token.get_value() == "nil":
            self.read("nil", "KEYWORD")
            self.buildTree("nil", "NIL", 0)
        elif self.next_token.get_value() == "(":
            self.read("(", "PUNCTUATION")
            self.procedure_E()
            self.read(")", "PUNCTUATION")
        elif self.next_token.get_value() == "dummy":
            self.read("dummy", "KEYWORD")
            self.buildTree("dummy", "DUMMY", 0)

    def procedure_D(self):
        self.procedure_Da()
        if self.next_token.get_value() == "within":
            self.read("within", "KEYWORD")
            self.procedure_Da()
            self.buildTree("within", "KEYWORD", 2)

    def procedure_Da(self):
        self.procedure_Dr()
        n = 1
        while self.next_token.get_value() == "and":
            n += 1
            self.read("and", "KEYWORD")
            self.procedure_Dr()
        if n > 1:
            self.buildTree("and", "KEYWORD", n)

    def procedure_Dr(self):
        if self.next_token.get_value() == "rec":
            self.read("rec", "KEYWORD")
            self.procedure_Db()
            self.buildTree("rec", "KEYWORD", 1)
        else:
            self.procedure_Db()

    def procedure_Db(self):
        if self.next_token.get_value() == "(":
            self.read("(", "PUNCTUATION")
            self.procedure_D()
            self.read(")", "PUNCTUATION")
        elif self.next_token.get_type() == "ID":
            self.read(self.next_token.get_value(), "ID")
            n = 1
            if self.next_token.get_value() in ["=", ","]:
                while self.next_token.get_value() == ",":
                    self.read(",", "PUNCTUATION")
                    self.read(self.next_token.get_value(), "ID")
                    n += 1
                if n > 1:
                    self.buildTree(",", "KEYWORD", n)
                self.read("=", "OPERATOR")
                self.procedure_E()
                self.buildTree("=", "KEYWORD", 2)
            else:
                while self.next_token.get_type() == "ID" or self.next_token.get_value() == "(":
                    self.procedure_Vb()
                    n += 1
                self.read("=", "OPERATOR")
                self.procedure_E()
                self.buildTree("function_form", "KEYWORD", n + 1)

    def procedure_Vb(self):
        if self.next_token.get_type() == "ID":
            self.read(self.next_token.get_value(), "ID")
        elif self.next_token.get_value() == "(":
            self.read("(", "PUNCTUATION")
            if self.next_token.get_value() == ")":
                self.read(")", "PUNCTUATION")
                self.buildTree("()", "KEYWORD", 0)
            else:
                self.procedure_Vl()
                self.read(")", "PUNCTUATION")

    def procedure_Vl(self):
        n = 1
        self.read(self.next_token.get_value(), "ID")
        while self.next_token.get_value() == ",":
            self.read(",", "PUNCTUATION")
            self.read(self.next_token.get_value(), "ID")
            n += 1
        if n > 1:
            self.buildTree(",", "KEYWORD", n)


