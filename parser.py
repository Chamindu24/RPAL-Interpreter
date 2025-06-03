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
                t.displaySyntaxTree(0)

            if t:
                self.MST(t)
                print("Made standard tree (MST) from syntax tree.")

                if self.ast_flag == 2:
                    t.displaySyntaxTree(0)

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
                t.displaySyntaxTree(0)

            self.MST(t)

            if self.ast_flag == 2:
                t.displaySyntaxTree(0)

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
        return self.parsingComplete


    def MST(self,t):
        self.makeStandardTree(t)


    def makeStandardTree(self, t):
        
        if t is None:
            return None
        #print(f"makeStandardTree: Processing node with value={t.get_value()} and type={t.get_type()}")

        self.makeStandardTree(t.left_node)
        self.makeStandardTree(t.right_node)

        val = t.get_value()

        if val == "let":
            if t.left_node.get_value() == "=":
                t.set_value("gamma")
                t.set_type("KEYWORD")
                P = Tree.build_node(t.left_node.right_node.get_value(), t.left_node.right_node.get_type())
                X = Tree.build_node(t.left_node.left_node.get_value(), t.left_node.left_node.get_type())
                E = Tree.build_node(t.left_node.left_node.right_node.get_value(), t.left_node.left_node.right_node.get_type())
                t.left_node = Tree.build_node("lambda", "KEYWORD")
                t.left_node.right_node = E
                lambda_node = t.left_node
                lambda_node.left_node = X
                lambda_node.left_node.right_node = P

        elif val == "and" and t.left_node.get_value() == "=":
            equal = t.left_node
            t.set_value("=")
            t.set_type("KEYWORD")
            t.left_node = Tree.build_node(",", "PUNCTUATION")
            comma = t.left_node
            comma.left_node = Tree.build_node(equal.left_node.get_value(), equal.left_node.get_type())
            t.left_node.right_node = Tree.build_node("tau", "KEYWORD")
            tau = t.left_node.right_node

            tau.left_node = Tree.build_node(equal.left_node.right_node.get_value(), equal.left_node.right_node.get_type())
            tau = tau.left_node
            comma = comma.left_node
            equal = equal.right_node

            while equal is not None:
                comma.right_node = Tree.build_node(equal.left_node.get_value(), equal.left_node.get_type())
                comma = comma.right_node
                tau.right_node = Tree.build_node(equal.left_node.right_node.get_value(), equal.left_node.right_node.get_type())
                tau = tau.right_node

                equal = equal.right_node

        elif val == "where":
            t.set_value("gamma")
            t.set_type("KEYWORD")
            if t.left_node.right_node.get_value() == "=":
                P = Tree.build_node(t.left_node.get_value(), t.left_node.get_type())
                X = Tree.build_node(t.left_node.right_node.left_node.get_value(), t.left_node.right_node.left_node.get_type())
                E = Tree.build_node(t.left_node.right_node.left_node.right_node.get_value(), t.left_node.right_node.left_node.right_node.get_type())
                t.left_node = Tree.build_node("lambda", "KEYWORD")
                t.left_node.right_node = E
                t.left_node.left_node = X
                t.left_node.left_node.right_node = P

        elif val == "within":
            if t.left_node.get_value() == "=" and t.left_node.right_node.get_value() == "=":
                X1 = Tree.build_node(t.left_node.left_node.get_value(), t.left_node.left_node.get_type())
                E1 = Tree.build_node(t.left_node.left_node.right_node.get_value(), t.left_node.left_node.right_node.get_type())
                X2 = Tree.build_node(t.left_node.right_node.left_node.get_value(), t.left_node.right_node.left_node.get_type())
                E2 = Tree.build_node(t.left_node.right_node.left_node.right_node.get_value(), t.left_node.right_node.left_node.right_node.get_type())
                t.set_value("=")
                t.set_type("KEYWORD")
                t.left_node = X2
                t.left_node.right_node = Tree.build_node("gamma", "KEYWORD")
                temp = t.left_node.right_node
                temp.left_node = Tree.build_node("lambda", "KEYWORD")
                temp.left_node.right_node = E1
                temp = temp.left_node
                temp.left_node = X1
                temp.left_node.right_node = E2

        elif val == "rec" and t.left_node.get_value() == "=":
            X = Tree.build_node(t.left_node.left_node.get_value(), t.left_node.left_node.get_type())
            E = Tree.build_node(t.left_node.left_node.right_node.get_value(), t.left_node.left_node.right_node.get_type())

            t.set_value("=")
            t.set_type("KEYWORD")
            t.left_node = X
            t.left_node.right_node = Tree.build_node("gamma", "KEYWORD")
            t.left_node.right_node.left_node = Tree.build_node("YSTAR", "KEYWORD")
            ystar = t.left_node.right_node.left_node

            ystar.right_node = Tree.build_node("lambda", "KEYWORD")

            ystar.right_node.left_node = Tree.build_node(X.get_value(), X.get_type())
            ystar.right_node.left_node.right_node = Tree.build_node(E.get_value(), E.get_type())

        elif val == "function_form":
            P = Tree.build_node(t.left_node.get_value(), t.left_node.get_type())
            V = t.left_node.right_node

            t.set_value("=")
            t.set_type("KEYWORD")
            t.left_node = P

            temp = t
            while V.right_node and V.right_node.right_node is not None:
                temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
                temp = temp.left_node.right_node
                temp.left_node = Tree.build_node(V.get_value(), V.get_type())
                V = V.right_node

            temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
            temp = temp.left_node.right_node

            temp.left_node = Tree.build_node(V.get_value(), V.get_type())
            temp.left_node.right_node = V.right_node

        elif val == "lambda":
            if t.left_node is not None:
                V = t.left_node
                temp = t
                if V.right_node is not None and V.right_node.right_node is not None:
                    while V.right_node.right_node is not None:
                        temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
                        temp = temp.left_node.right_node
                        temp.left_node = Tree.build_node(V.get_value(), V.get_type())
                        V = V.right_node

                    temp.left_node.right_node = Tree.build_node("lambda", "KEYWORD")
                    temp = temp.left_node.right_node
                    temp.left_node = Tree.build_node(V.get_value(), V.get_type())
                    temp.left_node.right_node = V.right_node

        elif val == "@":
            E1 = Tree.build_node(t.left_node.get_value(), t.left_node.get_type())
            N = Tree.build_node(t.left_node.right_node.get_value(), t.left_node.right_node.get_type())
            E2 = Tree.build_node(t.left_node.right_node.right_node.get_value(), t.left_node.right_node.right_node.get_type())
            t.set_value("gamma")
            t.set_type("KEYWORD")
            t.left_node = Tree.build_node("gamma", "KEYWORD")
            t.left_node.right_node = E2
            t.left_node.left_node = N
            t.left_node.left_node.right_node = E1

        return None
    
    def build_control_structures(self, x, controlNodeArray):
        # Base case
        if x is None:
            return

        row = self.control_structures_row
        column = self.control_structures_column

        # Handle the current node
        if column >= 200:  # Prevent column overflow
            row += 1
            column = 0

        if row >= 200:  # Prevent row overflow
            return

        controlNodeArray[row][column] = Tree.build_node(x.get_value(), x.get_type())
        column += 1

        # Process children recursively
        if x.left_node:
            self.build_control_structures(x.left_node, controlNodeArray)
        if x.right_node:
            self.build_control_structures(x.right_node, controlNodeArray)

        # Update instance variables
        self.control_structures_row = row
        self.control_structures_column = column

    def build_node(self, x, value, node_type):
        print(f"build_node: Creating node with value={value}, type={node_type}")

        # Handle lambda nodes
        if x.get_value() == "lambda":
            import io
            ss = io.StringIO()

            row = self.control_structures_row
            column = self.control_structures_column
            index = self.control_structures_index
            betaCount = self.control_structures_beta

            t_var_1 = row
            counter = 0
            self.controlNodeArray[row][column] = Tree.build_node("", "")
            self.control_structures_row = 0

            # Count rows in controlNodeArray where first column is not None
            while row < len(self.controlNodeArray) and self.controlNodeArray[row][0] is not None:
                row += 1
                counter += 1
            self.control_structures_row = t_var_1

            ss.write(str(counter))
            index += 1
            str_val = ss.getvalue()
            temp = Tree.build_node(str_val, "deltaNumber")

            self.controlNodeArray[row][column] = temp
            column += 1

            self.controlNodeArray[row][column] = x.left_node
            column += 1

            self.controlNodeArray[row][column] = x
            column += 1

            saved_index = row
            tempj = column + 3

            # Move row to next empty
            while row < len(self.controlNodeArray) and self.controlNodeArray[row][0] is not None:
                row += 1
            column = 0

            # Recursive call
            self.build_control_structures(x.left_node.right_node, self.controlNodeArray)

            self.control_structures_row = saved_index
            self.control_structures_column = tempj
            self.control_structures_index = index
            self.control_structures_beta = betaCount
            return

        # Handle conditional nodes "->"
        elif x.get_value() == "->":
            row = self.control_structures_row
            column = self.control_structures_column
            index = self.control_structures_index
            betaCount = self.control_structures_beta

            saved_index = row
            tempj = column
            nextDelta = index
            counter = row

            ss2 = str(nextDelta)
            temp1 = Tree.build_node(ss2, "deltaNumber")
            self.controlNodeArray[row][column] = temp1
            column += 1

            nextToNextDelta = index
            ss3 = str(nextToNextDelta)
            temp2 = Tree.build_node(ss3, "deltaNumber")
            self.controlNodeArray[row][column] = temp2
            column += 1

            beta = Tree.build_node("beta", "beta")
            self.controlNodeArray[row][column] = beta
            column += 1

            # Count rows with first column not None
            while counter < len(self.controlNodeArray) and self.controlNodeArray[counter][0] is not None:
                counter += 1
            firstIndex = counter
            lamdaCount = index

            self.build_control_structures(x.left_node, self.controlNodeArray)
            diffLc = index - lamdaCount

            # Move row to next empty
            while row < len(self.controlNodeArray) and self.controlNodeArray[row][0] is not None:
                row += 1
            column = 0

            self.build_control_structures(x.left_node.right_node, self.controlNodeArray)

            while row < len(self.controlNodeArray) and self.controlNodeArray[row][0] is not None:
                row += 1
            column = 0

            self.build_control_structures(x.left_node.right_node.right_node, self.controlNodeArray)

            # Update controlNodeArray at saved_index, tempj and tempj + 1
            if diffLc == 0 or row < lamdaCount:
                str5 = str(firstIndex)
            else:
                str5 = str(row - 1)

            self.controlNodeArray[saved_index][tempj].set_value(str5)
            str6 = str(row)
            self.controlNodeArray[saved_index][tempj + 1].set_value(str6)

            self.control_structures_row = saved_index
            self.control_structures_column = 0

            while column < len(self.controlNodeArray[row]) and self.controlNodeArray[row][column] is not None:
                column += 1

            betaCount += 2

            self.control_structures_index = index
            self.control_structures_beta = betaCount
            return

        # Handle tau nodes
        elif x.get_value() == "tau":
            row = self.control_structures_row
            column = self.control_structures_column

            tauLeft = x.left_node
            numOfChildren = 0
            while tauLeft is not None:
                numOfChildren += 1
                tauLeft = tauLeft.right_node

            countNode = Tree.build_node(str(numOfChildren), "CHILDCOUNT")
            self.controlNodeArray[row][column] = countNode
            column += 1

            tauNode = Tree.build_node("tau", "tau")
            self.controlNodeArray[row][column] = tauNode
            column += 1

            self.build_control_structures(x.left_node, self.controlNodeArray)
            x_iter = x.left_node
            while x_iter is not None:
                self.build_control_structures(x_iter.right_node, self.controlNodeArray)
                x_iter = x_iter.right_node

            return

        # Handle other nodes
        else:
            row = self.control_structures_row
            column = self.control_structures_column

            self.controlNodeArray[row][column] = Tree.build_node(x.get_value(), x.get_type())
            column += 1
            self.build_control_structures(x.left_node, self.controlNodeArray)
            if x.left_node is not None:
                self.build_control_structures(x.left_node.right_node, self.controlNodeArray)

        # Update static variables before returning
        self.control_structures_row = row
        self.control_structures_column = column
        self.control_structures_index = index
        self.control_structures_beta = betaCount

    def cse_machine(self, control_struct):
        print("Starting CSE machine execution...")
        
        # Initialize stacks and environment
        if not control_struct:
            print("Error: Empty control structure - nothing to execute")
            return

        control = deque()
        machine_stack = deque()
        environment_stack = deque()
        environment_tracker = deque()

        curr_env_index = 0
        curr_env = Environment()  # Fixed: Use self.Environment()
        curr_env.name = "env0"

        curr_env_index += 1
        machine_stack.append(Tree.build_node(curr_env.name, "ENV"))
        control.append(Tree.build_node(curr_env.name, "ENV"))
        environment_stack.append(curr_env)
        environment_tracker.append(curr_env)

        # Load initial control structure - FIXED: Only load once
        if control_struct and len(control_struct) > 0:
            cor_control_struct = control_struct[0]
            for node in reversed(cor_control_struct):  # Reverse to maintain order
                if node:
                    control.append(node)

        print(f"Initial control stack: {[node.get_value() for node in control]}")
        print(f"Initial machine stack: {[node.get_value() for node in machine_stack]}")

        # MAIN EXECUTION LOOP - FIXED: Single loop, no restarts
        while control:
            next_token = control.pop()
            print(f"Processing token: {next_token.get_value()} ({next_token.get_type()})")
            print(f"Control stack: {[node.get_value() for node in control]}")
            print(f"Machine stack: {[node.get_value() for node in machine_stack]}")

            # Handle nil tokens
            if next_token.get_value() == "nil":
                next_token.set_type("tau")

            # Handle operands and built-in functions
            if (next_token.get_type() in ["INT", "STR", "BOOL", "NIL", "DUMMY"] or 
                next_token.get_value() in ["lambda", "YSTAR", "Print", "Isinteger", "Istruthvalue", 
                                        "Isstring", "Istuple", "Isfunction", "Isdummy", "Stem", 
                                        "Stern", "Conc", "Order", "nil"]):
                
                if next_token.get_value() == "lambda":
                    # FIXED: Proper lambda handling
                    if len(control) < 2:
                        print(f"Warning: Insufficient tokens in control stack for lambda. Available: {len(control)}")
                        machine_stack.append(next_token)
                        continue
                    
                    bound_variable = control.pop()
                    next_delta_index = control.pop()
                    env = Tree.build_node(curr_env.name, "ENV")
                    machine_stack.extend([next_delta_index, bound_variable, env, next_token])
                else:
                    machine_stack.append(next_token)

            # Handle gamma instruction
            elif next_token.get_value() == "gamma":
                if not machine_stack:
                    print("Error: Machine stack empty during gamma processing")
                    continue  # FIXED: Continue instead of return to avoid premature exit
                    
                machine_top = machine_stack[-1]
                
                if machine_top.get_value() == "lambda":  # Apply lambda (CSE Rule 4)
                    machine_stack.pop()  # Pop lambda
                    if len(machine_stack) < 3:
                        print("Error: Insufficient tokens in machine stack for lambda application")
                        continue  # FIXED: Continue instead of return
                        
                    prev_env = machine_stack.pop()
                    bound_variable = machine_stack.pop()
                    next_delta_index = machine_stack.pop()

                    # Create new environment
                    new_env = self.Environment()  # FIXED: Use self.Environment()
                    new_env.name = f"env{curr_env_index}"

                    # Find previous environment
                    temp_env = list(environment_stack)
                    while temp_env and temp_env[-1].name != prev_env.get_value():
                        temp_env.pop()
                    if temp_env:
                        new_env.prev = temp_env[-1]

                    # Bind variables to environment
                    if bound_variable.get_value() == "," and machine_stack and machine_stack[-1].get_value() == "tau":
                        # Handle multiple parameter binding
                        bound_variables = []
                        left_of_comma = bound_variable.left_node
                        while left_of_comma:
                            bound_variables.append(Tree.build_node(left_of_comma.get_value(), left_of_comma.get_type()))
                            left_of_comma = left_of_comma.right_node

                        bound_values = []
                        tau = machine_stack.pop()
                        tau_left = tau.left_node
                        while tau_left:
                            bound_values.append(tau_left)
                            tau_left = tau_left.right_node

                        for var, val in zip(bound_variables, bound_values):
                            if val.get_value() == "tau":
                                res = deque()
                                self.arrangeTuple(val, res)
                            new_env.boundVariable[var] = [val]

                    elif machine_stack and machine_stack[-1].get_value() == "lambda":
                        # Handle lambda binding
                        node_value_vector = []
                        temp = deque()
                        for _ in range(min(4, len(machine_stack))):
                            temp.append(machine_stack.pop())
                        while temp:
                            node_value_vector.append(temp.pop())
                        new_env.boundVariable[bound_variable] = node_value_vector

                    else:
                        # Handle simple binding
                        if machine_stack:
                            bound_val = machine_stack.pop()
                            new_env.boundVariable[bound_variable] = [bound_val]

                    # Update current environment
                    curr_env = new_env
                    control.append(Tree.build_node(curr_env.name, "ENV"))
                    machine_stack.append(Tree.build_node(curr_env.name, "ENV"))
                    environment_stack.append(curr_env)
                    environment_tracker.append(curr_env)

                    # Load next control structure - FIXED: Better error handling
                    try:
                        next_control_index = int(next_delta_index.get_value())
                        if 0 <= next_control_index < len(control_struct):
                            next_delta = control_struct[next_control_index]
                            for node in reversed(next_delta):  # Reverse to maintain order
                                if node:  # FIXED: Check node exists
                                    control.append(node)
                        else:
                            print(f"Warning: Control structure index {next_control_index} out of range")
                    except (ValueError, IndexError) as e:
                        print(f"Error loading control structure: {e}")
                        
                    curr_env_index += 1

                elif machine_top.get_value() == "tau":
                    tau = machine_stack.pop()
                    if not machine_stack:
                        print("Error: No index for tuple selection")
                        continue  # FIXED: Continue instead of return
                    child_index = machine_stack.pop()
                    try:
                        tuple_index = int(child_index.get_value())
                        tau_left = tau.left_node
                        for _ in range(tuple_index - 1):
                            if tau_left:
                                tau_left = tau_left.right_node
                        if tau_left:
                            selected_child = Tree.build_node(tau_left.get_value(), tau_left.get_type())
                            machine_stack.append(selected_child)
                    except (ValueError, AttributeError) as e:
                        print(f"Error in tuple selection: {e}")

                # Add other gamma cases here (YSTAR, built-in functions, etc.)

            # Handle environment restoration
            elif next_token.get_value().startswith("env"):
                stack_to_restore = deque()
                if machine_stack and machine_stack[-1].get_value() == "lambda":
                    for _ in range(min(4, len(machine_stack))):
                        stack_to_restore.append(machine_stack.pop())
                elif machine_stack:
                    stack_to_restore.append(machine_stack.pop())
                    
                if machine_stack:
                    rem_env = machine_stack[-1]
                    if next_token.get_value() == rem_env.get_value():
                        machine_stack.pop()
                        if environment_tracker:
                            environment_tracker.pop()
                        curr_env = environment_tracker[-1] if environment_tracker else None
                        
                while stack_to_restore:
                    machine_stack.append(stack_to_restore.pop())

            # Handle variable lookup
            elif (next_token.get_type() == "ID" and 
                next_token.get_value() not in ["Print", "Isinteger", "Istruthvalue", "Isstring", 
                                            "Istuple", "Isfunction", "Isdummy", "Stem", "Stern", "Conc"]):
                temp = curr_env
                found = False
                while temp and not found:
                    for var, temp_val in temp.boundVariable.items():
                        if next_token.get_value() == var.get_value():
                            for val in temp_val:
                                machine_stack.append(val)
                            found = True
                            break
                    temp = temp.prev
                    
                if not found:
                    print(f"Error: Variable '{next_token.get_value()}' not found in environment")
                    # FIXED: Don't return, just continue

            # Handle operators
            elif hasattr(self, 'isBinaryOperator') and (self.isBinaryOperator(next_token.get_value()) or next_token.get_value() in ["neg", "not"]):
                op = next_token.get_value()
                if self.isBinaryOperator(op):
                    if len(machine_stack) < 2:
                        print(f"Error: Insufficient operands for binary operator {op}")
                        continue  # FIXED: Continue instead of return
                    node1 = machine_stack.pop()
                    node2 = machine_stack.pop()
                    
                    # Handle arithmetic operations
                    if node1.get_type() == "INT" and node2.get_type() == "INT":
                        num1 = int(node1.get_value())
                        num2 = int(node2.get_value())
                        result = None
                        
                        if op == "+":
                            result = Tree.build_node(str(num2 + num1), "INT")
                        elif op == "-":
                            result = Tree.build_node(str(num2 - num1), "INT")
                        elif op == "*":
                            result = Tree.build_node(str(num2 * num1), "INT")
                        elif op == "/":
                            if num1 == 0:
                                print("Error: Division by zero")
                                continue
                            result = Tree.build_node(str(num2 // num1), "INT")
                        # Add other operators as needed
                        
                        if result:
                            machine_stack.append(result)
                            
                elif op == "neg":
                    if not machine_stack:
                        print("Error: No operand for negation")
                        continue
                    node1 = machine_stack.pop()
                    if node1.get_type() == "INT":
                        num1 = int(node1.get_value())
                        machine_stack.append(Tree.build_node(str(-num1), "INT"))

            # Handle conditional (beta)
            elif next_token.get_value() == "beta":
                if len(machine_stack) < 1 or len(control) < 2:
                    print("Error: Insufficient data for conditional")
                    continue  # FIXED: Continue instead of return
                bool_val = machine_stack.pop()
                else_index = control.pop()
                then_index = control.pop()
                
                try:
                    index = int(then_index.get_value()) if bool_val.get_value() == "true" else int(else_index.get_value())
                    if 0 <= index < len(control_struct):
                        next_delta = control_struct[index]
                        for node in reversed(next_delta):
                            if node:  # FIXED: Check node exists
                                control.append(node)
                except (ValueError, IndexError) as e:
                    print(f"Error in conditional: {e}")

            # Handle tuple creation (tau)
            elif next_token.get_value() == "tau":
                if not control:
                    print("Error: No count for tuple creation")
                    continue  # FIXED: Continue instead of return
                no_of_items = control.pop()
                try:
                    num_of_items = int(no_of_items.get_value())
                    if len(machine_stack) < num_of_items:
                        print("Error: Insufficient items for tuple creation")
                        continue  # FIXED: Continue instead of return
                        
                    tuple_node = Tree.build_node("tau", "tau")
                    if num_of_items > 0:
                        tuple_node.left_node = machine_stack.pop()
                        current = tuple_node.left_node
                        for _ in range(1, num_of_items):
                            current.right_node = machine_stack.pop()
                            current = current.right_node
                    machine_stack.append(tuple_node)
                except ValueError as e:
                    print(f"Error in tuple creation: {e}")

            print()  # Empty line for readability

        print("CSE machine execution completed")
        
        # Output final result
        if machine_stack:
            final_result = machine_stack[-1]
            print("Output of the above program is:")
            if hasattr(self, 'cse_flag') and self.cse_flag == 1:
                if final_result.get_type() == "STR":
                    print(self.addSpaces(final_result.get_value()) if hasattr(self, 'addSpaces') else final_result.get_value())
                else:
                    print(final_result.get_value())
                self.cse_flag = 0
            else:
                print(final_result.get_value())
        else:
            print("Warning: Machine stack is empty at completion")


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


