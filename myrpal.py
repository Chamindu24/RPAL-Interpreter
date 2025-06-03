import sys
import os

from parser import Parser  # Assumes parser.py contains the Parser class


def main():
    args = sys.argv

    if len(args) > 1:
        argv_idx = 1
        ast_flag = 0
        lex_flag = 0

        if len(args) == 3:
            argv_idx = 2
            if args[1] == "-ast":
                ast_flag = 1
            elif args[1] == "-st":
                ast_flag = 2
            elif args[1] == "-lex":
                lex_flag = 1

        filepath = args[argv_idx]

        if not os.path.exists(filepath):
            print(f'File "{filepath}" not found!')
            return 1

        # Read file contents
        with open(filepath, 'r') as file:
            file_str = file.read()

        file_array = list(file_str)

        # Create a parser object and parse
        rpal_parser = Parser(file_array, 0, len(file_str), ast_flag)

        if lex_flag:
            rpal_parser.parse()
            if not rpal_parser.isParsingComplete():
                print("Error: Parsing is not complete")
                return 1

            try:
                with open("output_token_sequence.txt", "r") as in_file:
                    for line in in_file:
                        print(line.strip())
            except FileNotFoundError:
                print("Unable to open file output_token_sequence.txt")
                return 1
        else:
            rpal_parser.cse_parse()
    else:
        print("Error: Incorrect number of inputs")


if __name__ == "__main__":
    main()
