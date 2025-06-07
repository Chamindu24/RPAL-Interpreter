# RPAL Interpreter in Python

A complete implementation of an RPAL language processor including lexical analysis, parsing, AST generation, tree standardization, and CSE machine execution.

## Features

- Lexical analyzer following RPAL specifications
- Parser implementing RPAL grammar rules
- Abstract Syntax Tree (AST) generation
- Tree standardization (ST)
- CSE machine for program execution
- Command-line interface with multiple output options

## Installation

1. Ensure Python 3.6+ is installed
2. Clone this repository
3. No additional dependencies required (pure Python implementation)

## Usage

### Basic Commands

```bash
# Execute an RPAL program
python myrpal.py filename.rpal

# Generate token sequence
python myrpal.py -lex filename.rpal

# Generate Abstract Syntax Tree
python myrpal.py -ast filename.rpal

# Generate Standardized Tree
python myrpal.py -st filename.rpal
