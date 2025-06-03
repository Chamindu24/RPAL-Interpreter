from typing import Optional, Dict, List
from tree import Tree  # Assuming your Tree class is in tree.py


class Environment:
    def __init__(self, name: str = "env0", prev: Optional['Environment'] = None):
        self.prev: Optional[Environment] = prev
        self.name: str = name
        self.bound_variable: Dict[Tree, List[Tree]] = {}

    # Copy constructor (deep copy of environment)
    @classmethod
    def copy(cls, env: 'Environment') -> 'Environment':
        new_env = cls(name=env.name, prev=env.prev)
        # Deep copy the bound_variable map
        new_env.bound_variable = {k: v.copy() for k, v in env.bound_variable.items()}
        return new_env

    # Assignment operator behavior (mimic Python's default assignment)
    def assign_from(self, env: 'Environment'):
        self.name = env.name
        self.prev = env.prev
        self.bound_variable = {k: v.copy() for k, v in env.bound_variable.items()}
