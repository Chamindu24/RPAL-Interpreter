class Token:
    def __init__(self, token_type: str = "", token_value: str = ""):
        self.type = token_type  # Token type (e.g., ID, INT, etc.)
        self.value = token_value  # Actual string value

    def set_type(self, token_type: str):
        self.type = token_type

    def set_value(self, token_value: str):
        self.value = token_value

    def get_type(self) -> str:
        return self.type

    def get_value(self) -> str:
        return self.value

    def __ne__(self, other):
        return not (self.type == other.type and self.value == other.value)

    def __repr__(self):
        return f"Token(type='{self.type}', value='{self.value}')"
