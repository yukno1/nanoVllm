from dataclasses import fields
from config import Config


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}

    def exit(self):
        pass
