from typing import Any

class Dotdict(dict):
    def __getattr__(self, __name: str) -> Any:
        if __name in self:
            return self[__name]
        raise AttributeError(__name)   # hasattr 동작/자동완성에 중요

    def __setattr__(self, __name: str, __value: Any) -> None:
        super().__setitem__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        super().__delitem__(__name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

DATA_PATH = "/data"
FAIRFACE_DATA_PATH = "./data/fairface"
PROMPT_DATA_PATH   = "./data/prompt_templates.csv"
FACET_DATA_PATH    = "./data/FACET"
UTKFACE_DATA_PATH  = "./data/utkface/UTKFace"

__all__ = [
    "Dotdict",
    "DATA_PATH",
    "FAIRFACE_DATA_PATH",
    "PROMPT_DATA_PATH",
    "FACET_DATA_PATH",
    "UTKFACE_DATA_PATH",
]
