import yaml
from _ctx_impl import build_context as _build

def build(query, cfg_or_path):
    """
    Универсальная обёртка: принимает либо словарь с конфигом, либо путь к config.yaml.
    """
    if isinstance(cfg_or_path, dict):
        cfg = cfg_or_path
    else:
        cfg = yaml.safe_load(open(cfg_or_path, "r", encoding="utf-8"))
    return _build(query, cfg)
