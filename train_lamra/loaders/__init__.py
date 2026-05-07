LOADERS = {}

def register_loader(name):
    def register_loader_cls(cls):
        if name in LOADERS:
            return LOADERS[name]
        LOADERS[name] = cls
        return cls
    return register_loader_cls

# from .deepseek_ocr_3b import DeepseekOCREmbeddingModel
# from .qwen2_vl_7b import Qwen2VL7BModelLoader
# from .qwen2_vl_2b import Qwen2VL2BModelLoader
from .qwen3_vl_2b import Qwen3VL2BModelLoader
# from .internvl_25 import InternVL2_52BModelLoader