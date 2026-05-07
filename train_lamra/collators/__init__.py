COLLATORS = {}

def register_collator(name):
    def register_collator_cls(cls):
        if name in COLLATORS:
            return COLLATORS[name]
        COLLATORS[name] = cls
        return cls
    return register_collator_cls

# from .qwen2_vl_2b_hardneg import Qwen2VL2BHardNegDataCollator
# from .qwen2_vl_7b import Qwen2VL7BDataCollator
# from .qwen2_vl_2b import Qwen2VL2BDataCollator
# from .deepseek_ocr_3b import DeepseekOCR3BDataCollator
from .qwen3_vl_2b import Qwen3VL2BDataCollator
from .qwen3_vl_2b_hardneg import Qwen3VL2BHardNegDataCollator
# from .internvl_25 import InternVL25DataCollator