import yaml
from torch.fx import GraphModule

import model_compressor  # isort:skip

from quantization.utils import get_kwargs  # isort:skip


def quantize_model(
    model: GraphModule, qconfig_path: str, qparam_path: str, qformat_path: str
) -> GraphModule:
    with open(qconfig_path, "r") as f:
        qconfig = yaml.safe_load(f)
    model.config.use_cache = False

    return model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig)
    )
