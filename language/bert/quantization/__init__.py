import yaml

import model_compressor  # isort:skip

from quantization.custom_symbolic_trace import custom_symbolic_trace  # isort:skip
from quantization.utils import get_kwargs  # isort:skip


def quantize_model(model, qconfig_path, qparam_path, qformat_path):
    with open(qconfig_path, "r") as f:
        qconfig = yaml.safe_load(f)
    model, _, _ = custom_symbolic_trace(model)
    model.config.use_cache = False

    return model_compressor.create_quantsim_model(
        model,
        qformat_path=qformat_path,
        qparam_path=qparam_path,
        disable_inout=(True, True),
        **get_kwargs(model_compressor.create_quantsim_model, qconfig)
    )
