
import model_compressor


def turn_on_mcp_dumping(quant_model, logit_file_path):

    model_compressor.set_model_to_dump_golden_model(
        logit_file_path,
        quant_model,
        dumping_range="qlv4_linear",
        dumping_mode="only-in-out",
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
    )

    
    