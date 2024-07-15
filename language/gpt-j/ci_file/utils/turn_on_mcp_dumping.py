
import model_compressor


def turn_on_mcp_dumping(model_dict, prefill_logit_file_path, decode_logit_file_path):

    model_compressor.set_model_to_dump_golden_model(
        prefill_logit_file_path,
        model_dict["prefill"],
        dumping_range='lm_head',
        dumping_mode='only-in-out', 
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
        dump_in_append_mode=True,)

    model_compressor.set_model_to_dump_golden_model(
        decode_logit_file_path,
        model_dict["decode"],
        dumping_range='lm_head',
        dumping_mode='only-in-out', 
        qlv4_skip_output_rounding=False,
        dumping_before_rounding=True,
        dump_in_append_mode=True,)
    
    