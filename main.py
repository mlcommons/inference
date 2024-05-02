def define_env(env):

   @env.macro
   def mlperf_inference_implementation_readme(spaces, model, implementation):
     pre_space = ""
     for i in range(1,spaces):
       pre_space  = pre_space + " "
     f_pre_space = pre_space
     pre_space += " "

     content=""
     if implementation == "reference":
       devices = [ "CPU", "CUDA", "ROCm" ]
       if model.lower() == "resnet50":
         frameworks = [ "Onnxruntime", "Tensorflow", "Deepsparse" ]
       elif model.lower() == "retinanet":
         frameworks = [ "Onnxruntime", "Pytorch" ]
       elif "bert" in model.lower():
         frameworks = [ "Onnxruntime", "Pytorch", "Tensorflow" ]
       else:
         frameworks = [ "Pytorch" ]
     elif implementation == "nvidia":
       devices = [ "CUDA" ]
       frameworks = [ "TensorRT" ]
     elif implementation == "intel":
       devices = [ "CPU" ]
       frameworks = [ "Pytorch" ]
     elif implementation == "qualcomm":
       devices = [ "QAIC" ]
       frameworks = [ "Glow" ]
     elif implementation == "cpp":
       devices = [ "CPU", "CUDA" ]
       frameworks = [ "Onnxruntime" ]

     if model.lower() == "bert-99.9":
       categories = [ "Datacenter" ]
     elif "dlrm" in model.lower() or "llama2" in model.lower():
       categories = [ "Datacenter" ]
     else:
       categories = [ "Edge", "Datacenter" ]

     for category in categories:
       if category == "Edge":
         scenarios = [ "Offline", "SingleStream" ]
         if model.lower() in [ "resnet50", "retinanet" ]:
           scenarios.append("Multistream")
       elif category == "Datacenter":
         scenarios = [ "Offline", "Server" ] 

       content += f"{pre_space}=== \"{category.lower()}\"\n\n"

       cur_space = pre_space + "    "
       scenarios_string = ", ".join(scenarios)
     
       content += f"{cur_space}#### {category} category \n\n{cur_space} In the {category.lower()} category, {model} has {scenarios_string} scenarios and all the scenarios are mandatory for a closed division submission.\n\n"


       for framework in frameworks:
         cur_space1 = cur_space + "    "
         content += f"{cur_space}=== \"{framework}\"\n"
         content += f"{cur_space1}##### {framework} framework\n\n"

         for device in devices:
           if framework.lower() == "deepsparse":
             if device.lower() != "cpu":
               continue
           cur_space2 = cur_space1 + "    "
           content += f"{cur_space1}=== \"{device}\"\n"
           content += f"{cur_space2}###### {device} device\n\n"
         
           for scenario in scenarios:
             cur_space3 = cur_space2 + "    "
             content += f"{cur_space2}=== \"{scenario}\"\n{cur_space3}####### {scenario}\n"
             run_cmd = mlperf_inference_run_command(spaces+16, model, implementation, framework.lower(), category.lower(), scenario, device.lower(), "valid")
             content += run_cmd
           content += f"{cur_space2}=== \"All Scenarios\"\n{cur_space3}####### All Scenarios\n"
           run_cmd = mlperf_inference_run_command(spaces+16, model, implementation, framework.lower(), category.lower(), "All Scenarios", device.lower(), "valid")
           content += run_cmd

     return content

     
   @env.macro
   def mlperf_inference_run_command(spaces, model, implementation, framework, category, scenario, device="cpu", execution_mode="test", test_query_count="20"):
     pre_space = ""
     for i in range(1,spaces):
       pre_space  = pre_space + " "
     f_pre_space = pre_space
     pre_space += "  "

     if scenario == "All Scenarios":
       scenario_variation_tag = ",_all-scenarios"
       scenario_option = ""
     else:
       scenario_variation_tag = ""
       scenario_option = f"\\\n {pre_space} --scenario={scenario}"

     cmd_suffix = f" \\\n {pre_space} --docker" 
     #cmd_suffix = f"" 
     if execution_mode == "test":
       cmd_suffix += f" \\\n {pre_space} --test_query_count={test_query_count}"

     return f"\n{f_pre_space} ```bash\n{f_pre_space} cm run script --tags=run-mlperf,inference{scenario_variation_tag} \\\n {pre_space} --model={model} \\\n {pre_space} --implementation={implementation} \\\n {pre_space} --framework={framework} \\\n {pre_space} --category={category} {scenario_option} \\\n {pre_space} --execution-mode={execution_mode} \\\n {pre_space} --device={device} {cmd_suffix}\n{f_pre_space} ```\n"
