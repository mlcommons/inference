def define_env(env):

   @env.macro
   def mlperf_inference_implementation_readme(spaces, model, implementation):
     pre_space = ""
     for i in range(1,spaces):
       pre_space  = pre_space + " "
     f_pre_space = pre_space
     pre_space += " "

     content=""
     scenarios = []
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
       if model not in [ "bert-99", "bert-99.9", "gptj-99", "gptj-99.9" ]:
            return pre_space+"    WIP"
       devices = [ "CPU" ]
       frameworks = [ "Pytorch" ]
     elif implementation == "qualcomm":
       devices = [ "QAIC" ]
       frameworks = [ "Glow" ]
     elif implementation == "cpp":
       devices = [ "CPU", "CUDA" ]
       frameworks = [ "Onnxruntime" ]
     elif implementation == "ctuning-cpp":
       scenarios = [ "SingleStream" ]
       devices = [ "CPU" ]
       if model.lower() == "resnet50":
         frameworks = [ "TFLite" ]
       else:
         frameworks = []

     if model.lower() == "bert-99.9":
       categories = [ "Datacenter" ]
     elif "dlrm" in model.lower() or "llama2" in model.lower():
       categories = [ "Datacenter" ]
     else:
       categories = [ "Edge", "Datacenter" ]

     for category in categories:
       if category == "Edge" and not scenarios:
         scenarios = [ "Offline", "SingleStream" ]
         if model.lower() in [ "resnet50", "retinanet" ]:
           scenarios.append("MultiStream")
       elif category == "Datacenter" and not scenarios:
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
         
           content += f"{cur_space2}###### Docker Setup Command\n\n"
           test_query_count=get_test_query_count(model, implementation, device)

           content += mlperf_inference_run_command(spaces+12, model, implementation, framework.lower(), category.lower(), "Offline", device.lower(), "test", test_query_count, True)
           content += f"{cur_space2}The above command should get you to an interactive shell inside the docker container and do a quick test run for the Offline scenario. Once inside the docker container please do the below commands to do the accuracy + performance runs for each scenario.\n\n"
           content += f"{cur_space2}<details>\n"
           content += f"{cur_space2}<summary> Please click here to see more options for the docker launch </summary>\n\n"
           content += f"{cur_space2}* `--docker_cm_repo <Custom CM repo URL>`: to use a custom fork of cm4mlops repository inside the docker image\n\n"
           content += f"{cur_space2}* `--docker_cache=no`: to not use docker cache during the image build\n"

           if device.lower() not in [ "cuda" ]:
             content += f"{cur_space2}* `--docker_os=ubuntu`: ubuntu and rhel are supported. \n"
             content += f"{cur_space2}* `--docker_os_version=20.04`: [20.04, 22.04] are supported for Ubuntu and [8, 9] for RHEL\n"

           content += f"{cur_space2}</details>\n"
           run_suffix = ""
           run_suffix += f"\n{cur_space2}    ###### Run Options\n\n"
           run_suffix += f"{cur_space2}     * Use `--division=closed` to do a closed division submission which includes compliance runs\n\n"
           run_suffix += f"{cur_space2}     * Use `--rerun` to do a rerun even when a valid run exists\n\n"

           for scenario in scenarios:
             cur_space3 = cur_space2 + "    "
             content += f"{cur_space2}=== \"{scenario}\"\n{cur_space3}####### {scenario}\n"
             run_cmd = mlperf_inference_run_command(spaces+16, model, implementation, framework.lower(), category.lower(), scenario, device.lower(), "valid")
             content += run_cmd
             content += run_suffix

           content += f"{cur_space2}=== \"All Scenarios\"\n{cur_space3}####### All Scenarios\n"
           run_cmd = mlperf_inference_run_command(spaces+16, model, implementation, framework.lower(), category.lower(), "All Scenarios", device.lower(), "valid")
           content += run_cmd
           content += run_suffix

     return content

   def get_test_query_count(model, implementation, device, num_devices=1):

       if model == "resnet50":
           p_range = 1000
       elif model in [ "retinanet", "bert-99", "bert-99.9" ]:
           p_range = 100
       else:
           p_range = 50
       if device == "cuda":
           p_range *= 40
       p_range *= num_devices

       return p_range


     
   @env.macro
   def mlperf_inference_run_command(spaces, model, implementation, framework, category, scenario, device="cpu", execution_mode="test", test_query_count="20", docker=False):
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

     if docker:
       docker_cmd_suffix = f" \\\n {pre_space} --docker --quiet"
       docker_cmd_suffix += f" \\\n {pre_space} --test_query_count={test_query_count}"

       docker_setup_cmd = f"\n{f_pre_space} ```bash\n{f_pre_space} cm run script --tags=run-mlperf,inference,_find-performance,_full{scenario_variation_tag} \\\n {pre_space} --model={model} \\\n {pre_space} --implementation={implementation} \\\n {pre_space} --framework={framework} \\\n {pre_space} --category={category} {scenario_option} \\\n {pre_space} --execution-mode=test \\\n {pre_space} --device={device} {docker_cmd_suffix}\n{f_pre_space} ```\n"

       return docker_setup_cmd

     else:
       cmd_suffix = f"\\\n {pre_space} --quiet"

       if execution_mode == "test":
         cmd_suffix += f" \\\n {pre_space} --test_query_count={test_query_count}"

       run_cmd = f"\n{f_pre_space} ```bash\n{f_pre_space} cm run script --tags=run-mlperf,inference{scenario_variation_tag} \\\n {pre_space} --model={model} \\\n {pre_space} --implementation={implementation} \\\n {pre_space} --framework={framework} \\\n {pre_space} --category={category} {scenario_option} \\\n {pre_space} --execution-mode={execution_mode} \\\n {pre_space} --device={device} {cmd_suffix}\n{f_pre_space} ```\n"

       return run_cmd
