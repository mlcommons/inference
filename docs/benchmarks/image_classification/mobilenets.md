# Image Classification using Mobilenet models

Mobilenet models are not official MLPerf models and so cannot be used for a Closed division MLPerf inference submission. But since they can be run with Imagenet dataset, we are allowed to use them for Open division submission. Only CPU runs are supported now. 

## TFLite Backend

=== "Mobilenet-V1"
    ### Mobilenet V1
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_mobilenet-v1 \
     --adr.compiler.tags=gcc
    ```
=== "Mobilenet-V2"
    ### Mobilenet V2
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_mobilenet-v2 \
     --adr.compiler.tags=gcc
    ```
=== "Mobilenet-V2"
    ### Mobilenet V2
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_mobilenet-v2 \
     --adr.compiler.tags=gcc
    ```
=== "Mobilenets"
    ### Mobilenet V1,V2,V3
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_mobilenet \
     --adr.compiler.tags=gcc
    ```
=== "Efficientnet"
    ### Efficientnet
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_efficientnet \
     --adr.compiler.tags=gcc
    ```

## ARMNN Backend
=== "Mobilenet-V1"
    ### Mobilenet V1
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_armnn,_mobilenet-v1 \
     --adr.compiler.tags=gcc
    ```
=== "Mobilenet-V2"
    ### Mobilenet V2
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_armnn,_mobilenet-v2 \
     --adr.compiler.tags=gcc
    ```
=== "Mobilenet-V2"
    ### Mobilenet V2
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_armnn,_mobilenet-v2 \
     --adr.compiler.tags=gcc
    ```
=== "Mobilenets"
    ### Mobilenet V1,V2,V3
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_armnn,_mobilenet \
     --adr.compiler.tags=gcc
    ```
=== "Efficientnet"
    ### Efficientnet
    ```bash
    cm run script --tags=run,mobilenet-models,_tflite,_armnn,_efficientnet \
     --adr.compiler.tags=gcc
    ```

