---
hide:
  - toc
---

# Graph Neural Network using R-GAT 

## Dataset

The benchmark implementation run command will automatically download the validation and calibration datasets and do the necessary preprocessing. In case you want to download only the datasets, you can use the below commands.

=== "Full Dataset"
    R-GAT validation run uses the IGBH dataset consisting of 547,306,935 nodes and 5,812,005,639 edges.

    ### Get Full Dataset
    ```
    mlcr get,dataset,igbh,_full -j
    ```

=== "Debug Dataset"
    R-GAT debug run uses the IGBH debug dataset(tiny).

    ### Get Full Dataset
    ```
    mlcr get,dataset,igbh,_debug -j
    ```

## Model
The benchmark implementation run command will automatically download the required model and do the necessary conversions. In case you want to only download the official model, you can use the below commands.

Get the Official MLPerf R-GAT Model

=== "PyTorch"

    ### PyTorch
    ```
    mlcr get,ml-model,rgat -j
    ```
## Automated command for submission generation via MLCFlow

Please see the [new docs site](https://docs.mlcommons.org/inference/submission/) for an automated way to generate submission through MLCFlow.     

