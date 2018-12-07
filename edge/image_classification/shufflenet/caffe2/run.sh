```
#!/bin/bash

DATABASE_ADDRESS=52.91.209.88
DATABASE_NAME=shufflenet
MODEL_NAME=ShuffleNet_Caffe2
MODEL_VERSION=1.0
NUM_FILE_PARTS=-1
BATCH_SIZE=1
TRACE_LEVEL=MODEL_TRACE

docker run --network host -t -v $HOME:/root carml/caffe2-agent:amd64-cpu-latest predict dataset \
      --fail_on_error=true \
      --verbose \
      --publish=true \
      --publish_predictions=false \
      --gpu=0 \
      --num_file_parts=$NUM_FILE_PARTS \
      --batch_size=$BATCH_SIZE \
      --model_name=$MODEL_NAME \
      --model_version=$MODEL_VERSION \
      --database_address=$DATABASE_ADDRESS \
      --database_name=$DATABASE_NAME \
      --trace_level=$TRACE_LEVEL
```