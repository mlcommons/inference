
echo "Clearing caches."
sync && echo 3 | tee /host_proc/sys/vm/drop_caches


cd /root

common_opt="--count 500 --time 10"
dataset="--dataset-path /data"

if [ $profile == "resnet50-tf" ] ; then
    model=/model/resnet50_v1.pb
fi
if [ $profile == "resnet50-onnxruntime" ] ; then
    model=/model/resnet50_v1.onnx
fi


start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING RUN AT $start_fmt"

python3 python/main.py --profile $profile $common_opt --model $model $dataset --output /output/results.json $EXTRA_OPS

end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING RUN AT $end_fmt"
