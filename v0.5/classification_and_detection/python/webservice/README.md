# Web inference service for MLperf benchmark 

Based on the [MLperf](https://github.com/mlperf/inference), we packaged the inference model into web services in addtion to the existing script operation mode. And we also contribute a load generator to send http requests for measurement. The web service is implemented using python flask+guicorn framework which can work in concurrent requests scenarios. The load generator is implemented using java httpclient+threadpool that provides [open-loop](https://ieeexplore_ieee.xilesou.top/abstract/document/7581261/) [1] request load, and it can generate dynamically changeable request loads. We designed a web GUI for the realtime latency watch (e.g., 99th tail-latency, QPS, RPS, etc.). Meanwhile, we also provided RMI interfaces for external call to support runtime control and data collection.

## Web inference
* currently support `mobilenet-coco300-tf` and `resnet-coco1200-tf` model inference, in continuous update... (easy to scale)
* support mutil-process server end in single GPU card
* provide docker image for fast build and source code for download 
* support the service building and scaling using kubernetes 
## Load generator
* support thousands level concurrent request per second
* support dynamiclly changeable QPS in open-loop
* support web GUI for real-time latency watch
* support RMI interface for external call

> Web inference codes is in `inference/v0.5/classification_and_detection/python/webservice` <br>
> Load generator code is in `inference/v0.5/classification_and_detection/python/webservice/loadGen`
# Building and installation
The steps of building the web service and load generator are shown as below:

## Hardware environment
In our experiment environment, the configuration of nodes are shown as below:

| hostname | description | IP | role |
| ---- | ---- | ---- | ----|
| tank-node1 | where the load generator is deployed | 192.168.3.110 | k8s master |
| tank-node3 | where the inference service is deployed | 192.168.3.130 | k8s slave |

## Part 1: Build web inference service
Here we use the `mobilenet-coco300-tf` model as an example, which uses 300px*300px size of coco dataset and tensorflow framework

### Step 1. Prepare the web service runtime environment
We provide the well-configured docker image for download (Recommended)
```Bash
$ docker pull ynyang/mobilenet-coco-flask:v1
```
In another way, you can use the Dockerfile to build image (Optional), the `Dockerfile` is shown as below:

```
# Content of DockerFile
# install the runtime env and library
FROM tensorflow/tensorflow:1.13.1-gpu-py3-jupyter
MAINTAINER YananYang
RUN pip install pillow
RUN pip install opencv-python
RUN apt-get update && apt-get install -y libsm6 libxrender1 libxext-dev
RUN pip install Cython
RUN pip install pycocotools
RUN pip install gunicorn
RUN pip install flask 
RUN pip install gevent
rm -rf /var/lib/apt/lists/*
```
The command for building docker image using `Dockerfile` 
```bash
$ docker build -t ynyang/mobilenet-coco-flask:v1 .
```
### Step 2. Prepare the image dataset of coco

| dataset | picture download link | annotation download link |
| ---- | ---- | ----|
| coco (validation) | http://images.cocodataset.org/zips/val2017.zip | http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip

Download the coco image dataset and scale the pictures to 300px*300px size. The scale tool is [here](../../../tools/upscale_coco).
You can run it for ssd-mobilenet like:
```bash
python upscale_coco.py --inputs /data/coco/ --outputs /data/coco-300 --size 300 300 --format png
```
The ready dataset and nanotation should be like this:
```bash
$ ls /home/tank/yanan/data/coco-300/
annotations  val2017
```
The storage directory path can be optionally specify in your system, but notice the directory `annotations` and `val2017` should be **in the same level directory**

### Step 3. Prepare the trained inference model
| model | framework | accuracy | dataset | model link | model source | precision | notes |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| ssd-mobilenet 300x300 quantized finetuned | tensorflow | mAP 0.23594 | coco resized to 300x300 | [from zenodo](https://zenodo.org/record/3252084/files/mobilenet_v1_ssd_8bit_finetuned.tar.gz) | Habana | int8 | ??? |

Download the model from the linke, and the ready trained inference model should be like this:
```bash
$ ls /home/tank/yanan/model
ssd_mobilenet_v1_coco_2018_01_28.pb
```
### Step 4. Prepare the source code for web service
Download the inference source code of [MLperf](https://github.com/mlperf/inference), and confirm the code has web inference interface module

The ready source code should be like this:
```bash
$ ls /home/tank/yanan/inference
build            CONTRIBUTING.md  loadgen_pymodule_setup_lib.py  README.md                venv
BUILD.gn         DEPS             loadgen_pymodule_setup_src.py  SubmissionExample.ipynb
build_overrides  LICENSE.md       Makefile                       third_party
calibration      loadgen          others                         v0.5
```

Modify the `inference/v0.5/classification_and_detection/python/backend_tf.py` and add three lines of code as below, the added code minimizes GPU the memory usage of inference model
```Python
  1 """
  2 tensorflow backend (https://github.com/tensorflow/tensorflow)
  3 """
  4
  5 # pylint: disable=unused-argument,missing-docstring,useless-super-delegation
  6
  7 import tensorflow as tf
  8 from tensorflow.core.framework import graph_pb2
  9
 10
 11 import backend
 12 tf_config = tf.ConfigProto()            #<------add code here
 13 tf_config.gpu_options.per_process_gpu_memory_fraction = 0.01  #<------add code here
 14 session = tf.Session(config=tf_config)  #<------add code here

class BackendTensorflow(backend.Backend):
```
To adapt the tensorflow verison in docker image, you need also modify the 45 line of code in `backend_tf.py` as below, if not, the code will not work
```Python
 41         graph_def = graph_pb2.GraphDef()
 42         with open(model_path, "rb") as f:
 43             graph_def.ParseFromString(f.read())
 44         g = tf.import_graph_def(graph_def, name='')
 45         self.sess = tf.compat.v1.Session(graph=g) #<------modify code here, origin code:self.sess = tf.Session(graph=g)
 46         return self
 47
 48     def predict(self, feed):
```

### Step 5. Prepare the start script for web service
`start-service-mobilenet-coco300-tf.sh` is used to start the web service process of  inference model when container is created 

The content of `start-service-mobilenet-coco300-tf.sh` is shown as below:
```bash
#!/bin/bash
# These shell commands are executed in container
cd /home/inference/v0.5/classification_and_detection/python/webservice
gunicorn -w 8 -b 0.0.0.0:5000 web_mobilenet_coco300_tf:app
```
> -w: number of web service process in one GPU card

The ready start script of service should be like this:
```bash
$ ls /home/tank/yanan/script
start-service-mobilenet-coco300-tf.sh
```
### Step 6. Create docker container

We use the kubernetes to create pod and service for the inference model, the container mounts host's files and directories to it's volume when it is created

Files and directories in host machine are shown as below:
```bash
/home/tank/yanan
           |----/data/coco-300
           |          |----/annotations/instances_val2017.json
           |          +----/val2017/00000xxxxx.jpg
           |----/model/ssd_mobilenet_v1_coco_2018_01_28.pb
           |----/script/start-service-mobilenet-coco300-tf.sh
           +----/inference/v0.5/classification_and_detection/*
```
Files and directories in container are shown as below:
```bash
/----/home
 |    |----/script/start-service-mobilenet-coco300-tf.sh
 |    +----/inference/v0.5/classification_and_detection/*   
 |----/data/coco-300
 |          |----/annotations/instances_val2017.json
 |          +----/val2017/00000xxxxx.jpg
 +----/model/ssd_mobilenet_v1_coco_2018_01_28.pb
```
Then use `create-pod.yaml` to create pod in kubernetes `master` node
```bash
$ kubectl create -f create-pod.yaml
pod/mobilenet-coco300-pod created
$ kubectl get pods -n yyn
NAME                     READY   STATUS    RESTARTS   AGE
mobilenet-coco300-pod0   1/1     Running   0          19s
```

The content of `create-pod.yaml` is shown as below:
```python
apiVersion: v1                          # api version
kind: Pod                               # component type
metadata:
  name: mobilenet-coco300-pod0
  namespace: yyn
  labels:                               # label
    app: mobilenet-coco300-app-label
spec:
  nodeName: tank-node3                  # node that you want to deploy pod in
  containers:
  - name: mobilenet-coco300-flask-con      # container name
    image: ynyang/mobilenet-coco-flask:v1   # image
    imagePullPolicy: IfNotPresent
    ports:
    - containerPort: 5000              # service port in container
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"                       # use GPU card 0 (e.g. 1,2)
    command: ["sh", "-c", "/home/script/start-service-mobilenet-coco300-tf.sh"]
    volumeMounts:                      # paths in container
    - name: model-path
      mountPath: /model
    - name: data-path
      mountPath: /data
    - name: code-path
      mountPath: /home/inference
    - name: script-path
      mountPath: /home/script
  volumes:  
  - name: model-path
    hostPath:                          # paths in host machine
      path: /home/tank/yanan/model
  - name: data-path
    hostPath:
      path: /home/tank/yanan/data
  - name: code-path
    hostPath:
      path: /home/tank/yanan/inference
  - name: script-path
    hostPath:
      path: /home/tank/yanan/script
```

After the pod is created, use `create-service.yaml` to create service in kubernetes master node
```bash
$ kubectl create -f create-service.yaml
pod/mobilenet-coco300-service created
$ kubectl get service -n yyn
NAME                        TYPE       CLUSTER-IP      EXTERNAL-IP   PORT(S)          AGE
mobilenet-coco300-service   NodePort   10.105.250.36   <none>        5000:31500/TCP   5d6h
```
The content of `create-service.yaml` is shown as below:
```bash
apiVersion: v1
kind: Service
metadata:
  name: mobilenet-coco300-service
  namespace: yyn
spec:
  selector:
    app: mobilenet-coco300-app-label
  type: NodePort
  ports:
  - name: http
    protocol: TCP
    port: 5000  # the port of container
    targetPort: 5000 # the cluster port for internal calls
    nodePort: 31500  # the cluster port for external calls
```
Test the web service is successfully started, `192.168.3.130` is the IP of node (`tank-node3`) that deployed the pod, which provides `31500` port mapping the pod 
```bash
$ curl 192.168.3.130:31500/gpu
ok
```
The output of "ok" means the web service is available now, and we use `nvidia-smi -l 1` command can see the inference processes has been created in `tank-node3`

```bash
$ nvidia-smi -l 1
Wed Nov 13 22:16:48 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 430.40       Driver Version: 430.40       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:18:00.0 Off |                  N/A |
| 27%   26C    P8    15W / 250W |   5268MiB / 11019MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 208...  Off  | 00000000:3B:00.0 Off |                  N/A |
| 27%   24C    P8    19W / 250W |      0MiB / 11019MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
|   2  GeForce RTX 208...  Off  | 00000000:86:00.0 Off |                  N/A |
| 27%   24C    P8    22W / 250W |      0MiB / 11019MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|    0     38812      C   /usr/bin/python3                             657MiB |
|    0     38825      C   /usr/bin/python3                             657MiB |
|    0     38827      C   /usr/bin/python3                             657MiB |
|    0     38830      C   /usr/bin/python3                             657MiB |
|    0     38894      C   /usr/bin/python3                             657MiB |
|    0     38977      C   /usr/bin/python3                             657MiB |
|    0     39025      C   /usr/bin/python3                             657MiB |
|    0     39089      C   /usr/bin/python3                             657MiB |
+-----------------------------------------------------------------------------+
```

# Part 2: Build Load generator
The load generator is a Java maven project which is implemented using httpclient+threadpool that works in [open-loop](https://ieeexplore_ieee.xilesou.top/abstract/document/7581261/) [1], it has a web GUI for the realtime latency watch (e.g., 99th tail-latency, QPS, RPS, etc.). Meanwhile, the load generator provides RMI interface for external call and supports dynamically changeable request loads

* support thousands level concurrent request per second in single node
* support web GUI for real-time latency watch
* support dynamiclly changeable QPS in open-loop
* support RMI interface for external call

##  Code architecture

```bash
/LoadGen
 |----/build/sdcloud.war   # the executable war package
 |----/src/main
           |----/java/scs/
           |          |----/controller/*      # MVC controller layer
           |          |----/pojo/*	 # entity bean layer
           |          +----/util
           |                |----/format/*  # format time and data
           |                |----/loadGen
           |                |     |----/loadDriver/*  # generate request loads
           |                |     |----/recordDriver/* # record request metrics
           |                |     +----/strategy/* 
           |                |----/respository/*   # in-memory data storage  
           |                |----/rmi/*       # RMI service and interfaces   
           |                +----/tools/*   # some tools
           |----/resources/*  # configuration files
           +----/webapp/*     # GUI pages
```



##  Build load generator
The load generator is writen in Java, it can be deployed in container or host machine, and we need install Java JDK and apache tomcat before using it 


### Step 1: Install Java JDK
Download java jdk and install to /usr/local/java/
```bash
$ wget https://download.oracle.com/otn/java/jdk/8u231-b11/5b13a193868b4bf28bcb45c792fce896/jdk-8u231-linux-x64.tar.gz
$ tar -zxvf jdk-8u231-linux-x64.tar.gz /usr/local/java/
```
Modify the `/etc/profile` file
```bash
$ vi /erc/profile
```
Config Java environment variables, append the following content into the file
```bash
export JAVA_HOME=/usr/local/java/jdk1.8.0_231
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=$PATH:${JAVA_HOME}/bin 
```
Enable the configuration
```
$ source /etc/profile
$ java -version
java version "1.8.0_231"
Java(TM) SE Runtime Environment (build 1.8.0_231-b12)
Java HotSpot(TM) 64-Bit Server VM (build 25.231-b12, mixed mode)
```

### Step 2: Install apache tomcat
Download apache tomcat and install to /usr/local/tomcat/
```bash
$ http://mirrors.tuna.tsinghua.edu.cn/apache/tomcat/tomcat-8/v8.5.47/bin/apache-tomcat-8.5.47.tar.gz
$ tar -zxvf apache-tomcat-8.5.47.tar.gz /usr/local/tomcat/
```
### Step 3: Deploy the load generator into tomcat

> An executable war package has been provided in loadGen/build/, you can also build the source code uses `jar` or `eclipse IDE`

Deploy the web package into tomcat webapp/
```bash
$ mv loadGen/build/sdcloud.war /usr/local/tomcat/apache-tomcat-8.5.47/webapp
$ /usr/local/tomcat/apache-tomcat-8.5.47/bin/startup.sh
```
Validate if the depolyment is successful
```bash
$ curl http://localhost:8080/sdcloud/
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<html>
...
welcome to the load generator page!
...
</body>
```
The command will output the content of a welcome page, means the war package has been deployed successfully

Then modify the configuration file in `loadGen` and restart tomcat
```bash
$ vi LoadGen/src/main/resources/conf/sys.properties
```
Modify the content of `sys.properties` as below:
```python
# URL of web inference service that can be accessed by http 
imageClassifyBaseURL=http://192.168.3.130:31500/gpu 
# node IP that deployed load generator
serverIp=192.168.3.110     
# the port of RMI service provided by load generator 
rmiPort=22222
# window size of latency recorder, which can be seen from GUI page
windowSize=60
# record interval of latency, default: 1000ms
recordInterval=1000
```
Then restart the tomcat
```bash
$ /usr/local/tomcat/apache-tomcat-8.5.47/bin/shutdown.sh
$ /usr/local/tomcat/apache-tomcat-8.5.47/bin/startup.sh
```
If error occured when restart like `java.rmi.server.ExportException: Port already in use: 22222`, we need to kill the process that uses this port and restart tomcat
```bash
$ netstat -apn |grep 22222
tcp6       1      0 202.113.8.12:36284     127.0.0.1:22222       USING  35487/java
$ kill 35487
$ /usr/local/tomcat/apache-tomcat-8.5.47/bin/startup.sh
```

### Step 4: Test the load generator
Open the exlporer and visit url `http://192.168.3.110:8080/sdcloud/`
The GUI page is shown as below:

![realtime Latency](https://github.com/yananYangYSU/book/blob/master/welcome.png?raw=true)
#### Type 1: Using the URL interfaces
We provide four URL interfaces to control the load generator as below:

| url interface | description |  type  | parameter |
| ---- | ---- | ---- | ---- |
| `startOnlineQuery.do?intensity=1&serviceId=0` |  start to generate the request load | GET |intensity: concurrent requests per second (RPS) <br> serviceId: inference service index id  |
| `goOnlineQuery.do?serviceId=0` |  visit the real-time latency page | GET |  serviceId: inference service index id  |
| `setIntensity.do?intensity=20&serviceId=0` |   change the RPS dynamically | GET | intensity: RPS <br> serviceId: inference service index id  |
| `stopOnlineQuery.do?serviceId=0` |   stop the load generator | GET |  ---  |

> The index id of mobilenet-coco300-tf is 0, and the second inference service should be set to 1, ..., etc. The supported service in load generator is easy to scale 


For examle, firstly, click the `startOnlineQuery.do?intensity=1&serviceId=0` link to generate request loads, and you will see the page in a waiting state (circle loading), then after `$windowSize seconds`, click the `goOnlineQuery.do?serviceId=0` link to watch the real-time latency as below. 

![realtime Latency](https://github.com/yananYangYSU/BigTailDemo/blob/master/realtime-latency.png?raw=true)
Metrics in real-time latency watch page:
* realRPS: The concurrent requests number of last second from client, RPS (request per second)
* realQPS: The response number of last second in server, QPS (query per second)
* AvgRPS: The average concurrent RPS in `$windowsize` time scale
* AvgQPS: The average QPS in `$windowsize` time scale
* SR: The average service rate in `$windowsize` time scale, `SR=AvgQPS/AvgRPS*100%`
* queryTime: The 99th tail-latency of the concurrent requests per second
* Avg99th: The average `queryTimes` in `$windowsize` time scale   

When the load generator is running, click the `setIntensity.do?intensity=N&serviceId=0` link to change the RPS, please replace `N` to the number of concurrent requests per second you want. Finally, click `stopOnlineQuery.do?serviceId=0` to stop the load testing

#### Type 2: Using the RMI interfaces
We also provide the Java RMI interfaces for the remote funciton calls in users' external code without clicking URL links. Using RMI, you can control the load generator and collect metric data. The RMI interface file is in `LoadGen/src/main/java/scs/util/rmi/LoadInterface.java`, the interface functions are shown as below: 
```Java
package scs.util.rmi;

import java.rmi.Remote;
import java.rmi.RemoteException; 
/**
 * RMI interface class, which is used to control the load generator
 * The functions can be call by remote client code
 * @author Yanan Yang
 * @date 2019-11-11
 * @address TANKLab, TianJin University, China
 */
public interface LoadInterface extends Remote{
	public float getWindowAvgPerSec99thLatency(int serviceId) throws RemoteException; //return the value of Avg99th
	public float getRealPerSec99thLatency(int serviceId) throws RemoteException; //return the value of queryTime
	//public float getWindowSize95thRealLatency(int serviceId) throws RemoteException; //return the value of queryTime (95th), unused
	//public float getLcCurLatency999thRealLatency(int serviceId) throws RemoteException; //return the value of queryTime (99.9th), unused
	public int getRealQueryIntensity(int serviceId) throws RemoteException; //return the value of realQPS
	public int getRealRequestIntensity(int serviceId) throws RemoteException;  //return the value of realRPS
	public float getWindowAvgServiceRate(int serviceId) throws RemoteException; //return the value of SR
	public void execStartHttpLoader(int serviceId) throws RemoteException; //start load generator for serviceId
	public void execStopHttpLoader(int serviceId) throws RemoteException; //stop load generator for serviceId
	public int setIntensity(int intensity,int serviceId) throws RemoteException; //change the RPS dynamically
}
```
 When tomcat starts, the server side will automatically setup the RMI service using `serverIp` and `rmiPort` in `LoadGen/src/main/resources/conf/sys.properties`, the RMI service function is shown as below:


```Java
package scs.util.rmi; 
 
import java.net.MalformedURLException;
import java.rmi.Naming;
import java.rmi.RemoteException;
import java.rmi.registry.LocateRegistry;

public class RmiService {
	private static RmiService loadService=null;
	private RmiService(){}
	public synchronized static RmiService getInstance() {
		if (loadService == null) {
			loadService = new RmiService();
		}
		return loadService;
	}  
	public void setupService(String serverIp,int rmiPort) {
		try {
			System.setProperty("java.rmi.server.hostname",serverIp);
			LocateRegistry.createRegistry(rmiPort);
			LoadInterface load = new LoadInterfaceImpl();  
			Naming.rebind("rmi://"+serverIp+":"+rmiPort+"/load", load);
		} catch (RemoteException e) {
			e.printStackTrace();
		} catch (MalformedURLException e) {}
	}
}
```

The client side need to setup the RMI connection before controling the load generator, a stand connection function is shown as below:
```Java
	private static void setupRmiConnection(){
		try {
			LoadInterface loader=(LoadInterface) Naming.lookup("rmi://192.168.3.110:22222/load");
		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (RemoteException e) {
			e.printStackTrace();
		} catch (NotBoundException e) {
			e.printStackTrace();
		}
		if(loader!=null){
			System.out.println(Repository.loaderRmiUrl +"connection successed");
		}else{
			System.out.println(Repository.loaderRmiUrl +"connection failed");
		}
	}
```
More tutorial of RMI interface can be see from [here](https://docs.oracle.com/javase/7/docs/technotes/guides/rmi/hello/hello-world.html)

# Performance evaluation of tools
We evaluate the performance of load generator tool and web inference service and show the results as below: 
### Performance testing of load generator
To aviod the performance bottleneck of web service interfering the testing results, we set the request url in `LoadGen/src/main/resources/conf/sys.properties` to an empty url (this url does nothing and just returns 'helloworld')
```bash
mageClassifyBaseURL=http://192.168.3.130:31500/helloworld
```
Then we test the concurrent ability of load generator and collect the latency data, the client and server are deployed individually on two nodes that connected with 1Gbps WLAN

![evaluationOfLoadGen](https://github.com/yananYangYSU/book/blob/master/evaluationOfLoadGen.png?raw=true)

Fig.1 depicts the 99th tail-latency collected by load generator with the workloads ranges from `RPS=1` to `RPS=2000`, the requests are sent using multi-threads in [open-loop](https://ieeexplore_ieee.xilesou.top/abstract/document/7581261/), the worst 99th tail-latency < 250ms when the `RPS=2000`, which shows the low queue latency in load generator. Fig.2 shows the 99th tail-latency increases linearly with the `RPS`, this demonstrates the load generator is well designed and has a good performance of workload scalability. Fig.3 shows the CPU usage in server end with increasing `RPS`, which has a same trend with the tail-latency in Fig.1. The inference service consumes < 0.5 CPU core when `RPS=400`, while the CPU usage no more than 2 CPU cores when `RPS=2000`, it demonstrates the low overhead of guicorn+flask framework

### Evalution for web inference service

We evaluate the inference service using 8 web processes in ubuntu 16.04 with one GPU card GeForce RTX 2080 Ti, and the CPU is Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz, evaluation results are shown as below:

| RPS | 1 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 99th latency /ms | 50.4 | 116 | 194 | 269 | 356 | 443 | 539 | 640 | 731 | 820 | 906 |
| GPU usage % | 0.116 | 9.83 | 17.08 | 25.1 | 33.93 | 41.866| 51.15 | 61.7 | 70.5 | 79.5 | 88.4 |

The inference 
For `mobilenet-coco300-tf`, the image preprocess is done in CPU and the model inference is done with GPU, the average preprocess time is 0.010s, while the inference time for one picture is 0.065s. 

From this table, we can see that single GPU card can support 50 concurrent requests per second while the 99th latency is nearly close to 1s, this is because the image inference takes a lot of SM core in GPU (nearly 90%), and we have varified this phenomenon is similar as the production environment in company, so the design of our web inference is rational and reliable  


## Question and Support
We have used the load generator for a long time in our work [2,3], and fixed many bugs that have been found. 
While web inference service is continusly in update, any question please contact us via email. The author is a first year Phd student in TianJin University, China

Enjoy coding, enjoy life<br>
Email: ynyang@tju.edu.cn



## Reference
[1] Kasture H, Sanchez D. Tailbench: a benchmark suite and evaluation methodology for latency-critical applications[C]//2016 IEEE International Symposium on Workload Characterization (IISWC). IEEE, 2016: 1-10.<br>
[2] Y. Yang, L. Zhao, Z. Li, L. Nie, P. Chen and K. Li. ElaX: Provisioning Resource Elastically for Containerized Online Cloud Services[C]//2019 IEEE 21st International Conference on High Performance Computing and Communications (HPCC). IEEE, 2019: 1987-1994.<br>
[3] L. Zhao, Y. Yang, K. Zhang, etc. Rhythm: Component-distinguishable Workload
Deployment in Datacenters[C]//EuroSys 2020. ACM. Under review <br>
