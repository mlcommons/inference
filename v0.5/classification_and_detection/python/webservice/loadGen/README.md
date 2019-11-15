# Load generator
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

## Hardware environment
In our experiment environment, the configuration of nodes are shown as below:

| hostname | description | IP | role |
| ---- | ---- | ---- | ----|
| tank-node1 | where the load generator is deployed | 192.168.3.110 | k8s master |
| tank-node3 | where the inference service is deployed | 192.168.3.130 | k8s slave |

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
##  Future work
The latest released version of load generator has satisfied our experiment needs, in the future, we plan to implement these functions as below:
* Distributed load generator
* Diversified output statistics (e.g., PDF, hist graph)

##  Bug report & Question 
We have used the load generator for a long time in our work [2,3], and fixed many bugs that have been found. If you have some new findings, please contact us via Email: ynyang@tju.edu.cn

## Reference
[1] Kasture H, Sanchez D. Tailbench: a benchmark suite and evaluation methodology for latency-critical applications[C]//2016 IEEE International Symposium on Workload Characterization (IISWC). IEEE, 2016: 1-10.<br>
[2] Y. Yang, L. Zhao, Z. Li, L. Nie, P. Chen and K. Li. ElaX: Provisioning Resource Elastically for Containerized Online Cloud Services[C]//2019 IEEE 21st International Conference on High Performance Computing and Communications (HPCC). IEEE, 2019: 1987-1994.<br>
[3] L. Zhao, Y. Yang, K. Zhang, etc. Rhythm: Component-distinguishable Workload
Deployment in Datacenters[C]//EuroSys 2020. ACM. Under review <br>
