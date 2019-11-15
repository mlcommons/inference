package scs.util.repository;

import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import scs.pojo.QueryData;
import scs.util.rmi.RmiService; 
/**
 * System static repository class
 * Provide memory storage in the form of static variables for data needed in system operation
 * Including some system parameters, application run data, control signs and so on
 * @author Yanan Yang
 *
 */
public class Repository{ 
	private static Repository repository=null;
	private Repository(){}
	public synchronized static Repository getInstance() {
		if (repository == null) {
			repository = new Repository();
		}
		return repository;
	}  

	private final static int NUMBER_LC=2; //number of LC services 
	
	public static int windowSize=60; //window size of latency recorder
	public static int recordInterval=1000; //record interval of latency recorder
	
	/**
	 * System variables of online load generator module 
	 */
	
	public static boolean[] onlineQueryThreadRunning=new boolean[NUMBER_LC]; 
	public static boolean[] onlineDataFlag=new boolean[NUMBER_LC]; 
	public static boolean[] sendFlag=new boolean[NUMBER_LC]; 
	
	public static int[] realRequestIntensity=new int[NUMBER_LC]; 
	public static int[] realQueryIntensity=new int[NUMBER_LC];  
	private static int[] windowOnLineDataListCount=new int[NUMBER_LC];	
	public static int[] statisticsCount=new int[NUMBER_LC];	
	public static int[] totalRequestCount=new int[NUMBER_LC];
	public static int[] totalQueryCount=new int[NUMBER_LC];
	
	public static List<ArrayList<Integer>> onlineDataList=new ArrayList<ArrayList<Integer>>();
	public static List<ArrayList<Integer>> tempOnlineDataList=new ArrayList<ArrayList<Integer>>();
	public static List<ArrayList<QueryData>> windowOnlineDataList=new ArrayList<ArrayList<QueryData>>();
	private static List<ArrayList<QueryData>> tempWindowOnlineDataList=new ArrayList<ArrayList<QueryData>>();
	
	public static QueryData[] latestOnlineData=new QueryData[NUMBER_LC];
	public static float[] windowAvgPerSec99thQueryTime=new float[NUMBER_LC]; 
	
	public static String serverIp="";
	public static int rmiPort;
	public static String imageClassifyBaseURL="";
	
	/**
	 * static code
	 */
	static {
		initList();
		readProperties();
		RmiService.getInstance().service(Repository.serverIp, Repository.rmiPort);//start the RMI service
	}
	/**
	 * read properties 
	 */
	private static void readProperties(){
		Properties prop = new Properties();
		InputStream is = Repository.class.getResourceAsStream("/conf/sys.properties");
		try {
			prop.load(is);
		} catch (IOException e) {
			e.printStackTrace();
		}
		Repository.windowSize=Integer.parseInt(prop.getProperty("windowSize").trim());
		Repository.serverIp=prop.getProperty("serverIp").trim();
		Repository.rmiPort=Integer.parseInt(prop.getProperty("rmiPort").trim());
		Repository.recordInterval=Integer.parseInt(prop.getProperty("recordInterval").trim()); 
		Repository.imageClassifyBaseURL=prop.getProperty("imageClassifyBaseURL").trim();
	}	
	
	/**
	 * init 
	 */
	private static void initList(){
		 for(int i=0;i<NUMBER_LC;i++){
			 onlineDataList.add(new ArrayList<Integer>());
			 tempOnlineDataList.add(new ArrayList<Integer>());
			 windowOnlineDataList.add(new ArrayList<QueryData>());
			 tempWindowOnlineDataList.add(new ArrayList<QueryData>());
		 }
	}	

	/**
	 * Adds a new data to the window array
	 * Loop assignment in Repository.windowSize
	 * @param data
	 */
	public void addWindowOnlineDataList(QueryData data, int serviceId){
		latestOnlineData[serviceId]=data;
		Repository.realQueryIntensity[serviceId]=data.getRealQps();
		if(windowOnlineDataList.get(serviceId).size()<Repository.windowSize){
			windowOnlineDataList.get(serviceId).add(data);
		}else{
			windowOnlineDataList.get(serviceId).set(windowOnLineDataListCount[serviceId]%Repository.windowSize,new QueryData(data));
			windowOnLineDataListCount[serviceId]++;
		}
	}

	/**
	 * Calculate the variance of query time
	 * @return 
	 */
	public float getOnlineVarQueryTime(int serviceId){
		tempWindowOnlineDataList.get(serviceId).clear();
		tempWindowOnlineDataList.get(serviceId).addAll(Repository.windowOnlineDataList.get(serviceId));
		int size=tempWindowOnlineDataList.get(serviceId).size();
		float avgQueryTime=0;

		for(QueryData item:tempWindowOnlineDataList.get(serviceId)){
			avgQueryTime+=item.getQueryTime99th();
		}
		avgQueryTime=avgQueryTime/size;

		float var=0;
		for(QueryData item:tempWindowOnlineDataList.get(serviceId)){
			var+=Math.pow((item.getQueryTime99th()-avgQueryTime),2); 
		}
		var=var/size;
		return var;
	}
	/**
	 * Calculate the mean of query time
	 * @return 
	 */
	public float getOnlineAvgQueryTime(int serviceId){
		while (Repository.windowOnlineDataList.get(serviceId).isEmpty()) {
			 try {
				Thread.sleep(200);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		tempWindowOnlineDataList.get(serviceId).clear();
		tempWindowOnlineDataList.get(serviceId).addAll(Repository.windowOnlineDataList.get(serviceId));
		int size=tempWindowOnlineDataList.get(serviceId).size();
		float avgQueryTime=0;
		for(QueryData item:tempWindowOnlineDataList.get(serviceId)){
			avgQueryTime+=item.getQueryTime99th();
		} 
		avgQueryTime=avgQueryTime/size; 
		Repository.windowAvgPerSec99thQueryTime[serviceId]=avgQueryTime;
		return avgQueryTime; 
	}



}
