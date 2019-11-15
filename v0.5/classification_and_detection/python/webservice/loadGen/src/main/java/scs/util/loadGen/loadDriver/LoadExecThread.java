package scs.util.loadGen.loadDriver;

import java.util.concurrent.CountDownLatch; 
import org.apache.http.impl.client.CloseableHttpClient;
import scs.util.repository.Repository;
import scs.util.tools.HttpClientPool; 
/**
 * 请求发送线程,发送请求并记录时间
 * @author yanan
 *
 */
public class LoadExecThread extends Thread{
	private CloseableHttpClient httpclient;//httpclient对象
	private String url;//请求的url
	private CountDownLatch begin;
	private int serviceId;
	/**
	 * 线程构造方法
	 * @param httpclient httpclient对象
	 * @param url 要访问的链接 
	 */
	public LoadExecThread(CloseableHttpClient httpclient,String url,CountDownLatch begin,int serviceId){
		this.httpclient=httpclient;
		this.url=url;
		this.begin=begin;
		this.serviceId=serviceId;
	}

	@Override
	public void run(){ 
		try{
			begin.await();
			int time=HttpClientPool.getResponseTime(httpclient,url);
			synchronized (Repository.onlineDataList.get(serviceId)) {
				Repository.onlineDataList.get(serviceId).add(time);
			}
		} catch (Exception e) {
			e.printStackTrace();
		} 

	}



}
