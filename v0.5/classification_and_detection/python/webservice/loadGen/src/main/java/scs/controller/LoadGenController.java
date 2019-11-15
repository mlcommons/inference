package scs.controller;

import java.util.ArrayList;
import java.util.List;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse; 
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;

import net.sf.json.JSONArray;
import scs.pojo.QueryData;
import scs.util.format.DataFormats;
import scs.util.loadGen.loadDriver.ImageClassifyDriver;
import scs.util.loadGen.recordDriver.RecordDriver;
import scs.util.repository.Repository; 
/**
 * Load generator controller class, it includes interfaces as follows:
 * 1.Control the open/close of load generator
 * 2.Support the dynamic QPS setting
 * 3.support GPI for user to view the realtime latency and QPS
 * @author YananYang 
 * @date 2019-11-12
 * @email ynyang@tju.edu.cn
 */
@Controller
public class LoadGenController { 
	private DataFormats dataFormat=DataFormats.getInstance();
	private Repository instance=Repository.getInstance();
	/**
	 * Start the load generator for latency-critical services
	 * @param intensity The concurrent request number per second (RPS)
	 * @param serviceId The index id of web inference service, started from 0 by default
	 */
	@RequestMapping("/startOnlineQuery.do")
	public void startOnlineQuery(HttpServletRequest request,HttpServletResponse response,
			@RequestParam(value="intensity",required=true) int intensity,
			@RequestParam(value="serviceId",required=true) int serviceId){
		try{ 
			intensity=intensity<=0?1:intensity;//validation
			Repository.realRequestIntensity[serviceId]=intensity;
			if(Repository.onlineQueryThreadRunning[serviceId]==true){
				System.out.println("online query threads"+serviceId+" are already running");
			}else{
				Repository.onlineDataFlag[serviceId]=true; 
				Repository.statisticsCount[serviceId]=0;//init statisticsCount
				Repository.totalQueryCount[serviceId]=0;//init totalQueryCount
				Repository.totalRequestCount[serviceId]=0;//init totalRequestCount
				Repository.onlineDataList.get(serviceId).clear();//clear onlineDataList
				Repository.windowOnlineDataList.get(serviceId).clear();//clear windowOnlineDataList
				if(serviceId==0){
					RecordDriver.getInstance().execute(serviceId);
					ImageClassifyDriver.getInstance().executeJob(serviceId);//
				}
			}
		
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	/**
	 * dynamically set the RPS of web-inference service
	 * @param request
	 * @param response
	 * @param intensity The concurrent request number per second (RPS)
	 */
	@RequestMapping("/setIntensity.do")
	public void setIntensity(HttpServletRequest request,HttpServletResponse response,
			@RequestParam(value="intensity",required=true) int intensity,
			@RequestParam(value="serviceId",required=true) int serviceId){
		try{ 
			intensity=intensity<=0?1:intensity;//合法性校验
			Repository.realRequestIntensity[serviceId]=intensity;
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	/**
	 * Stop the load generator for latency-critical services
	 * @param request
	 * @param response
	 */
	@RequestMapping("/stopOnlineQuery.do")
	public void stopOnlineQuery(HttpServletRequest request,HttpServletResponse response,
			@RequestParam(value="serviceId",required=true) int serviceId){
		try{
			Repository.onlineDataFlag[serviceId]=false; 
		}catch(Exception e){
			e.printStackTrace();
		}
	}
	/**
	 * Turn into the GPI page to see the real-time request latency line
	 * @param request
	 * @param response
	 * @param model
	 * @return
	 */
	@RequestMapping("/goOnlineQuery.do")
	public String goOnlineQuery(HttpServletRequest request,HttpServletResponse response,Model model,
			@RequestParam(value="serviceId",required=true) int serviceId){
		StringBuffer strName=new StringBuffer();
		StringBuffer strData=new StringBuffer();
		StringBuffer HSeries=new StringBuffer();
		strName.append("{name:'queryTime',");
		strData.append("data:[");

		List<QueryData> list=new ArrayList<QueryData>();
		list.addAll(Repository.windowOnlineDataList.get(serviceId));
		while(list.size()<Repository.windowSize){
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			list.clear();
			list.addAll(Repository.windowOnlineDataList.get(serviceId));
		} 
		int size=list.size();

		for(int i=0;i<size-1;i++){
			strData.append("[").append(list.get(i).getGenerateTime()).append(",").append(list.get(i).getQueryTime99th()).append("],");
		}
		strData.append("[").append(list.get(size-1).getGenerateTime()).append(",").append(list.get(size-1).getQueryTime99th()).append("]]");

		HSeries.append(strName).append(strData).append("}");
		
		model.addAttribute("seriesStr",HSeries.toString());  
		model.addAttribute("serviceId",serviceId);
		
		return "onlineData";
	}

	/**
	 * obtain the latest 99th latency of last second
	 * this is done by Ajax, no pages switch
	 * @param request
	 * @param response
	 */
	@RequestMapping("/getOnlineQueryTime.do")
	public void getOnlineQueryTime(HttpServletRequest request,HttpServletResponse response,
			@RequestParam(value="serviceId",required=true) int serviceId){
		try{
			response.getWriter().write(JSONArray.fromObject(Repository.latestOnlineData[serviceId]).toString().replace("}",",\"OnlineAvgQueryTime\":"+dataFormat.subFloat(instance.getOnlineAvgQueryTime(serviceId),2)+"}"));
		}catch(Exception e){
			e.printStackTrace();
		}
	}

}
