package scs.util.rmi;

import java.rmi.RemoteException;
import java.rmi.server.UnicastRemoteObject;
import scs.util.loadGen.loadDriver.ImageClassifyDriver;
import scs.util.repository.Repository; 

public class LoadInterfaceImpl extends UnicastRemoteObject implements LoadInterface {

	private static final long serialVersionUID = 1L;

	public LoadInterfaceImpl() throws RemoteException {
		super();
		// TODO Auto-generated constructor stub
	}

	@Override
	public float getWindowAvgPerSec99thLatency(int serviceId) {
		// TODO Auto-generated method stub
		return Repository.windowAvgPerSec99thQueryTime[serviceId];
	}

	@Override
	public int setIntensity(int intensity,int serviceId){
		// TODO Auto-generated method stub
		Repository.realRequestIntensity[serviceId]=intensity;
		return 1;
	}

	@Override
	public int getRealQueryIntensity(int serviceId) throws RemoteException {
		// TODO Auto-generated method stub
		return Repository.realQueryIntensity[serviceId];
	}

	@Override
	public int getRealRequestIntensity(int serviceId) throws RemoteException {
		// TODO Auto-generated method stub
		return Repository.realRequestIntensity[serviceId];
	} 

	@Override
	public float getWindowAvgServiceRate(int serviceId) throws RemoteException {
		// TODO Auto-generated method stub
		return Repository.latestOnlineData[serviceId].getWindowAvgServiceRate();
	}

	@Override
	public float getRealPerSec99thLatency(int serviceId) throws RemoteException {
		// TODO Auto-generated method stub
		return Repository.latestOnlineData[serviceId].getQueryTime99th();
	}

//	@Override
//	public float getLcCurLatency95th(int serviceId) throws RemoteException {
//		// TODO Auto-generated method stub
//		return Repository.latestOnlineData[serviceId].getQueryTime95th();
//	}
//
//	@Override
//	public float getLcCurLatency999th(int serviceId) throws RemoteException {
//		// TODO Auto-generated method stub
//		return Repository.latestOnlineData[serviceId].getQueryTime999th();
//	}

	@Override
	public void execStartHttpLoader(int serviceId) throws RemoteException {
		// TODO Auto-generated method stub
		try{ 
			Repository.realRequestIntensity[serviceId]=1;
			if(Repository.onlineQueryThreadRunning[serviceId]==true){
				System.out.println("online query threads"+serviceId+"are already running");
			}else{
				Repository.onlineDataFlag[serviceId]=true; 
				Repository.statisticsCount[serviceId]=0;//init statisticsCount
				Repository.totalQueryCount[serviceId]=0;//init totalQueryCount
				Repository.totalRequestCount[serviceId]=0;//init totalRequestCount
				Repository.onlineDataList.get(serviceId).clear();//clear onlineDataList
				Repository.windowOnlineDataList.get(serviceId).clear();//clear windowOnlineDataList
				if(serviceId==0){
					ImageClassifyDriver.getInstance().executeJob(serviceId);
				}
			}

		}catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	public void execStopHttpLoader(int serviceId) throws RemoteException {
		Repository.onlineDataFlag[serviceId]=false;
	}
}


