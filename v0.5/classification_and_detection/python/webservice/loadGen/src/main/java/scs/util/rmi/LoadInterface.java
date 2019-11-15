package scs.util.rmi;

import java.rmi.Remote;
import java.rmi.RemoteException; 
/**
 * RMI interface class, which is used to control the load generator
 * The functions can be call by remote client code
 * @author Yanan Yang
 * @date 2019-11-11
 * @address TianJin University
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
