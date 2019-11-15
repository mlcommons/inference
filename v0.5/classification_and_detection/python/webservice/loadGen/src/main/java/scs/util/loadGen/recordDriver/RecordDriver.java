package scs.util.loadGen.recordDriver;

import scs.util.repository.Repository;

public class RecordDriver{
 
	private static RecordDriver driver=null;
	public RecordDriver(){};
	public synchronized static RecordDriver getInstance() {
		if (driver == null) {
			driver = new RecordDriver();
		}  
		return driver;
	}
	/**
	 * start the recorder thread
	 */
	public void execute(int serviceId){
		new RecordExecThread(Repository.recordInterval,serviceId).start(); // execution period=($recordInterval)ms
	}






}