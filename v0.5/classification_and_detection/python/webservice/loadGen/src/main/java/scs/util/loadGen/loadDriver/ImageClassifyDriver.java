package scs.util.loadGen.loadDriver;
 
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;   
import scs.util.repository.Repository;
import scs.util.tools.HttpClientPool;
import scs.util.tools.RandomString; 
/**
 * Image recognition service request class
 * GPU inference
 * @author Yanan Yang
 *
 */
public class ImageClassifyDriver extends AbstractJobDriver{
	/**
	 * Singleton code block
	 */
	private static ImageClassifyDriver driver=null;
	public ImageClassifyDriver(){initVariables();}
	public synchronized static ImageClassifyDriver getInstance() {
		if (driver == null) {
			driver = new ImageClassifyDriver();
		}  
		return driver;
	}
 
	@Override
	protected void initVariables() {
		httpClient=HttpClientPool.getInstance().getConnection();
		queryItemsStr=Repository.imageClassifyBaseURL+"?rand=";
	}

	/**
	 * using countDown to send requests in open-loop
	 */
	public void executeJob(int serviceId) {
		ExecutorService executor = Executors.newCachedThreadPool();
	 
		Repository.onlineQueryThreadRunning[serviceId]=true;
		Repository.sendFlag[serviceId]=true;
		while(Repository.onlineDataFlag[serviceId]==true){
			if(Repository.sendFlag[serviceId]==true){
				CountDownLatch begin=new CountDownLatch(1);
				for (int i=0;i<Repository.realRequestIntensity[serviceId];i++){
					executor.execute(new LoadExecThread(httpClient,queryItemsStr+RandomString.generateString(2),begin,serviceId));
				}
				Repository.sendFlag[serviceId]=false;
				Repository.totalRequestCount[serviceId]+=Repository.realRequestIntensity[serviceId];
				begin.countDown();

			}else{
				try {
					Thread.sleep(10);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				//System.out.println("loader watting "+TestRepository.list.size());
			}
		}
		executor.shutdown();
		while(!executor.isTerminated()){
			try {
				Thread.sleep(2000);
			} catch(InterruptedException e){
				e.printStackTrace();
			}
		}  
		Repository.onlineQueryThreadRunning[serviceId]=false; 
	}



}