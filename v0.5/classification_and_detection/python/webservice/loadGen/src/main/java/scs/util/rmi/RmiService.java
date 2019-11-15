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
	public void service(String ip,int port) {
		try {
			System.setProperty("java.rmi.server.hostname",ip);
			LocateRegistry.createRegistry(port);
			LoadInterface load = new LoadInterfaceImpl();  
			Naming.rebind("rmi://"+ip+":"+port+"/load", load);
			System.out.println(""+ip+":"+port+" rmi server started");

		} catch (RemoteException e) {
			e.printStackTrace();
			System.out.println(port+"is already used");
		} catch (MalformedURLException e) {
		}
	}

}
