package scs.pojo;

public class QueryData{
	private long generateTime;
	private float queryTime99th; //99th latency per second
	//private float queryTime95th;
	//private float queryTime999th;
	private int avgQps; //window size average QPS
	private int avgRps; //window size average RPS
	private float windowAvgServiceRate; //window size average QPS/RPS
	private int realQps; //average QPS per second
	private int realRps; //average RPS per second
	
	public QueryData(long generateTime, int queryTime99th) {
		super();
		this.generateTime = generateTime;
		this.queryTime99th = queryTime99th;
	}
	public QueryData(QueryData data) {
		this.generateTime = data.getGenerateTime();
		this.queryTime99th = data.getQueryTime99th();
		this.avgQps = data.getAvgQps();
		this.avgRps = data.getAvgRps();
		this.windowAvgServiceRate = data.getWindowAvgServiceRate();
	}
	public QueryData() {
	}
	public long getGenerateTime() {
		return generateTime;
	}
	public void setGenerateTime(long generateTime) {
		this.generateTime = generateTime;
	}
	public float getQueryTime99th() {
		return queryTime99th;
	}
	public void setQueryTime99th(float queryTime99th) {
		this.queryTime99th = queryTime99th;
	}
	public int getAvgQps(){
		return avgQps;
	}
	public void setAvgQps(int qps){
		this.avgQps=qps;
	}
	public int getAvgRps(){
		return avgRps;
	}
	public void setAvgRps(int rps){
		this.avgRps=rps;
	}
	public float getWindowAvgServiceRate() {
		return windowAvgServiceRate;
	}
	public void setWindowAvgServiceRate(float serviceRate) {
		this.windowAvgServiceRate = serviceRate;
	}
//	public float getQueryTime95th() {
//		return queryTime95th;
//	}
//	public float getQueryTime999th() {
//		return queryTime999th;
//	}
//	public void setQueryTime95th(float queryTime95th) {
//		this.queryTime95th = queryTime95th;
//	}
//	public void setQueryTime999th(float queryTime999th) {
//		this.queryTime999th = queryTime999th;
//	}
	public int getRealQps() {
		return realQps;
	}
	public int getRealRps() {
		return realRps;
	}
	public void setRealQps(int realQps) {
		this.realQps = realQps;
	}
	public void setRealRps(int realRps) {
		this.realRps = realRps;
	}
 
}