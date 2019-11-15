package scs.util.loadGen.strategy;

public class IndexDistriPattern implements PatternInterface{
	
	@Override
	public int getIntervalTime() {
		// TODO Auto-generated method stub
		return this.getExponentialVariable(2.0);
	}
    
	/**
     * 获取指数分布的响应函数
     * @param x
     * @param lamda
     * @return 访问的间隔秒数
     * 加类型，单位，次数 的函数
     * 存每次请求响应的时间
     */
	private int getExponentialVariable(double lamda){
        int x = 0;
        double y = Math.random(), cfd = getExponentialProbability(x,lamda);
        while (cfd < y){
            x++;
            cfd += getExponentialProbability(x,lamda);
        }
        return x;

    }
    private double  getExponentialProbability(int x, double lamda){
        double c = Math.exp(-lamda * x);
        return lamda * c;
    }
	
}
