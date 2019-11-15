package scs.util.loadGen.strategy;

public class PossionDistriPattern implements PatternInterface{
	@Override
	public int getIntervalTime() {
		// TODO Auto-generated method stub
		return this.getPossionVariable(2.0);
	}
	/**
	 * 获取泊松变量
	 * @param lamda
	 * @return 访问的间隔秒数
	 */
	private int getPossionVariable(double lamda){  
		int x = 0;  
		double y = Math.random(), cdf = getPossionProbability(x, lamda);  
		while (cdf < y) {  
			x++;  
			cdf += getPossionProbability(x, lamda);  
		}  
		return x+1;  
	}  
	private double getPossionProbability(int k, double lamda) {  
		double c = Math.exp(-lamda), sum = 1;  
		for (int i = 1; i <= k; i++) {  
			sum *= lamda / i;  
		}  
		return sum * c;  
	}   
	 
}
