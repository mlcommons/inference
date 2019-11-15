package scs.util.loadGen.strategy;

import java.util.Random;

public class RandomDistriPattern implements PatternInterface{
	private Random rand=new Random();
	@Override
	public int getIntervalTime() {
		// TODO Auto-generated method stub
		return rand.nextInt(10);
	}
    
	
	
}
