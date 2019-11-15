package scs.util.format;

import java.math.BigDecimal;
import java.util.ArrayList; 
import java.util.Collections;
import java.util.List;

/**
 * Data formatting utility class
 * @author Yanan Yang
 * @desc Base conversion, string inversion, etc
 * @date 2017-8-18
 */
public class DataFormats {
	private static DataFormats dataFormat=null;

	private DataFormats(){}

	public synchronized static DataFormats getInstance() {
		if (dataFormat == null) {  
			dataFormat = new DataFormats();
		}  
		return dataFormat;
	}
	public float subFloat(float value,int length){
		BigDecimal bd=new BigDecimal((double)value);    
		bd=bd.setScale(length,4);    
		return bd.floatValue();   
	}
	/**
	 * hexadecimal (length=2) to binary (length=8)
	 * @param hexStr hexadecimal (length=2) e.g., 1F 
	 * @return binary (length=8),
	 */
	public String hexToBinary(String hexStr){
		String zeroStr = "00000000";
		String binStr = Integer.toBinaryString(Integer.parseInt(hexStr, 16));
		return zeroStr.substring(0, 8 - binStr.length()) + binStr;
	}
	/**
	 * binary (length=8)  to hexadecimal (length=2)  
	 * @param binStr binary (length=8)
	 * @return hexadecimal (length=2) e.g., 1F 
	 */
	public String binaryToHex(String binStr){
		String zeroStr="00";
		String hexStr=Integer.toHexString(Integer.parseInt(binStr,2));
		return zeroStr.substring(0,2-hexStr.length())+hexStr;
	}
	/**
	 * 10进制数转指定位数位数的2进制
	 * @param intStr 2位16进制数 例如 1F 
	 * @param length 2进制字符串的长度
	 * @return 8位2进制,左端补0
	 */
	public String decimalToBinary(int decimal,int length){
		StringBuffer zeroStr=new StringBuffer();
		for(int i=0;i<length;i++){
			zeroStr.append("0");
		}
		String binStr=Integer.toBinaryString(decimal);
		return zeroStr.substring(0,length-binStr.length())+binStr;
	}
	/**
	 * decimal to hexadecimal
	 * @param decimal 10进制数
	 * @return
	 */
	public String decimalToHex(int decimal){
		String result="";
		String binStr=Integer.toHexString(decimal);//10进制转16进制
		int length=binStr.length();
		if(length>3&&length<=8){
			result="00000000".substring(0,8-length)+binStr;
		}else if(length>=0&&length<=3){
			result="000".substring(0,3-length)+binStr;
		}
		return result;//进行补位操作,左端补0凑成8位二进制数
	}
	/**
	 * decimal to hexadecimal
	 * @param decimal  
	 * @param length The length of result
	 * @return 
	 */
	public String decimalToHex(int decimal,int length){
		StringBuffer zeroStr=new StringBuffer();
		for(int i=0;i<length;i++){
			zeroStr.append("0");
		}
		String binStr=Integer.toHexString(decimal).toUpperCase();
		return zeroStr.substring(0,length-binStr.length())+binStr;
	}
	/**
	 * reverse string
	 * @param oldStr old string e.g.,:"111111112222222"
	 * @return new String e.g.,"222222211111111"
	 */
	public String reverseStr(String oldStr){
		char[] tmp=oldStr.toCharArray();
		int length=tmp.length;
		int rows=length>>3;
		List<String> rowsList=new ArrayList<String>();
		for(int i=0;i<rows;i++){
			rowsList.add(oldStr.substring((i<<3),(i<<3)+8));
		}
		StringBuffer newStr=new StringBuffer();
		for(int i=rows-1;i>=0;i--)
			newStr.append(rowsList.get(i));
		return newStr.toString();
	}
	/**
	 * Calculate the p quantile
	 * @param data
	 * @param p 0-1
	 * @return
	 */
	public float percentile(ArrayList<Float> data,float p){
		int n = data.size();  
		Collections.sort(data);  
		float px =  p*(n-1);  
		int i = (int)java.lang.Math.floor(px);  
		Float g = px - i;  
		if(g==0){   
			return data.get(i);  
		}else{  
			return ((1-g)*data.get(i)+g*data.get(i+1));  
		}  
	} 

	/**
	 * float reserve two decimal fractions
	 * 51.1f->51.10f
	 * @param value
	 * @return
	 */
	public String float2Str(float value){ 
		String temStr=Float.toString(value); 
		if(temStr.length()-temStr.indexOf('.')==2)
			temStr=temStr+"0";
		return temStr;
	}
	public static void main(String[] a) {
		ArrayList<Float> aa=new ArrayList<Float>();
		aa.add(0.5f);
		aa.add(0.1f);
		System.out.println(DataFormats.getInstance().percentile(aa,0.99f));
	}

}
