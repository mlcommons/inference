package scs.util.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.LineNumberReader;
import java.io.UnsupportedEncodingException; 
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale; 

public class FileOperation {
	private SimpleDateFormat dateFormat = new SimpleDateFormat("dd/MMM/yyyy:HH:mm:ss", Locale.ENGLISH);

	public int headFile(String filePath,int headnum) throws IOException {  

		File file = new File(filePath); 
		BufferedReader reader = null;
		try {
			FileInputStream fileInputStream = new FileInputStream(file);
			InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
			reader = new BufferedReader(inputStreamReader);
			String line = ""; 
			/*
			 * 跳过前1行
			 */ 
			int count=0;

			while ((line = reader.readLine()) != null) { 
				count++;
				if(count>headnum)
					break;
				else
					System.out.println(line);

			}

			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		return 0;
	}
	/**
	 * 单纯读取xapian的lats.txt文件
	 * @param filePath
	 * @return 返回list数组
	 * @throws IOException
	 */
	public List<Long> readLogFile(String filePath) throws IOException {  
		List<Long> timeList=new ArrayList<Long>();

		File file = new File(filePath); 
		BufferedReader reader = null;
		try {
			FileInputStream fileInputStream = new FileInputStream(file);
			InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
			reader = new BufferedReader(inputStreamReader);
			String line = ""; 
			/*
			 * 跳过前1行
			 */ 
			int start=0;
			while ((line = reader.readLine()) != null) { 
				start=line.indexOf("[")+1;

				try{
					timeList.add(dateFormat.parse(line.substring(start,start+20)).getTime());
				}catch(Exception e){
					continue;
				}
			}

			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return timeList;
	}
	public List<Double> readDoubleFile(String filePath) throws IOException {  
		List<Double> timeList=new ArrayList<Double>();

		File file = new File(filePath); 
		BufferedReader reader = null;
		try {
			FileInputStream fileInputStream = new FileInputStream(file);
			InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
			reader = new BufferedReader(inputStreamReader);
			String line = ""; 

			while ((line = reader.readLine()) != null) { 
				timeList.add(Double.parseDouble(line));
			}

			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return timeList;
	}
	public List<String> readStringFile(String filePath) throws IOException {  
		List<String> timeList=new ArrayList<String>();

		File file = new File(filePath); 
		BufferedReader reader = null;
		try {
			FileInputStream fileInputStream = new FileInputStream(file);
			InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
			reader = new BufferedReader(inputStreamReader);
			String line = ""; 

			while ((line = reader.readLine()) != null) { 
				timeList.add(line);
			}

			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return timeList;
	}
	public List<Integer> readIntFile(String filePath) throws IOException {  
		List<Integer> timeList=new ArrayList<Integer>();

		File file = new File(filePath); 
		BufferedReader reader = null;
		try {
			FileInputStream fileInputStream = new FileInputStream(file);
			InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
			reader = new BufferedReader(inputStreamReader);
			String line = ""; 

			while ((line = reader.readLine()) != null) { 
				timeList.add(Integer.parseInt(line));
			}

			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return timeList;
	}
	public int splitFile(String fileInputPath,int lineNumPerFile) throws IOException {  
		int totalLineNum=countLineNum(fileInputPath);
		int count=0;
		int fileIndex=0;
		FileWriter outFile=null;
		File file = new File(fileInputPath); 
		BufferedReader reader = null;
		try {
			FileInputStream fileInputStream = new FileInputStream(file);
			InputStreamReader inputStreamReader = new InputStreamReader(fileInputStream, "UTF-8");
			reader = new BufferedReader(inputStreamReader);
			String line = ""; 
			while ((line = reader.readLine()) != null) {
				count++;
				if(count%lineNumPerFile==1){
					if(outFile!=null){
						outFile.flush();
						outFile.close();
					}
					outFile=new FileWriter(fileInputPath.replace(".","_"+fileIndex+"."));
					fileIndex++;
					System.out.println("process:"+count*100.0/totalLineNum+"%");
				} 
				outFile.write(line+"\r\n");
			}
			if(outFile!=null){
				outFile.flush();
				outFile.close();
			}
			if (reader!=null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} finally {
			if (reader != null) {
				try {
					reader.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}

		return count;
	}
	public int countLineNum(String filePath){
		int linenumber = 0;
		try{
			File file =new File(filePath);
			if(file.exists()){
				FileReader fr = new FileReader(file);
				LineNumberReader lnr = new LineNumberReader(fr);
				lnr.skip(file.length());
				linenumber=lnr.getLineNumber();
				lnr.close();
			}else{
				System.out.println("File does not exists!");
			}
		}catch(IOException e){
			e.printStackTrace();
		}
		return linenumber;

	}
	/**
	 * 写入文件
	 * @param resList
	 * @param filePath
	 * @param type row:按行写 col:按列
	 */
	public <T> void writeResFile(List<T> resList,String filePath,String type){
		try{
			FileWriter writer = new FileWriter(filePath);
			int size=resList.size();
			if(type!=null&&type.equals("row")){
				for(int i=0;i<size;i++){
					writer.write(resList.get(i)+"\n");
					if(i%1000==0)
						writer.flush();
				}
			}else{
				for(int i=0;i<size;i++){
					writer.write(resList.get(i).toString());
					if(i%1000==0)
						writer.flush();
				}
			}

			writer.flush();
			writer.close();
			System.out.println("写入完毕");
		} catch (IOException e1) {
			e1.printStackTrace();
		} 
	}

}

