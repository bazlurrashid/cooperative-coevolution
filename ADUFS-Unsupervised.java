package adufs;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import org.apache.commons.lang.ArrayUtils;

import com.google.common.base.Stopwatch;

public class CSVFileComparator {
	
	public static String DATASET_NAME, detectionTechnique, RTF_DS = "";	public static String COMMA_DELIMITER = ",", NEW_LINE_SEPARATOR = "\n";	
    public static String DS_RTF_HEADER = "";	public static FileWriter dsRTFWriter;
    public static SimpleDateFormat formatter; public static Date startDate, endDate;	public static Stopwatch timer;	public static double TPR;
    public static String DIRECTORY = "D:\\feature-selection\\output\\anomaly-detection\\unsupervised\\";
	
	public static void main(String[] args) throws IOException {
		formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");	startDate = new Date();
		String path="D:\\feature-selection\\security-dataset\\anomaly_detection\\";
		
	    String file2="Windows7_Anomaly_Class.csv";
	    String file1="Windows7_k-NN_Anomaly_Detection.csv";	DATASET_NAME = "Windows7";	detectionTechnique = "k-NN";

	    ArrayList<String> al1=new ArrayList<String>();
	    ArrayList<String> al2=new ArrayList<String>();
	    
	    defineFileNamesWithDateTime();
	    try {
			createAndInitiateFileWriters();
	        System.out.println("\nPrediction crossmatching starts exceution at: " + formatter.format(startDate) + "\n");
	        dsRTFWriter.append("Prediction crossmatching starts exceution at: " + formatter.format(startDate) + "\n\n");
			timer = Stopwatch.createUnstarted();	timer.start();		endDate = new Date();
			System.out.println("Dataset Name: " + DATASET_NAME + "\tAnomaly detection technique Used: " + detectionTechnique);
			dsRTFWriter.append("Dataset Name: " + DATASET_NAME + "\tAnomaly detection technique Used: " + detectionTechnique);
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
	    
	    BufferedReader CSVFile1 = new BufferedReader(new FileReader(path+file1));
	    String dataRow1 = CSVFile1.readLine();
	    while ((dataRow1 = CSVFile1.readLine()) != null) {
	    	String[] data = dataRow1.split(",");
	    	al1.add(data[0]);
//	        al1.add(dataRow1);
	    }

	    CSVFile1.close();

	    BufferedReader CSVFile2 = new BufferedReader(new FileReader(path+file2));
	    String dataRow2 = CSVFile2.readLine();
	    while ((dataRow2 = CSVFile2.readLine()) != null) {
	    	String[] data = dataRow2.split(",");
	    	al2.add(data[0]);
//	    	al2.add(dataRow2);
	    }
	     CSVFile2.close();
	     
	     int correct = 0, inCorrect = 0;
	     int[] track = new int[al1.size()];
	     int tc = 0;
	     
	     int[] nums = java.util.stream.IntStream.rangeClosed(0, al2.size()-1).toArray();
	     
	     for(int x = 0; x < al1.size(); x++) {
	    	 label: for(int y = 0; y < nums.length; y++) {
	    		 if(al1.get(x).equals(al2.get(nums[y]))) {
	    			 correct++;
	    			 nums = ArrayUtils.remove(nums, y);
	    			 break label;
	    		 }
	    	 }
    		 System.out.print(x + ", ");
	    	 if(x > 0 && x % 50 == 0)
	    		 System.out.println();
	     }
	     
	     System.out.println();
	     
	     System.out.println("\nCorrect: " + correct);
	     
	     int anomalies;
	     
	     anomalies = al2.size();
	     
	     DecimalFormat df = new DecimalFormat("###.####");
	     
	     inCorrect = al2.size() - correct;
	     
	     TPR = (double) correct/anomalies;
	     
	     System.out.println("\nActual anomalies = " + al2.size() + "\tCorresponding Outliers = " + al1.size());
	     System.out.println("\nCorrectly classified anomalies = " + correct + "\tIncorrectly classified anomalies = " + inCorrect + "\tTPR = " + df.format(TPR));
	     dsRTFWriter.append("\nActual anomalies = " + al2.size() + "\tCorresponding Outliers = " + al1.size());
	     dsRTFWriter.append(NEW_LINE_SEPARATOR);
	     dsRTFWriter.append("\nCorrectly classified anomalies = " + correct + "\tIncorrectly classified anomalies = " + inCorrect + "\tTPR = " + df.format(TPR));
	     dsRTFWriter.append(NEW_LINE_SEPARATOR);
	     
	     endDate = new Date();

			System.out.println("\nRTF file created successfully!!! at \n\n" + RTF_DS + "\n");
			dsRTFWriter.append("\nRTF file created successfully!!! at \n\n" + RTF_DS + "\n");
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			
			System.out.println("Anomaly Detection stops at: " + formatter.format(endDate));
			System.out.println("Anomaly Detection execution completed successfully!!!");
			System.out.println("\n====== Total Time Elapsed for the Execution: " + String.valueOf(timer) + "\n");
			dsRTFWriter.append("Anomaly Detection stops at: " + formatter.format(endDate));	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append("Anomaly Detection execution completed successfully!!!");	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append("\n====== Total Time Elapsed for the Execution: " + String.valueOf(timer) + "\n");	dsRTFWriter.append(NEW_LINE_SEPARATOR);
		}	// end of try 	
		catch (Exception e) {
			System.out.println("Error in RTF FileWriter !!!");
     	e.printStackTrace();
 	}	// end of catch 	
		finally {
 			try {
 				dsRTFWriter.flush();	dsRTFWriter.close();
 			} 	catch (IOException e) {
 					System.out.println("Error while flushing/closing Writer !!!");
 					e.printStackTrace();
 				}	// End of 2nd catch block
 	}	// End of finally
	     
	}	// End of main() function
	
	public static void defineFileNamesWithDateTime() {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_kkmmss");	// Time in 24 hours format -> use kk for hour; hh for 12 hours format
        Date curDate = new Date();
        String strDate = sdf.format(curDate);
        RTF_DS = DIRECTORY + DATASET_NAME + "_"  + detectionTechnique + "_DS_" + strDate + ".rtf";
	}
	
	public static void createAndInitiateFileWriters() throws IOException {
        dsRTFWriter = new FileWriter(RTF_DS);
        DS_RTF_HEADER += "";
        //Write the CSV file header and Add a new line separator after the header
        dsRTFWriter.append(DS_RTF_HEADER.toString());	dsRTFWriter.append(NEW_LINE_SEPARATOR);
	}	

}
