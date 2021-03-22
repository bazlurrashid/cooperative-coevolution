package urad;

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
	
	public static String DATASET_NAME, detectionTechnique, RTF_DS = "";	
	public static String COMMA_DELIMITER = ",", NEW_LINE_SEPARATOR = "\n";	
    public static String DS_RTF_HEADER = "";	
	public static FileWriter dsRTFWriter;
    public static SimpleDateFormat formatter; 
	public static Date startDate, endDate;	
	public static Stopwatch timer;	
	public static double TPR;
    public static String DIRECTORY = "D:\\feature-selection\\output\\rare_anomaly_detection\\";	// Directory where the output file will be written
	
	public static void main(String[] args) throws IOException {
		formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");	
		startDate = new Date();
		String path="D:\\feature-selection\\security-dataset\\rare_anomaly_detection\\";	// Directory from where the CSV file of detected outlier files will be read
	    String file2="UNSW_NB15_Anomaly_Class.csv";		// CSV file that contains only the anomalous/infrequent patterns with class labels
		//Below is the CSV file containing the total number of outliers detected by RapidMiner equivalent to the total number of anomalous patterns in the original dataset; Use the corresponding technique name for writing the file name with that technique
	    String file1="UNSW_NB15_k-NN_Anomaly_Detection.csv";	DATASET_NAME = "UNSW_NB15";	detectionTechnique = "k-NN";	

	    ArrayList<String> al1=new ArrayList<String>();
	    ArrayList<String> al2=new ArrayList<String>();
	    
	    defineFileNamesWithDateTime();
	    try {
			createAndInitiateFileWriters();
	        System.out.println("\nPrediction crossmatching starts exceution at: " + formatter.format(startDate) + "\n");
	        dsRTFWriter.append("Prediction crossmatching starts exceution at: " + formatter.format(startDate) + "\n\n");
			timer = Stopwatch.createUnstarted();	
			timer.start();		
			endDate = new Date();
			System.out.println("Dataset Name: " + DATASET_NAME + "\tAnomaly detection technique Used: " + detectionTechnique);
			dsRTFWriter.append("Dataset Name: " + DATASET_NAME + "\tAnomaly detection technique Used: " + detectionTechnique);
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
	    
	    BufferedReader CSVFile1 = new BufferedReader(new FileReader(path+file1));
	    String dataRow1 = CSVFile1.readLine();
	    while ((dataRow1 = CSVFile1.readLine()) != null) {
	    	String[] data = dataRow1.split(",");
	    	al1.add(data[0]);
	    }

	    CSVFile1.close();

	    BufferedReader CSVFile2 = new BufferedReader(new FileReader(path+file2));
	    String dataRow2 = CSVFile2.readLine();
	    while ((dataRow2 = CSVFile2.readLine()) != null) {
	    	String[] data = dataRow2.split(",");
	    	al2.add(data[0]);
	    }
	     CSVFile2.close();
	     
	     int correct = 0, inCorrect = 0;
	     int[] rareCorrect = new int[al1.size()]; 
	     int[] rareIncorrect = new int[al1.size()];
	     int[] track = new int[al1.size()];
	     int tc = 0;
	     
	     int[] nums = java.util.stream.IntStream.rangeClosed(0, al2.size()-1).toArray();
	     
	     for(int x = 0; x < al1.size(); x++) {
	    	 label: for(int y = 0; y < nums.length; y++) {
	    		 if(al1.get(x).equals(al2.get(nums[y]))) {
	    			 correct++;
					 // The numbers specified below are the corresponding row numbers of anomalous patterns in the dataset.
					 // Other than writing manually this can also be fetched from the dataset and use accordingly
	    			 if(Integer.valueOf(al1.get(x)) >= 37001 && Integer.valueOf(al1.get(x)) <= 37677) {
	    				 rareCorrect[0]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 37678 && Integer.valueOf(al1.get(x)) <= 38260) {
	    				 rareCorrect[1]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 38261 && Integer.valueOf(al1.get(x)) <= 42349) {
	    				 rareCorrect[2]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 42350 && Integer.valueOf(al1.get(x)) <= 53481) {
	    				 rareCorrect[3]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 53482 && Integer.valueOf(al1.get(x)) <= 59543) {
	    				 rareCorrect[4]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 59544 && Integer.valueOf(al1.get(x)) <= 78414) {
	    				 rareCorrect[5]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 78415 && Integer.valueOf(al1.get(x)) <= 81910) {
	    				 rareCorrect[6]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 81911 && Integer.valueOf(al1.get(x)) <= 82288) {
	    				 rareCorrect[7]++;
	    			 }
	    			 else if(Integer.valueOf(al1.get(x)) >= 82289 && Integer.valueOf(al1.get(x)) <= 82332) {
	    				 rareCorrect[8]++;
	    			 }
	    			 else {
	    				 // Do nothing ...
	    			 }
	    			 System.out.print(al1.get(x) + "\t");
	    			 nums = ArrayUtils.remove(nums, y);
	    			 break label;
	    		 }
	    	 }
	     }
	     
	     System.out.println();
	     
	     DecimalFormat df = new DecimalFormat("###.####");
	     
	     System.out.println("\nCorrect: " + correct 
				// The numbers specified below are the corresponding total numbers of each anomalous pattern in the dataset.
				// Other than writing manually this can also be fetched from the dataset and use accordingly
	    		 + "\nAnalysis: " + rareCorrect[0] + ": TPR: " + (double) rareCorrect[0]/677 + " OR " + df.format((double) rareCorrect[0]/677)
	    		 + "\nBackdoor: " + rareCorrect[1] + ": TPR: " + (double) rareCorrect[1]/583 + " OR " + df.format((double) rareCorrect[1]/583)
	    		 + "\nDoS: " + rareCorrect[2] + ": TPR: " + (double) rareCorrect[2]/4089 + " OR " + df.format((double) rareCorrect[2]/4089)
	    		 + "\nExploits: " + rareCorrect[3] + ": TPR: " + (double) rareCorrect[3]/11132 + " OR " + df.format((double) rareCorrect[3]/11132)
	    		 + "\nFuzzers: " + rareCorrect[4] + ": TPR: " + (double) rareCorrect[4]/6062 + " OR " + df.format((double) rareCorrect[4]/6062)
	    		 + "\nGeneric: " + rareCorrect[5] + ": TPR: " + (double) rareCorrect[5]/18871 + " OR " + df.format((double) rareCorrect[5]/18871)
	    		 + "\nReconnaissance: " + rareCorrect[6] + ": TPR: " + (double) rareCorrect[6]/3496 + " OR " + df.format((double) rareCorrect[6]/3496)
	    		 + "\nShellcode: " + rareCorrect[7] + ": TPR: " + (double) rareCorrect[7]/378 + " OR " + df.format((double) rareCorrect[7]/378)
	    		 + "\nWorms: " + rareCorrect[8] + ": TPR: " + (double) rareCorrect[8]/44 + " OR " + df.format((double) rareCorrect[8]/44));
	     
	     int anomalies;
	     
	     anomalies = al2.size();
	     
	     inCorrect = al2.size() - correct;
	     
	     TPR = (double) correct/anomalies;
	     
	     System.out.println("\nActual anomalies = " + al2.size() + "\tCorresponding Outliers = " + al1.size());
	     System.out.println("\nCorrectly classified anomalies = " + correct + "\tIncorrectly classified anomalies = " + inCorrect + "\tTPR = " + df.format(TPR));
	     dsRTFWriter.append("\nActual anomalies = " + al2.size() + "\tCorresponding Outliers = " + al1.size());
	     dsRTFWriter.append(NEW_LINE_SEPARATOR);
	     dsRTFWriter.append("\nCorrectly classified anomalies = " + correct + "\tIncorrectly classified anomalies = " + inCorrect + "\tTPR = " + df.format(TPR));
	     dsRTFWriter.append(NEW_LINE_SEPARATOR);
	     
	     dsRTFWriter.append("\nCorrect: " + correct 
				// The numbers specified below are the corresponding total numbers of each anomalous pattern in the dataset.
				// Other than writing manually this can also be fetched from the dataset and use accordingly
	    		 + "\nAnalysis: " + rareCorrect[0] + ": TPR: " + (double) rareCorrect[0]/677 + " OR " + df.format((double) rareCorrect[0]/677)
	    		 + "\nBackdoor: " + rareCorrect[1] + ": TPR: " + (double) rareCorrect[1]/583 + " OR " + df.format((double) rareCorrect[1]/583)
	    		 + "\nDoS: " + rareCorrect[2] + ": TPR: " + (double) rareCorrect[2]/4089 + " OR " + df.format((double) rareCorrect[2]/4089)
	    		 + "\nExploits: " + rareCorrect[3] + ": TPR: " + (double) rareCorrect[3]/11132 + " OR " + df.format((double) rareCorrect[3]/11132)
	    		 + "\nFuzzers: " + rareCorrect[4] + ": TPR: " + (double) rareCorrect[4]/6062 + " OR " + df.format((double) rareCorrect[4]/6062)
	    		 + "\nGeneric: " + rareCorrect[5] + ": TPR: " + (double) rareCorrect[5]/18871 + " OR " + df.format((double) rareCorrect[5]/18871)
	    		 + "\nReconnaissance: " + rareCorrect[6] + ": TPR: " + (double) rareCorrect[6]/3496 + " OR " + df.format((double) rareCorrect[6]/3496)
	    		 + "\nShellcode: " + rareCorrect[7] + ": TPR: " + (double) rareCorrect[7]/378 + " OR " + df.format((double) rareCorrect[7]/378)
	    		 + "\nWorms: " + rareCorrect[8] + ": TPR: " + (double) rareCorrect[8]/44 + " OR " + df.format((double) rareCorrect[8]/44));
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
