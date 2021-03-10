package adufs;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.supervised.attribute.AddClassification;

import com.google.common.base.Stopwatch;


public class ClassPredictor {
	public static String DIRECTORY = "D:\\feature-selection\\output\\anomaly-detection\\";
	public static String DATASET_NAME, classificationModel, CSV_GEN = "", RTF_DS = "";
    public static String COMMA_DELIMITER = ",", NEW_LINE_SEPARATOR = "\n";	
    public static String GEN_HEADER = "", GEN_RTF_HEADER = "", DS_RTF_HEADER = "";
    public static FileWriter genWriter, dsRTFWriter;
    
    public static SimpleDateFormat formatter; 
	public static Date startDate, endDate;	
	public static Stopwatch timer;
    
	public static String[] classValue, actual, predString;	
	public static double TP, TN, FP, FN, TPR, TNR, FPR, FNR;
	
	public static void main(String[] args) throws Exception{
		DataSource source = new DataSource("D:\\feature-selection\\security-dataset\\UNSW_NB15_Two_Classes_Training.arff");
		DataSource source2 = new DataSource("D:\\feature-selection\\security-dataset\\UNSW_NB15_No_Class_Test.arff");
		DataSource source3 = new DataSource("D:\\feature-selection\\security-dataset\\UNSW_NB15_Anomaly_Class_Test.arff");
		DataSource source4 = new DataSource("D:\\feature-selection\\security-dataset\\UNSW_NB15_Two_Classes_Test.arff");
		DATASET_NAME = "UNSW_NB15";	classificationModel = "NaiveBayes";	
		formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");	startDate = new Date();
        defineFileNamesWithDateTime();
		try {createAndInitiateFileWriters();
	        System.out.println("\nClassification prediction starts exceution at: " + formatter.format(startDate) + "\n");
	        dsRTFWriter.append("Classification prediction starts exceution at: " + formatter.format(startDate) + "\n\n");
			timer = Stopwatch.createUnstarted();	timer.start();		endDate = new Date();
			System.out.println("Dataset Name: " + DATASET_NAME + "\tClassifier Used: " + classificationModel);
			dsRTFWriter.append("Dataset Name: " + DATASET_NAME + "\tClassifier Used: " + classificationModel);
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			Instances traindata = source.getDataSet();	traindata.setClassIndex(traindata.numAttributes()-1);
			int numClasses = traindata.numClasses();
			System.out.println("\nNumber of Classes in the Dataset: " + numClasses + "\n");
			dsRTFWriter.append("\nNumber of Classes in the Dataset: " + numClasses);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			classValue = new String[numClasses];
			for (int x = 0; x < numClasses; x++){
				classValue[x] = traindata.classAttribute().value(x);
				System.out.println("The " + x + "th class value: " + classValue[x]);
				dsRTFWriter.append("The " + x + "th class value: " + classValue[x]);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			}
			NaiveBayes nb = new NaiveBayes();	
			nb.buildClassifier(traindata);
			
			Instances unlabeled = source2.getDataSet();
			unlabeled.setClassIndex(unlabeled.numAttributes()-1);
			
			int correct = 0, inCorrect = 0;
			double pctCorrect, pctIncorrect;
		
			int[] classCorrect = null, classIncorrect = null;
			double[] pctClassCorrect, pctClassIncorrect;
		
			int numTestInstances = unlabeled.numInstances();
			System.out.println("\nNumber of Test Instances: " + numTestInstances);
			dsRTFWriter.append("\nNumber of Test Instances: " + numTestInstances);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			
			Instances anomalydata = source3.getDataSet();
			anomalydata.setClassIndex(anomalydata.numAttributes()-1);
			int anomalyInstances;
			anomalyInstances = anomalydata.numInstances();
			System.out.println("\nActual Number of anomalies in the test dataset: " + anomalyInstances);
			dsRTFWriter.append("\nActual Number of anomalies in the test dataset: " + anomalyInstances);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
		
			Instances labeled = new Instances(unlabeled);
		
			actual = new String[numTestInstances];
			predString = new String[numTestInstances];
			
			Instances testdata = source4.getDataSet();
			testdata.setClassIndex(testdata.numAttributes()-1);
		
			for (int j = 0; j < numTestInstances; j++){
				double clsLabel = testdata.instance(j).classValue();
				actual[j] = testdata.classAttribute().value((int) clsLabel);
				genWriter.append(String.valueOf((j+1)));						genWriter.append(COMMA_DELIMITER);
				genWriter.append(String.valueOf((unlabeled.instance(j).stringValue(unlabeled.numAttributes()-1))));						genWriter.append(COMMA_DELIMITER);
				double preNB = nb.classifyInstance(unlabeled.instance(j));
				labeled.instance(j).setClassValue(preNB);
				predString[j] = labeled.instance(j).stringValue(labeled.numAttributes()-1);
				genWriter.append(String.valueOf((actual[j])));						genWriter.append(COMMA_DELIMITER);
				genWriter.append(String.valueOf((predString[j])));						genWriter.append(NEW_LINE_SEPARATOR);
			
				if(actual[j].compareTo(predString[j]) == 0)
					correct++;
				else
					inCorrect++;
			}
			
			String testClass;
			classCorrect = new int[numTestInstances];
			classIncorrect = new int[numTestInstances];
			
			for (int i = 0; i < numClasses; i++){
				testClass = classValue[i];
				for (int j = 0; j < numTestInstances; j++){
					if((testClass.compareTo(actual[j]) == 0) && (actual[j].compareTo(predString[j]) == 0)) {
						classCorrect[i]++;
					}
					else if((testClass.compareTo(actual[j]) == 0) && (actual[j].compareTo(predString[j]) != 0)) {
						classIncorrect[i]++;
					}
					else{
						// Do nothing...
					}
				}
			}
			
			DecimalFormat df = new DecimalFormat("###.####");
		
			pctClassCorrect = new double[numClasses];
			pctClassIncorrect = new double[numClasses];
			for (int i = 0; i < numClasses; i++){
				pctClassCorrect[i] = (double) classCorrect[i] / numTestInstances * 100;
				pctClassIncorrect[i] = (double) classIncorrect[i] / numTestInstances * 100;
				System.out.println("\nClass: "+ classValue[i] + ", Correctly classified:\t" + classCorrect[i] + " (" + pctClassCorrect[i] + " or " + df.format(pctClassCorrect[i]) + ")" + "\tIncorrectly classified:\t\t\t" + classIncorrect[i] + " (" + pctClassIncorrect[i] + " or " + df.format(pctClassIncorrect[i]) + ")");
				dsRTFWriter.append("\nClass: "+ classValue[i] + ", Correctly classified:\t" + classCorrect[i] + " (" + pctClassCorrect[i] + " or " + df.format(pctClassCorrect[i]) + ")" + "\tIncorrectly classified:\t\t\t\t" + classIncorrect[i] + " (" + pctClassIncorrect[i] + " or " + df.format(pctClassIncorrect[i]) + ")");
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
			}
		
		
			pctCorrect = (double) correct / numTestInstances * 100;
			pctIncorrect = (double) inCorrect / numTestInstances * 100;
			System.out.println("\nAll classes Correctly classified:\t" + correct + " (" + pctCorrect + " or " + df.format(pctCorrect) + ")" + "\tAll classes Incorrectly classified:\t" + inCorrect + " (" + pctIncorrect + " or " + df.format(pctIncorrect) + ")");
			dsRTFWriter.append("\nAll classes Correctly classified:\t\t" + correct + " (" + pctCorrect + " or " + df.format(pctCorrect) + ")" + "\tAll classes Incorrectly classified:\t" + inCorrect + " (" + pctIncorrect + " or " + df.format(pctIncorrect) + ")");			
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
		
			TP = classCorrect[0];	FN = classIncorrect[0];
			TN = classCorrect[1];	FP = classIncorrect[1];
			
			TPR = (TP / (TP + FN));
			FPR = (FP / (FP + TN));
			TNR = (TN / (FP + TN));
			FNR = (FN / (TP + FN));
			
			System.out.println();
			System.out.println("TP:\t" + TP + "\t\tFP:\t" + FP + "\t\tTN:\t" + TN + "\t\tFN:\t" + FN);
			System.out.println("TPR:\t" + df.format(TPR) + "\t\tFPR:\t" + df.format(FPR) + "\t\tTNR:\t" + df.format(TNR) + "\t\tFNR:\t" + df.format(FNR));
			
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append("TP:\t\t" + TP + "\t\tFP:\t\t" + FP + "\t\tTN:\t\t" + TN + "\t\tFN:\t\t" + FN);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append("TPR:\t" + df.format(TPR) + "\t\tFPR:\t" + df.format(FPR) + "\t\tTNR:\t" + df.format(TNR) + "\t\tFNR:\t" + df.format(FNR));	
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			
			double TPR_Anomalies;
			
			TPR_Anomalies = TN/anomalyInstances;
			
			System.out.println("\nCorrectly classified anomalies = " + TN + "\tTotal number of test anomaly instances = " + anomalyInstances);
			System.out.println("\nAnomaly detection rate by TPR = " + df.format(TPR_Anomalies));
			dsRTFWriter.append("\nCorrectly classified anomalies = " + TN + "\tTotal number of test anomaly instances = " + anomalyInstances);
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append("\nAnomaly detection rate by TPR = " + df.format(TPR_Anomalies));	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			
			endDate = new Date();

			System.out.println("\nCSV and RTF files created successfully!!! at \n\n" + CSV_GEN + "\n" + RTF_DS + "\n");
			dsRTFWriter.append("\nCSV and RTF files created successfully!!! at \n\n" + CSV_GEN + "\n" + RTF_DS + "\n");
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			
			System.out.println("CCFSRFG Anomaly Detection stops at: " + formatter.format(endDate));
			System.out.println("CCFSRFG Anomaly Detection execution completed successfully!!!");
			System.out.println("\n====== Total Time Elapsed for the Execution: " + String.valueOf(timer) + "\n");
			dsRTFWriter.append("CCFSRFG Anomaly Detection stops at: " + formatter.format(endDate));	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append("CCFSRFG Anomaly Detection execution completed successfully!!!");	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append("\n====== Total Time Elapsed for the Execution: " + String.valueOf(timer) + "\n");	dsRTFWriter.append(NEW_LINE_SEPARATOR);
		}	// end of try 	
		catch (Exception e) {
			System.out.println("Error in CSV and RTF FileWriters !!!");
        	e.printStackTrace();
    	}	// end of catch 	
		finally {
    			try {
    				genWriter.flush();		genWriter.close();		
    				dsRTFWriter.flush();	dsRTFWriter.close();
    			} 	catch (IOException e) {
    					System.out.println("Error while flushing/closing Writers !!!");
    					e.printStackTrace();
    				}	// End of 2nd catch block
    	}	// End of finally

	}	// End of main() function
	
	public static void defineFileNamesWithDateTime() {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_kkmmss");	// Time in 24 hours format -> use kk for hour; hh for 12 hours format
        Date curDate = new Date();
        String strDate = sdf.format(curDate);
        CSV_GEN = DIRECTORY + "CCFSRFG_1_" + DATASET_NAME + "_" + classificationModel + "_gen_" + strDate + ".csv";
        RTF_DS = DIRECTORY + "CCFSRFG_1_" + DATASET_NAME + "_"  + classificationModel + "_DS_" + strDate + ".rtf";
	}
	
	public static void createAndInitiateFileWriters() throws IOException {
        genWriter = new FileWriter(CSV_GEN);
        GEN_HEADER += "Instance," + "Class to predict," + "Actual," + "Predicted"; 
        dsRTFWriter = new FileWriter(RTF_DS);
        DS_RTF_HEADER += "";
        //Write the CSV file header and Add a new line separator after the header
        genWriter.append(GEN_HEADER.toString());	genWriter.append(NEW_LINE_SEPARATOR);
        dsRTFWriter.append(DS_RTF_HEADER.toString());	dsRTFWriter.append(NEW_LINE_SEPARATOR);
	}	

}
