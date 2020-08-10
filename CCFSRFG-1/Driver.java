package ccfsrfg1;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import javax.swing.JTextArea;

import com.google.common.base.Stopwatch;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Driver{	// start of the Driver class

	public static final int NO_OF_GENERATION = 10000, NO_OF_GENES = 10, SUBPOP_SIZE = 30;	// Specify here the maximum number of generation, number of features (i.e., NO_OF_GENES) in each subpopulation, and size of each subpopulation of a dataset
	public static int[] sizeofEachSubPop;
	public static final int NUMB_OF_ELITE_CHROMOSOMES = 1, TOURNAMENT_SELECTION_SIZE = 1;
	public static final Double CROSSOVER_RATE = 1.00, MUTATION_RATE = 0.05, weight1 = 0.60, weight2 = 0.40;
	public static int SOLUTION_SIZE, NO_OF_SUBPOPULATION;	
	public static String TRAIN_DATASET, TEST_DATASET, TRAIN_DATASET_NAME, TEST_DATASET_NAME;
	public static int NUMBER_OF_TRAIN_SAMPLES, NUMBER_OF_TEST_SAMPLES;
	public static String MODEL_SAVE = "C:\\Users\\Bazlur\\Desktop\\feature-selection\\output\\CCFSRFG-1\\trained.model";
	public static String DIRECTORY = "C:\\Users\\Bazlur\\Desktop\\feature-selection\\output\\CCFSRFG-1\\";
	public static String CSV_GEN = "", CSV_SUB = "", CSV_IND = "", RTF_GEN = "", RTF_SUB = "", RTF_IND = "", RTF_DS = "";
	
	public static DataSource trainSource, testSource;
	public static Instances trainData, testData;	
	
    public static String COMMA_DELIMITER = ",", NEW_LINE_SEPARATOR = "\n";	
    public static String GEN_HEADER = "generation,", SUB_HEADER = "generation,", IND_HEADER = "generation,";
    public static String GEN_RTF_HEADER = "", SUB_RTF_HEADER = "", IND_RTF_HEADER = "", DS_RTF_HEADER = "";
    public static FileWriter genWriter, subWriter, indWriter, genRTFWriter, subRTFWriter, indRTFWriter, dsRTFWriter;
    
    public static Double accuracy, precision, recall, f1Score, sensitivity, specificity, correctlyClassified;
    public static String strSummary, strClassDetails, strConfusionMatrix;
	private static Double weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score;
	private static Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
    
    public static JTextArea jTextArea;
    public static String classificationMode, classificationModel;
    
    public static Double[] fitnessStagnation;
    public static int stagnationCount = 0, terminatedGenerationNumber;
    
    public static Stopwatch timer;
    
    public static GeneticAlgorithm ga;
    public static SubPopulation[] subPopulation;
    public static ContextVector contextVector;
    public static Individual[] bestIndvidual;
    public static SortBestIndividualesByFitness sbi;
    
    public static RandomGrouping RG;
	public static List<Integer> attributIndices;
	public static List<List<Integer> > groupedIndices;
    
    public static int generationNumber;
    
    public static SimpleDateFormat formatter; 
    public static Date startDate, endDate;
	
    public Driver() {
    }

    public static void main(String[] args) throws Exception {	// Start of main() function
    	
    	formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");  
    	startDate = new Date();
        callGUIForDatasetInfo();
   	
        jTextArea.append("\nCCFSRFG-1 starts exceution at: " + formatter.format(startDate) + "\n");
		attributIndices = new ArrayList<>();
    	RG = new RandomGrouping();
		NO_OF_SUBPOPULATION = RG.getNumberofGroups();
		
		sizeofEachSubPop = RG.getSizeofEachGroup();
		
		System.out.println("sizeofEachSubPop: " + Arrays.toString(sizeofEachSubPop));
		
    	fitnessStagnation = new Double[NO_OF_GENERATION];
		ga = new GeneticAlgorithm();
		subPopulation = new SubPopulation[NO_OF_SUBPOPULATION];
		contextVector = new ContextVector();
		bestIndvidual = new Individual[NO_OF_SUBPOPULATION];
		sbi = new SortBestIndividualesByFitness(bestIndvidual);
		
		evaluateWithAllFeatures();
		defineFileNamesWithDateTime();
        
		try {
			
			createAndInitiateFileWriters();

			jTextArea.append("\n");
			
			timer = Stopwatch.createUnstarted();	// a timer used from Google Guava package
	    	timer.start();	// counting the start time of the main execution
	    	
			generationNumber = 0;	// starting at generation 0
			System.out.println("gen: " + generationNumber + " :: ");
			initializeSubPopulations();
			contextVector.findRandomCollaborators(subPopulation);	
			calculateFitnessForGeneration0();
            findBestIndividualsFromGeneration0();
            
			generationNumber = 1;
			while(generationNumber < NO_OF_GENERATION) {
				System.out.println("gen: " + generationNumber + " :: ");
				evolveSubPopulations();
				contextVector.findBestCollaborators(subPopulation);	
				calculateFitnessForGenerations();
				findBestIndividualsFromGenerations();

				if(stagnationCount >= 100) {
					terminatedGenerationNumber = generationNumber;
					timer.stop();	
					break;
				}
				
				generationNumber++;	// incrementing the generation number
			}	// End of while loop checking number of generations condition
			
			endDate = new Date();
			printDSBestIndividual(bestIndvidual[0]);	// displaying dataset best result found from all generation
			writeDSBestIndividual(bestIndvidual[0]);	// writing dataset best result found from all generation

			jTextArea.append("\nCSV and RTF files created successfully!!! at \n\n" + CSV_GEN + "\n" + RTF_DS + "\n");
		}	// end of try 	
		catch (Exception e) {
				jTextArea.append("Error in CSV and RTF FileWriters !!!");
	        	e.printStackTrace();
	    	}	// end of catch 	
			finally {
	    			try {
	    				genWriter.flush();		
	    				genWriter.close();		
	    				dsRTFWriter.flush();	dsRTFWriter.close();
	    			} 	catch (IOException e) {
	    					jTextArea.append("Error while flushing/closing Writers !!!");
	    					e.printStackTrace();
	    				}	// End of 2nd catch block
	    		}	// End of finally

		jTextArea.append("\nCCFSRFG-1 stops at: " + formatter.format(endDate) + "\n");
		jTextArea.append("\nCCFSRFG-1 execution completed successfully!!!");
	}	// End of main() function
	
	public static void findBestIndividualsFromGenerations() throws IOException {
		for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {
			bestIndvidual[x] = subPopulation[x].getIndividuals()[0];	// finding best individuals from all subpopulations
		}
		
		bestIndvidual = sbi.sortIndividuales();	// sorting all best individuals found from subpopulations based on objective function
		printGenBestIndividual(bestIndvidual[0], generationNumber);	// displaying generation-wise best individual found from all subpopulation
				writeGenBestIndividual(bestIndvidual[0], generationNumber);	// writing generation-wise best individual found from all subpopulation
		
		fitnessStagnation[generationNumber] = bestIndvidual[0].getFitness();
		if(Double.compare(fitnessStagnation[generationNumber], fitnessStagnation[generationNumber-1]) == 0) {
			stagnationCount++;
		}
		if(Double.compare(fitnessStagnation[generationNumber], fitnessStagnation[generationNumber-1]) != 0) {
			stagnationCount  = 0;
		}
	}

	public static void calculateFitnessForGenerations() throws Exception {
		for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {
			for(int y = 0; y < SUBPOP_SIZE; y++) {
				subPopulation[x].getIndividuals()[y].setFitness();	// Calculating fitness of each individual from all subpopulations
		}
			
		subPopulation[x].sortIndividualesByFitness();	// sorting all individuals in each subpopulation based on objective function 
		}
	}

	public static void evolveSubPopulations() throws Exception {
		System.out.println();
		for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {
			subPopulation[x] = ga.evolve(subPopulation[x], sizeofEachSubPop[x]);	// evolving each subpopulation using GA
		}
	}

	public static void findBestIndividualsFromGeneration0() throws IOException {
		for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {	
			bestIndvidual[x] = subPopulation[x].getIndividuals()[0];	// finding best individuals from all subpopulations
		}
	
		bestIndvidual = sbi.sortIndividuales();	// sorting all best individuals found from subpopulations based on objective function
		printGenBestIndividual(bestIndvidual[0], generationNumber);	// displaying generation-wise best individual found from all subpopulation
				writeGenBestIndividual(bestIndvidual[0], generationNumber);	// writing generation-wise best individual found from all subpopulation
		
		fitnessStagnation[generationNumber] = bestIndvidual[0].getFitness();
	}

	public static void calculateFitnessForGeneration0() throws Exception {

		for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {	
			for(int y = 0; y < SUBPOP_SIZE; y++) {
				subPopulation[x].getIndividuals()[y].setFitness();	// calculating fitness of each individual from all subpopulations
			}
			subPopulation[x].sortIndividualesByFitness();	// sorting all individuals in each subpopulation based on objective function 
		}
	}

	public static void initializeSubPopulations() throws Exception {

		RG.randomizeAttributeIndices();
		for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {	
			subPopulation[x] = new SubPopulation(SUBPOP_SIZE).initializeSubPopulation(x, sizeofEachSubPop[x]);	// initializing individuals of each subpopulation using random bits
		}
	}

	public static void createAndInitiateFileWriters() throws IOException {
        genWriter = new FileWriter(CSV_GEN);
        GEN_HEADER += "precision," + "recall," + "f1-score," + "sensitivity," + "specificity," + "fitness," + 
        		"correctlyClassified," + "reducedFeatures," + "accuracy," + "originalFeatures," + "originalAccuracy," + 
        		"microPrecision," + "microRecall," + "microF1Score," + "macroPrecision," + "macroRecall," + "macroF1Score," +
        		"weightedPrecision," + "weightedRecall," + "weightedF1Score," + "matthewsCorrelationCoefficient," + 
        		"areaUnderROC," + "areaUnderPRC," + "weightedMatthewsCorrelation," + "weightedAreaUnderROC," + "weightedAreaUnderPRC"; 
        dsRTFWriter = new FileWriter(RTF_DS);
        DS_RTF_HEADER += "";
        genWriter.append(GEN_HEADER.toString());	genWriter.append(NEW_LINE_SEPARATOR);
        dsRTFWriter.append(DS_RTF_HEADER.toString());	dsRTFWriter.append(NEW_LINE_SEPARATOR);
	}

	public static void defineFileNamesWithDateTime() {
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_kkmmss");	// Time in 24 hours format -> use kk for hour; hh for 12 hours format
        Date curDate = new Date();
        String strDate = sdf.format(curDate);
        CSV_GEN = DIRECTORY + "CCFSRFG_1_" + TRAIN_DATASET_NAME + "_" + classificationModel + "_gen_" + strDate + ".csv";
        RTF_DS = DIRECTORY + "CCFSRFG_1_" + TRAIN_DATASET_NAME + "_"  + classificationModel + "_DS_" + strDate + ".rtf";
	}
	public static void callGUIForDatasetInfo() throws Exception {
		CCEAFS cceafs = new CCEAFS();
		while(!cceafs.getStartButtonClickedStatus()) {
		// waiting here until the Start button is clicked to get the input dataset and other parameters.	
		}
		jTextArea = cceafs.getJTextArea();
		TRAIN_DATASET = cceafs.getTrainAbsolutePath();
		TRAIN_DATASET_NAME = cceafs.getTrainFileName();
		classificationMode = cceafs.getClassificationMode();
		if(classificationMode == "Train-Test") {
			TEST_DATASET = cceafs.getTestAbsolutePath();
			TEST_DATASET_NAME = cceafs.getTrainFileName();
		}
		classificationModel = cceafs.getClassificationModel();
		// Loading required WEKA packages
		weka.core.WekaPackageManager.loadPackages(false,true,false);	// enable it when executing first time in a machine to load WEKA packages if required!
    	trainSource = new DataSource(TRAIN_DATASET);
    	trainData = trainSource.getDataSet();
    	
		SOLUTION_SIZE = trainData.numAttributes()-1;
		NUMBER_OF_TRAIN_SAMPLES = trainData.numInstances();
		
		if(classificationMode == "Train-Test") {
			try {
				testSource = new DataSource(TEST_DATASET);
			} catch (Exception e) {
				e.printStackTrace();
			}
			try {
				testData = testSource.getDataSet();
			} catch (Exception e) {
				e.printStackTrace();
			}
			NUMBER_OF_TEST_SAMPLES = testData.numInstances();
		}
		
		jTextArea.append("====== Dataset information ====== \n");
		jTextArea.append("Dataset Name: " + TRAIN_DATASET_NAME + ", NUMBER_OF_TRAIN_SAMPLES: " + NUMBER_OF_TRAIN_SAMPLES + 
				", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		if(classificationMode == "Train-Test") {
			jTextArea.append("\nTest Dataset Name: " + TEST_DATASET_NAME + ", NUMBER_OF_TEST_SAMPLES: " + NUMBER_OF_TEST_SAMPLES + 
					", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		}
		jTextArea.append("\nClassification MOde: " + classificationMode + ", Classification MOdel: : " + classificationModel);
	}
	
	public static void evaluateWithAllFeatures() {
		EvaluateAllFeatures eTrainTest = null;
		EvaluateAllFeaturesCross eCross = null;
		List<Object> wekaEvaluation = null;
		if(Driver.classificationMode == "Train-Test") {
			try {
				eTrainTest = new EvaluateAllFeatures();
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			try {
				wekaEvaluation = eTrainTest.CorrectlyClassified();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		else if(Driver.classificationMode == "CV") {
			try {
				eCross = new EvaluateAllFeaturesCross();
			} catch (Exception e1) {
				e1.printStackTrace();
			}
			try {
				wekaEvaluation = eCross.CorrectlyClassified();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		accuracy = (Double) wekaEvaluation.get(3);	// To get the accuracy using all of the features in the dataset and using the same classifier used later using reduced feature
		precision = (Double) wekaEvaluation.get(0);
		recall = (Double) wekaEvaluation.get(1);
		f1Score = (Double) wekaEvaluation.get(2);
		sensitivity = (Double) wekaEvaluation.get(4);
		specificity = (Double) wekaEvaluation.get(5);
		correctlyClassified = (Double) wekaEvaluation.get(6);
		strSummary = (String) wekaEvaluation.get(7);
		strClassDetails = (String) wekaEvaluation.get(8);
		strConfusionMatrix = (String) wekaEvaluation.get(9);
		
		microPrecision = (Double) wekaEvaluation.get(10);
		microRecall = (Double) wekaEvaluation.get(11);	
		microF1Score = (Double) wekaEvaluation.get(12);
		macroPrecision = (Double) wekaEvaluation.get(13);	
		macroRecall = (Double) wekaEvaluation.get(14);	
		macroF1Score = (Double) wekaEvaluation.get(15);
		weightedPrecision = (Double) wekaEvaluation.get(16);	
		weightedRecall = (Double) wekaEvaluation.get(17); 
		weightedF1Score = (Double) wekaEvaluation.get(18);
		
		matthewsCorrelationCoefficient = (Double) wekaEvaluation.get(19);
		areaUnderROC = (Double) wekaEvaluation.get(20);
		areaUnderPRC = (Double) wekaEvaluation.get(21);
		weightedMatthewsCorrelation = (Double) wekaEvaluation.get(22);
		weightedAreaUnderROC = (Double) wekaEvaluation.get(23);
		weightedAreaUnderPRC = (Double) wekaEvaluation.get(24);
	}
	public void setTrainDataset(String train) {
		TRAIN_DATASET = train;
	}

	public static void printGenBestIndividual(Individual bestIndvidual, int gen) {
		jTextArea.append("Gen # " + gen);
		jTextArea.append(" | Precision: " + bestIndvidual.getPrecision());
		jTextArea.append(" | Recall: " + bestIndvidual.getRecall());
		jTextArea.append(" | F1-Score: " + bestIndvidual.getF1Score());
		jTextArea.append(" | Sensitivity: " + bestIndvidual.getSensitivity());
		jTextArea.append(" | Specificity: " + bestIndvidual.getSpecificity());
		jTextArea.append(" | bestFitness: " + bestIndvidual.getFitness());
		jTextArea.append(" | CorrectlyClassified: " + bestIndvidual.getCorrectlyClassified());
		jTextArea.append(" | Features: " + bestIndvidual.getNumberofOnes());
		jTextArea.append(" | Accuracy: " + bestIndvidual.getAccuracy());
		jTextArea.append("\n");
		jTextArea.append(" | microPrecision: " + bestIndvidual.getMicroPrecision());
		jTextArea.append(" | microRecall: " + bestIndvidual.getMicroRecall());
		jTextArea.append(" | microF1Score: " + bestIndvidual.getMicroF1Score());
		jTextArea.append(" | macroPrecision: " + bestIndvidual.getMacroPrecision());
		jTextArea.append(" | macroRecall: " + bestIndvidual.getMacroRecall());
		jTextArea.append(" | macroF1Score: " + bestIndvidual.getMacroF1Score());
		jTextArea.append(" | weightedPrecision: " + bestIndvidual.getWeightedPrecision());
		jTextArea.append(" | weightedRecall: " + bestIndvidual.getWeightedRecall());
		jTextArea.append(" | weightedF1Score: " + bestIndvidual.getWeightedF1Score());
		jTextArea.append("\n");
		jTextArea.append(" | matthewsCorrelationCoefficient: " + bestIndvidual.getMatthewsCorrelationCoefficient());
		jTextArea.append(" | areaUnderROC: " + bestIndvidual.getAreaUnderROC());
		jTextArea.append(" | areaUnderPRC: " + bestIndvidual.getAreaUnderPRC());
		jTextArea.append(" | weightedMatthewsCorrelation: " + bestIndvidual.getWeightedMatthewsCorrelation());
		jTextArea.append(" | weightedAreaUnderROC: " + bestIndvidual.getWeightedAreaUnderROC());
		jTextArea.append(" | weightedAreaUnderPRC: " + bestIndvidual.getWeightedAreaUnderPRC());
		jTextArea.append("\n");
	}	// End of printGenBestIndividual(Individual[] bestIndvidual, int gen)
	
	private static void writeGenBestIndividual(Individual bestIndvidual, int gen) throws IOException {
		genWriter.append(String.valueOf(gen));						genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getPrecision()));					genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getRecall()));					genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getF1Score()));					genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getSensitivity()));				genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getSpecificity()));				genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getFitness()));					genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getCorrectlyClassified()));		genWriter.append(COMMA_DELIMITER);			
		genWriter.append(String.valueOf(bestIndvidual.getNumberofOnes()));				genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getAccuracy()));					genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(SOLUTION_SIZE));								genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(accuracy));										genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getMicroPrecision()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getMicroRecall()));				genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getMicroF1Score()));				genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getMacroPrecision()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getMacroRecall()));				genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getMacroF1Score()));				genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getWeightedPrecision()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getWeightedRecall()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getWeightedF1Score()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getMatthewsCorrelationCoefficient()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getAreaUnderROC()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getAreaUnderPRC()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getWeightedMatthewsCorrelation()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getWeightedAreaUnderROC()));			genWriter.append(COMMA_DELIMITER);
		genWriter.append(String.valueOf(bestIndvidual.getWeightedAreaUnderPRC()));			genWriter.append(NEW_LINE_SEPARATOR);
	}	//End of writeGenBestIndividual(Individual[] bestIndvidual, int gen)
	
	public static void printDSBestIndividual(Individual bestIndvidual) {
		jTextArea.append("\n====== Dataset information ====== \n");
		jTextArea.append("Dataset Name: " + TRAIN_DATASET_NAME + ", NUMBER_OF_TRAIN_SAMPLES: " + NUMBER_OF_TRAIN_SAMPLES + 
				", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		if(classificationMode == "Train-Test") {
			jTextArea.append("\nTest Dataset Name: " + TEST_DATASET_NAME + ", NUMBER_OF_TEST_SAMPLES: " + NUMBER_OF_TEST_SAMPLES + 
					", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		}
		jTextArea.append("\nClassification MOde: " + classificationMode + ", Classification MOdel: : " + classificationModel);
		
		jTextArea.append("\n\n\n====== ### Dataset result without using feature selection ### ====== \n");
		jTextArea.append("Precision: " + precision);
		jTextArea.append(" | Recall: " + recall);
		jTextArea.append(" | F1-Score: " + f1Score);
		jTextArea.append(" | Sensitivity: " + sensitivity);
		jTextArea.append(" | Specificity: " + specificity);
		jTextArea.append(" | CorrectlyClassified: " + correctlyClassified.intValue());
		jTextArea.append(" | Accuracy: " + accuracy);
		jTextArea.append("\n");
		jTextArea.append("microPrecision: " + microPrecision);
		jTextArea.append(" | microRecall: " + microRecall);
		jTextArea.append(" | microF1Score: " + microF1Score);
		jTextArea.append(" | macroPrecision: " + macroPrecision);
		jTextArea.append(" | macroRecall: " + macroRecall);
		jTextArea.append(" | macroF1Score: " + macroF1Score);
		jTextArea.append(" | weightedPrecision: " + weightedPrecision);
		jTextArea.append(" | weightedRecall: " + weightedRecall);
		jTextArea.append(" | weightedF1Score: " + weightedF1Score);
		jTextArea.append("\n");
		jTextArea.append("matthewsCorrelationCoefficient: " + matthewsCorrelationCoefficient);
		jTextArea.append(" | areaUnderROC: " + areaUnderROC);
		jTextArea.append(" | areaUnderPRC: " + areaUnderPRC);
		jTextArea.append(" | weightedMatthewsCorrelation: " + weightedMatthewsCorrelation);
		jTextArea.append(" | weightedAreaUnderROC: " + weightedAreaUnderROC);
		jTextArea.append(" | weightedAreaUnderPRC: " + weightedAreaUnderPRC);
		jTextArea.append("\n");
		
		jTextArea.append("\n====== Detail output with all features ====== \n");
		jTextArea.append("\n====== Summary ======" + strSummary + "\n");
		jTextArea.append(strClassDetails + "\n");
		jTextArea.append(strConfusionMatrix + "\n");
		
		jTextArea.append("\n====== Parameters used in the evaluation for feature selection ====== \n");
		jTextArea.append("NO_OF_GENERATION: " + NO_OF_GENERATION + ", NO_OF_SUBPOPULATION: " + NO_OF_SUBPOPULATION + 
				", SUBPOP_SIZE: " + SUBPOP_SIZE + ", SOLUTION_SIZE: " + SOLUTION_SIZE + ", NO_OF_GENES: " + Arrays.toString(sizeofEachSubPop) + ", ");
		jTextArea.append("NUMB_OF_ELITE_CHROMOSOMES: " + NUMB_OF_ELITE_CHROMOSOMES + ", TOURNAMENT_SELECTION_SIZE: " + TOURNAMENT_SELECTION_SIZE + 
				", CROSSOVER_RATE: " + CROSSOVER_RATE + ", MUTATION_RATE: " + MUTATION_RATE + "\n");
		jTextArea.append("Weights used fitness function: w1 = " + String.valueOf(weight1) + 
				", w2: " + String.valueOf(weight2) + ", Terminated Generation Number: " + terminatedGenerationNumber+ "\n");
		
		jTextArea.append("\n====== ### Dataset best result using feature selection after all generation ### ====== \n");
		jTextArea.append("Precision: " + bestIndvidual.getPrecision());
		jTextArea.append(" | Recall: " + bestIndvidual.getRecall());
		jTextArea.append(" | F1-Score: " + bestIndvidual.getF1Score());
		jTextArea.append(" | Sensitivity: " + bestIndvidual.getSensitivity());
		jTextArea.append(" | Specificity: " + bestIndvidual.getSpecificity());
		jTextArea.append(" | bestFitness: " + bestIndvidual.getFitness());
		jTextArea.append(" | CorrectlyClassified: " + bestIndvidual.getCorrectlyClassified());
		jTextArea.append(" | Features: " + bestIndvidual.getNumberofOnes());
		jTextArea.append(" | Accuracy: " + bestIndvidual.getAccuracy());
		jTextArea.append("\n");
		jTextArea.append("microPrecision: " + bestIndvidual.getMicroPrecision());
		jTextArea.append(" | microRecall: " + bestIndvidual.getMicroRecall());
		jTextArea.append(" | microF1Score: " + bestIndvidual.getMicroF1Score());
		jTextArea.append(" | macroPrecision: " + bestIndvidual.getMacroPrecision());
		jTextArea.append(" | macroRecall: " + bestIndvidual.getMacroRecall());
		jTextArea.append(" | macroF1Score: " + bestIndvidual.getMacroF1Score());
		jTextArea.append(" | weightedPrecision: " + bestIndvidual.getWeightedPrecision());
		jTextArea.append(" | weightedRecall: " + bestIndvidual.getWeightedRecall());
		jTextArea.append(" | weightedF1Score: " + bestIndvidual.getWeightedF1Score());
		jTextArea.append("\n");
		jTextArea.append("matthewsCorrelationCoefficient: " + bestIndvidual.getMatthewsCorrelationCoefficient());
		jTextArea.append(" | areaUnderROC: " + bestIndvidual.getAreaUnderROC());
		jTextArea.append(" | areaUnderPRC: " + bestIndvidual.getAreaUnderPRC());
		jTextArea.append(" | weightedMatthewsCorrelation: " + bestIndvidual.getWeightedMatthewsCorrelation());
		jTextArea.append(" | weightedAreaUnderROC: " + bestIndvidual.getWeightedAreaUnderROC());
		jTextArea.append(" | weightedAreaUnderPRC: " + bestIndvidual.getAreaUnderPRC());
		jTextArea.append("\n");
		
		jTextArea.append("\n====== Detail output of the best result found from the last generation ====== \n");
		jTextArea.append("\n====== Summary ======" + bestIndvidual.getStrSummary() + "\n");
		jTextArea.append(bestIndvidual.getStrClassDetails() + "\n");
		jTextArea.append(bestIndvidual.getStrConfusionMatrix() + "\n");
		
		jTextArea.append("\n====== Selected Feature Subset ====== \n");
		jTextArea.append("\n====== Number of Attribute Selected: " + String.valueOf(bestIndvidual.getNumberofOnes().intValue()) + "\n");
		jTextArea.append("\n====== Attribute Indices ======\n" + Arrays.toString(bestIndvidual.getSolutionAttributesIndex()) + "\n");
		jTextArea.append("\n====== Attribute Names ======\n" + Arrays.toString(bestIndvidual.getSolutionAttributesName()) + "\n");
		
		jTextArea.append("\n====== Total Time Elapsed for the Execution: " + String.valueOf(timer) + "\n");
	}	// End of printDSBestIndividual(Individual bestIndvidual)
	
	public static void writeDSBestIndividual(Individual bestIndvidual) throws IOException {
		
		dsRTFWriter.append("CCFSRFG-1 starts execution at: " + formatter.format(startDate) + "\n\n");
		dsRTFWriter.append("====== Dataset information ====== \n");
		dsRTFWriter.append("Dataset Name: " + TRAIN_DATASET_NAME + ", NUMBER_OF_TRAIN_SAMPLES: " + NUMBER_OF_TRAIN_SAMPLES + 
				", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		if(classificationMode == "Train-Test") {
			dsRTFWriter.append("\nTest Dataset Name: " + TEST_DATASET_NAME + ", NUMBER_OF_TEST_SAMPLES: " + NUMBER_OF_TEST_SAMPLES + 
					", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		}
		dsRTFWriter.append("\nClassification MOde: " + classificationMode + ", Classification MOdel: : " + classificationModel);
		
		dsRTFWriter.append("\n\n\n====== ### Dataset result without using feature selection ### ====== \n");
		dsRTFWriter.append("Precision\t\t\tRecall\t\t\tF1-Score\t\t\tSensitivity\t\t\tSpecificity\t\t\tCorrectlyClassified\t\t\tAccuracy\n");
		dsRTFWriter.append(String.valueOf(precision) + "\t" + String.valueOf(recall) + "\t" + String.valueOf(f1Score) + "\t" + 
				String.valueOf(sensitivity) + "\t" + String.valueOf(specificity) + "\t" + String.valueOf(correctlyClassified.intValue()) + 
				"\t" + String.valueOf(accuracy));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("microPrecision\t\t\tmicroRecall\t\t\tmicroF1Score\t\t\tmacroPrecision\t\t\tmacroRecall\t\t\tmacroF1Score\n");
		dsRTFWriter.append(String.valueOf(microPrecision) + "\t" + String.valueOf(microRecall) + "\t" + String.valueOf(microF1Score) + 
				"\t" + String.valueOf(macroPrecision) + "\t" + String.valueOf(macroRecall) + "\t" + String.valueOf(macroF1Score));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("weightedPrecision\t\tweightedRecall\t\tweightedF1Score\n");
		dsRTFWriter.append(String.valueOf(weightedPrecision) + "\t" + String.valueOf(weightedRecall) + "\t" + String.valueOf(weightedF1Score));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("matthewsCorrelationCoefficient\tareaUnderROC\tareaUnderPRC\tweightedMatthewsCorrelation\tweightedAreaUnderROC\tweightedAreaUnderPRC\n");
		dsRTFWriter.append(String.valueOf(matthewsCorrelationCoefficient) + "\t\t" + String.valueOf(areaUnderROC) + "\t\t" + String.valueOf(areaUnderPRC) + 
				"\t\t" + String.valueOf(weightedMatthewsCorrelation) + "\t\t" + String.valueOf(weightedAreaUnderROC) + "\t\t" + String.valueOf(weightedAreaUnderPRC));
		dsRTFWriter.append("\n");
		
		dsRTFWriter.append("\n====== Detail output ====== \n");
		dsRTFWriter.append("\n====== Summary ======" + strSummary + "\n");
		dsRTFWriter.append(String.valueOf(strClassDetails) + "\n");
		dsRTFWriter.append(String.valueOf(strConfusionMatrix) + "\n");
		
		dsRTFWriter.append("\n====== Parameters used in the evaluation for feature selection ====== \n");
		dsRTFWriter.append("NO_OF_GENERATION: " + String.valueOf(NO_OF_GENERATION) + ", NO_OF_SUBPOPULATION: " + String.valueOf(NO_OF_SUBPOPULATION) + 
				", SUBPOP_SIZE: " + String.valueOf(SUBPOP_SIZE) + ", SOLUTION_SIZE: " + String.valueOf(SOLUTION_SIZE) + 
				", NO_OF_GENES: " + Arrays.toString(sizeofEachSubPop) + ", ");
		dsRTFWriter.append("\nNUMB_OF_ELITE_CHROMOSOMES: " + String.valueOf(NUMB_OF_ELITE_CHROMOSOMES) + 
				", TOURNAMENT_SELECTION_SIZE: " + String.valueOf(TOURNAMENT_SELECTION_SIZE) + 
				", CROSSOVER_RATE: " + String.valueOf(CROSSOVER_RATE) + ", MUTATION_RATE: " + String.valueOf(MUTATION_RATE));
		dsRTFWriter.append("\nWeights used fitness function: w1 = " + String.valueOf(weight1) + 
				", w2: " + String.valueOf(weight2) + ", Terminated Generation Number: " + terminatedGenerationNumber+ "\n");
		
		dsRTFWriter.append("\n====== ### Dataset best result using feature selection after all generation ### ====== \n");
		dsRTFWriter.append("Precision\t\t\tRecall\t\t\tF1-Score\t\t\tSensitivity\t\t\tSpecificity\t\t\tbestFitness\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getPrecision()) + "\t" + String.valueOf(bestIndvidual.getRecall()) + "\t" + 
				String.valueOf(bestIndvidual.getF1Score()) + "\t" + String.valueOf(bestIndvidual.getSensitivity()) + "\t" + 
				String.valueOf(bestIndvidual.getSpecificity()) + "\t" + String.valueOf(bestIndvidual.getFitness()));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("CorrectlyClassified\tFeatures\tAccuracy\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getCorrectlyClassified().intValue()) + "\t\t\t\t" + 
				String.valueOf(bestIndvidual.getNumberofOnes().intValue()) + "\t\t\t\t" + String.valueOf(bestIndvidual.getAccuracy()));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("microPrecision\t\t\tmicroRecall\t\t\tmicroF1Score\t\t\tmacroPrecision\t\t\tmacroRecall\t\t\tmacroF1Score\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getMicroPrecision()) + "\t" + String.valueOf(bestIndvidual.getMicroRecall()) + "\t" + 
				String.valueOf(bestIndvidual.getMicroF1Score()) + "\t" + String.valueOf(bestIndvidual.getMacroPrecision()) + "\t" + 
				String.valueOf(bestIndvidual.getMacroRecall()) + "\t" + String.valueOf(bestIndvidual.getMacroF1Score()));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("weightedPrecision\t\tweightedRecall\t\tweightedF1Score\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getWeightedPrecision()) + "\t" + String.valueOf(bestIndvidual.getWeightedRecall()) + "\t" + 
				String.valueOf(bestIndvidual.getWeightedF1Score()));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("matthewsCorrelationCoefficient\tareaUnderROC\tareaUnderPRC\tweightedMatthewsCorrelation\tweightedAreaUnderROC\tweightedAreaUnderPRC\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getMatthewsCorrelationCoefficient()) + "\t\t" + String.valueOf(bestIndvidual.getAreaUnderROC()) + "\t\t" + 
				String.valueOf(bestIndvidual.getAreaUnderPRC()) + "\t\t" + String.valueOf(bestIndvidual.getWeightedMatthewsCorrelation()) + "\t\t" +
				String.valueOf(bestIndvidual.getWeightedAreaUnderROC()) + "\t\t" + String.valueOf(bestIndvidual.getWeightedAreaUnderPRC()));
		dsRTFWriter.append("\n");
		
		dsRTFWriter.append("\n====== Detail output of the best result found from the last generation ====== \n");
		dsRTFWriter.append("\n====== Summary ======" + String.valueOf(bestIndvidual.getStrSummary()) + "\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getStrClassDetails()) + "\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getStrConfusionMatrix()) + "\n");
		
		dsRTFWriter.append("\n====== Selected Feature Subset ====== \n");
		dsRTFWriter.append("\n====== Number of Attribute Selected: " + String.valueOf(bestIndvidual.getNumberofOnes().intValue()) + "\n");
		dsRTFWriter.append("\n====== Attribute Indices ======\n\n" + Arrays.toString(bestIndvidual.getSolutionAttributesIndex()) + "\n");
		dsRTFWriter.append("\n====== Attribute Names ======\n\n" + Arrays.toString(bestIndvidual.getSolutionAttributesName()) + "\n");
		
		dsRTFWriter.append("\n====== Total Time Elapsed for the Execution: " + String.valueOf(timer) + "\n");
		
		dsRTFWriter.append("\nCSV and RTF files created successfully!!! at \n\n" + CSV_GEN + "\n" + RTF_DS + "\n");
		dsRTFWriter.append("\nCCFSRFG-1 ends execution at: " + formatter.format(endDate) + "\n");
	}	// End of writeDSBestIndividual(Individual bestIndvidual)
	
}	// End of Driver Class
