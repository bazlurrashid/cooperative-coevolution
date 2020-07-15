package cceafs;

import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.List;

import javax.swing.JTextArea;

import com.google.common.base.Stopwatch;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Driver{
	
	public static final int NO_OF_GENERATION = 10000;
	public static final int NO_OF_SUBPOPULATION = 3;	// set the number of sub-population based on dataset
	public static final int SUBPOP_SIZE = 30;
	public static int SOLUTION_SIZE;
	public static int NO_OF_GENES;
	public static final int NUMB_OF_ELITE_CHROMOSOMES = 1;
	public static final int TOURNAMENT_SELECTION_SIZE = 1;
	public static final Double CROSSOVER_RATE = 1.00;
	public static final Double MUTATION_RATE = 0.15;
	public static final Double weight1 = 0.60;
	public static final Double weight2 = 0.40;
	
	public static String TRAIN_DATASET;
	public static String TEST_DATASET;
	public static String TRAIN_DATASET_NAME;
	public static String TEST_DATASET_NAME;
	public static int NUMBER_OF_TRAIN_SAMPLES;	
	public static int NUMBER_OF_TEST_SAMPLES;	
	
	public static String MODEL_SAVE = "D:\\feature-selection\\output\\CCEAFS\\trained.model";
	public static String DIRECTORY = "D:\\feature-selection\\output\\CCEAFS\\";	
	public static String CSV_GEN = "", CSV_SUB = "", CSV_IND = "", RTF_GEN = "", RTF_SUB = "", RTF_IND = "", RTF_DS = "";
	
	public static DataSource trainSource;	
	public static DataSource testSource;	
	public static Instances trainData;	
	public static Instances testData;	
	
    public static String COMMA_DELIMITER = ",";
    public static String NEW_LINE_SEPARATOR = "\n";

    public static String GEN_HEADER = "generation,";	
    public static String SUB_HEADER = "generation,";	
    public static String IND_HEADER = "generation,";	
    public static String GEN_RTF_HEADER = "";	
    public static String SUB_RTF_HEADER = "";	
    public static String IND_RTF_HEADER = "";	
    public static String DS_RTF_HEADER = "";	
    public static FileWriter genWriter;	
    public static FileWriter subWriter;	
    public static FileWriter indWriter;	
    public static FileWriter genRTFWriter;	
    public static FileWriter subRTFWriter;	
    public static FileWriter indRTFWriter;	
    public static FileWriter dsRTFWriter;	
    
    public static Double accuracy;	
    public static Double precision;
    public static Double recall;
    public static Double f1Score;

    public static Double sensitivity;
    public static Double specificity;
    public static String strSummary;
    public static String strClassDetails;
    public static String strConfusionMatrix;
    public static Double correctlyClassified;
    
	private static Double weightedPrecision;
	private static Double weightedRecall;
	private static Double weightedF1Score;
	private static Double microPrecision;
	private static Double microRecall;
	private static Double microF1Score;
	private static Double macroPrecision;
	private static Double macroRecall;
	private static Double macroF1Score;
    
    public static JTextArea jTextArea;
    public static String classificationMode;
    public static String classificationModel;
    
    public static Double[] fitnessStagnation;
    public static int stagnationCount = 0;
    public static int terminatedGenerationNumber;
    
    public static Stopwatch timer;
	
    public Driver() {

    }
	@SuppressWarnings("unused")
	public static void main(String[] args) throws Exception {	// Start of main() function
		fitnessStagnation = new Double[NO_OF_GENERATION];
		
		SimpleDateFormat formatter = new SimpleDateFormat("dd/MM/yyyy HH:mm:ss");  
        Date date = new Date(); 
		
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
		jTextArea.append("\nCCEAFS starts at: " + formatter.format(date) + "\n");
		
		try {
			trainSource = new DataSource(TRAIN_DATASET);
		} catch (Exception e1) {
			e1.printStackTrace();
		}
		try {
			trainData = trainSource.getDataSet();
		} catch (Exception e1) {
			e1.printStackTrace();
		}
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
		
		accuracy = (Double) wekaEvaluation.get(3);	
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
		
		SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_kkmmss");	
        Date curDate = new Date();
        String strDate = sdf.format(curDate);
        CSV_GEN = DIRECTORY + TRAIN_DATASET_NAME + "_" + classificationModel + "_gen_" + strDate + ".csv";
        RTF_DS = DIRECTORY + TRAIN_DATASET_NAME + "_"  + classificationModel + "_DS_" + strDate + ".rtf";
       
		try {
            genWriter = new FileWriter(CSV_GEN);
            GEN_HEADER += "precision," + "recall," + "f1-score," + "sensitivity," + "specificity," + "fitness," + 
            		"correctlyClassified," + "reducedFeatures," + "accuracy," + "originalFeatures," + "originalAccuracy," + 
            		"microPrecision," + "microRecall," + "microF1Score," + "macroPrecision," + "macroRecall," + "macroF1Score," +
            		"weightedPrecision," + "weightedRecall," + "weightedF1Score"; 
            dsRTFWriter = new FileWriter(RTF_DS);
            DS_RTF_HEADER += "";
            genWriter.append(GEN_HEADER.toString());	genWriter.append(NEW_LINE_SEPARATOR);
            dsRTFWriter.append(DS_RTF_HEADER.toString());	dsRTFWriter.append(NEW_LINE_SEPARATOR);
		
			NO_OF_GENES = SOLUTION_SIZE / NO_OF_SUBPOPULATION;	// Deciding total number genes for each individual
			GeneticAlgorithm ga = new GeneticAlgorithm();
			SubPopulation[] subPopulation = new SubPopulation[NO_OF_SUBPOPULATION];
			ContextVector contextVector = new ContextVector();
			Individual[] bestIndvidual = new Individual[NO_OF_SUBPOPULATION];
			SortBestIndividualesByFitness sbi = new SortBestIndividualesByFitness(bestIndvidual);
			
			jTextArea.append("\n");
			timer = Stopwatch.createUnstarted();	
	    	timer.start();	
			
			int generationNumber = 0;	// starting at generation 0
			for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {	
				subPopulation[x] = new SubPopulation(SUBPOP_SIZE).initializeSubPopulation();	// initializing individuals of each subpopulation using random bits
			}
		
			contextVector.findRandomCollaborators(subPopulation);	// finding random collaborators from other subpopulation for generation 0
		
			for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {	
				for(int y = 0; y < SUBPOP_SIZE; y++) {
					subPopulation[x].getIndividuals()[y].setFitness();	// calculating fitness of each individual from all subpopulations
				}

				subPopulation[x].sortIndividualesByFitness();	// sorting all individuals in each subpopulation based on objective function 
			}
		
			for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {	
				bestIndvidual[x] = subPopulation[x].getIndividuals()[0];	// finding best individuals from all subpopulations
			}
			
			bestIndvidual = sbi.sortIndividuales();	// sorting all best individuals found from subpopulations based on objective function
			
			printGenBestIndividual(bestIndvidual[0], generationNumber);	// displaying generation-wise best individual found from all subpopulation
			writeGenBestIndividual(bestIndvidual[0], generationNumber);	// writing generation-wise best individual found from all subpopulation
			
			fitnessStagnation[generationNumber] = bestIndvidual[0].getFitness();
		
			generationNumber = 1;
			while(generationNumber < NO_OF_GENERATION) {
				for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {
					subPopulation[x] = ga.evolve(subPopulation[x], bestIndvidual[0]);	// evolving each subpopulation using GA
				}
			
				contextVector.findBestCollaborators(subPopulation);	// finding best individuals of other subpopulations from previous generation 

				for(int x = 0; x < NO_OF_SUBPOPULATION; x++) {
					for(int y = 0; y < SUBPOP_SIZE; y++) {
						subPopulation[x].getIndividuals()[y].setFitness();	// Calculating fitness of each individual from all subpopulations
				}
					
				subPopulation[x].sortIndividualesByFitness();	// sorting all individuals in each subpopulation based on objective function 
				}
			
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
				
//				if(stagnationCount >= NO_OF_GENERATION * 0.30) {
				if(stagnationCount >= 50) {	// set the number of generations to check no more improvement in solution
					terminatedGenerationNumber = generationNumber;
					timer.stop();	// counting the time when it terminates
					break;
				}
				
				generationNumber++;	// incrementing the generation number
			}	// End of while loop checking number of generations condition
			
			printDSBestIndividual(bestIndvidual[0]);	// displaying dataset best result found from all generation
			writeDSBestIndividual(bestIndvidual[0]);	// writing dataset best result found from all generation
			
			jTextArea.append("\nCSV and RTF files created successfully!!! at \n\n" + CSV_GEN + "\n" + CSV_SUB + "\n" + CSV_IND + "\n"
					+ RTF_GEN + "\n" + RTF_SUB + "\n" + RTF_IND + "\n" + RTF_DS + "\n");
			jTextArea.append("\nCCEAFS execution completed successfully!!!");
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
		date = new Date();  
		jTextArea.append("\nCCEAFS stops at: " + formatter.format(date) + "\n");
		
	}	// End of main() function
	
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
		jTextArea.append(" | CorrectlyClassified: " + bestIndvidual.getCorrectlyClassified().intValue());
		jTextArea.append(" | Features: " + bestIndvidual.getNumberofOnes().intValue());
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
	}	// End of printGenBestIndividual(Individual[] bestIndvidual, int gen)
	
	private static void writeGenBestIndividual(Individual bestIndvidual, int gen) throws IOException {
		genWriter.append("generation # " + String.valueOf(gen));						genWriter.append(COMMA_DELIMITER);
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
		genWriter.append(String.valueOf(bestIndvidual.getWeightedF1Score()));			genWriter.append(NEW_LINE_SEPARATOR);
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
		jTextArea.append(" | microPrecision: " + microPrecision);
		jTextArea.append(" | microRecall: " + microRecall);
		jTextArea.append(" | microF1Score: " + microF1Score);
		jTextArea.append(" | macroPrecision: " + macroPrecision);
		jTextArea.append(" | macroRecall: " + macroRecall);
		jTextArea.append(" | macroF1Score: " + macroF1Score);
		jTextArea.append(" | weightedPrecision: " + weightedPrecision);
		jTextArea.append(" | weightedRecall: " + weightedRecall);
		jTextArea.append(" | weightedF1Score: " + weightedF1Score);
		jTextArea.append("\n");
		
		jTextArea.append("\n====== Detail output with all features ====== \n");
		jTextArea.append("\n====== Summary ======" + strSummary + "\n");
		jTextArea.append(strClassDetails + "\n");
		jTextArea.append(strConfusionMatrix + "\n");
		
		jTextArea.append("\n====== Parameters used in the evaluation for feature selection ====== \n");
		jTextArea.append("NO_OF_GENERATION: " + NO_OF_GENERATION + ", NO_OF_SUBPOPULATION: " + NO_OF_SUBPOPULATION + 
				", SUBPOP_SIZE: " + SUBPOP_SIZE + ", SOLUTION_SIZE: " + SOLUTION_SIZE + ", NO_OF_GENES: " + NO_OF_GENES + ", ");
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
		jTextArea.append(" | CorrectlyClassified: " + bestIndvidual.getCorrectlyClassified().intValue());
		jTextArea.append(" | Features: " + bestIndvidual.getNumberofOnes().intValue());
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
		
		dsRTFWriter.append("====== Dataset information ====== \n");
		dsRTFWriter.append("Dataset Name: " + TRAIN_DATASET_NAME + ", NUMBER_OF_TRAIN_SAMPLES: " + NUMBER_OF_TRAIN_SAMPLES + 
				", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		if(classificationMode == "Train-Test") {
			dsRTFWriter.append("\nTest Dataset Name: " + TEST_DATASET_NAME + ", NUMBER_OF_TEST_SAMPLES: " + NUMBER_OF_TEST_SAMPLES + 
					", NUMBER_OF_FEATURES: " + SOLUTION_SIZE);
		}
		dsRTFWriter.append("\nClassification MOde: " + classificationMode + ", Classification MOdel: : " + classificationModel);
		
		dsRTFWriter.append("\n\n\n====== ### Dataset result without using feature selection ### ====== \n");
		dsRTFWriter.append("Precision: " + String.valueOf(precision));
		dsRTFWriter.append(" | Recall: " + String.valueOf(recall));
		dsRTFWriter.append(" | F1-Score: " + String.valueOf(f1Score));
		dsRTFWriter.append(" | Sensitivity: " + String.valueOf(sensitivity));
		dsRTFWriter.append(" | Specificity: " + String.valueOf(specificity));
		dsRTFWriter.append(" | CorrectlyClassified: " + String.valueOf(correctlyClassified.intValue()));
		dsRTFWriter.append(" | Accuracy: " + String.valueOf(accuracy));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("microPrecision: " + String.valueOf(microPrecision));
		dsRTFWriter.append(" | microRecall: " + String.valueOf(microRecall));
		dsRTFWriter.append(" | microF1Score: " + String.valueOf(microF1Score));
		dsRTFWriter.append("\n");
		dsRTFWriter.append(" | macroPrecision: " + String.valueOf(macroPrecision));
		dsRTFWriter.append(" | macroRecall: " + String.valueOf(macroRecall));
		dsRTFWriter.append(" | macroF1Score: " + String.valueOf(macroF1Score));
		dsRTFWriter.append("\n");
		dsRTFWriter.append(" | weightedPrecision: " + String.valueOf(weightedPrecision));
		dsRTFWriter.append(" | weightedRecall: " + String.valueOf(weightedRecall));
		dsRTFWriter.append(" | weightedF1Score: " + String.valueOf(weightedF1Score));
		dsRTFWriter.append("\n");
		
		dsRTFWriter.append("\n====== Detail output ====== \n");
		dsRTFWriter.append("\n====== Summary ======" + strSummary + "\n");
		dsRTFWriter.append(String.valueOf(strClassDetails) + "\n");
		dsRTFWriter.append(String.valueOf(strConfusionMatrix) + "\n");
		
		dsRTFWriter.append("\n====== Parameters used in the evaluation for feature selection ====== \n");
		dsRTFWriter.append("NO_OF_GENERATION: " + String.valueOf(NO_OF_GENERATION) + ", NO_OF_SUBPOPULATION: " + String.valueOf(NO_OF_SUBPOPULATION) + 
				", SUBPOP_SIZE: " + String.valueOf(SUBPOP_SIZE) + ", SOLUTION_SIZE: " + String.valueOf(SOLUTION_SIZE) + 
				", NO_OF_GENES: " + String.valueOf(NO_OF_GENES) + ", ");
		dsRTFWriter.append("NUMB_OF_ELITE_CHROMOSOMES: " + String.valueOf(NUMB_OF_ELITE_CHROMOSOMES) + 
				", TOURNAMENT_SELECTION_SIZE: " + String.valueOf(TOURNAMENT_SELECTION_SIZE) + 
				", CROSSOVER_RATE: " + String.valueOf(CROSSOVER_RATE) + ", MUTATION_RATE: " + String.valueOf(MUTATION_RATE) + "\n");
		dsRTFWriter.append("Weights used fitness function: w1 = " + String.valueOf(weight1) + 
				", w2: " + String.valueOf(weight2) + ", Terminated Generation Number: " + terminatedGenerationNumber+ "\n");
		
		dsRTFWriter.append("\n====== ### Dataset best result using feature selection after all generation ### ====== \n");
		dsRTFWriter.append("Precision: " + String.valueOf(bestIndvidual.getPrecision()));
		dsRTFWriter.append(" | Recall: " + String.valueOf(bestIndvidual.getRecall()));
		dsRTFWriter.append(" | F1-Score: " + String.valueOf(bestIndvidual.getF1Score()));
		dsRTFWriter.append(" | Sensitivity: " + String.valueOf(bestIndvidual.getSensitivity()));
		dsRTFWriter.append(" | Specificity: " + String.valueOf(bestIndvidual.getSpecificity()));
		dsRTFWriter.append(" | bestFitness: " + String.valueOf(bestIndvidual.getFitness()));
		dsRTFWriter.append(" | CorrectlyClassified: " + String.valueOf(bestIndvidual.getCorrectlyClassified().intValue()));
		dsRTFWriter.append(" | Features: " + String.valueOf(bestIndvidual.getNumberofOnes().intValue()));
		dsRTFWriter.append(" | Accuracy: " + String.valueOf(bestIndvidual.getAccuracy()));
		dsRTFWriter.append("\n");
		dsRTFWriter.append("microPrecision: " + String.valueOf(bestIndvidual.getMicroPrecision()));
		dsRTFWriter.append(" | microRecall: " + String.valueOf(bestIndvidual.getMicroRecall()));
		dsRTFWriter.append(" | microF1Score: " + String.valueOf(bestIndvidual.getMicroF1Score()));
		dsRTFWriter.append("\n");
		dsRTFWriter.append(" | macroPrecision: " + String.valueOf(bestIndvidual.getMacroPrecision()));
		dsRTFWriter.append(" | macroRecall: " + String.valueOf(bestIndvidual.getMacroRecall()));
		dsRTFWriter.append(" | macroF1Score: " + String.valueOf(bestIndvidual.getMacroF1Score()));
		dsRTFWriter.append("\n");
		dsRTFWriter.append(" | weightedPrecision: " + String.valueOf(bestIndvidual.getWeightedPrecision()));
		dsRTFWriter.append(" | weightedRecall: " + String.valueOf(bestIndvidual.getWeightedRecall()));
		dsRTFWriter.append(" | weightedF1Score: " + String.valueOf(bestIndvidual.getWeightedF1Score()));
		dsRTFWriter.append("\n");
		
		dsRTFWriter.append("\n====== Detail output of the best result found from the last generation ====== \n");
		dsRTFWriter.append("\n====== Summary ======" + String.valueOf(bestIndvidual.getStrSummary()) + "\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getStrClassDetails()) + "\n");
		dsRTFWriter.append(String.valueOf(bestIndvidual.getStrConfusionMatrix()) + "\n");
		
		dsRTFWriter.append("\n====== Selected Feature Subset ====== \n");
		dsRTFWriter.append("\n====== Number of Attribute Selected: " + String.valueOf(bestIndvidual.getNumberofOnes().intValue()) + "\n");
		dsRTFWriter.append("\n====== Attribute Indices ======\n" + Arrays.toString(bestIndvidual.getSolutionAttributesIndex()) + "\n");
		dsRTFWriter.append("\n====== Attribute Names ======\n" + Arrays.toString(bestIndvidual.getSolutionAttributesName()) + "\n");
		
		dsRTFWriter.append("\n====== Total Time Elapsed for the Execution: " + String.valueOf(timer) + "\n");
	}	// End of writeDSBestIndividual(Individual bestIndvidual)
	
}	// End of Driver Class
