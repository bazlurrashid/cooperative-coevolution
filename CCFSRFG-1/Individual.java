package ccfsrfg1;

import java.util.Arrays;
import java.util.List;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Individual{
	private int[] genes;	// To store the number of genes in each individual
	private int[] attributeIndices;
	private int[][] contextVectors;	// To store collaborative individuals from other subpopulation for each of the individual from current subpopulation
	private int[][] contextVectorsIndices;

	private int[] attributes;	// To store the total number of genes from the individual plus the collaborators
	private int[] solutionVector;
	private int[] solutionVectorIndices;
	private int[] solutionAttributesIndex;
	private String[] solutionAttributesName;
	private int[] deleteAttributes;
	private int numofDeleteAttributes;	// Counting the total number of zeros to be deleted from the solution vector to classify
	private Double fitness;	// To store the classification accuracy for each individual

	private boolean isFitnessChanged = true;

	private Double precision, recall, f1Score, accuracy, sensitivity, specificity, numberofOnes, correctlyClassified;
	private String strSummary, strClassDetails, strConfusionMatrix;
	private Double weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score;
	private Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
	
	Instances trainData;
	
	public Individual(int length) throws Exception {
		genes = new int[length];
		attributeIndices = new int[length];
		attributes = new int[Driver.SOLUTION_SIZE];
		contextVectors = new int[Driver.NO_OF_SUBPOPULATION - 1][length];
		contextVectorsIndices = new int[Driver.NO_OF_SUBPOPULATION - 1][length];
		solutionVector = new int[Driver.SOLUTION_SIZE];
		solutionVectorIndices = new int[Driver.SOLUTION_SIZE];
		trainData = Driver.trainData;
	}
	
	public Individual initializeIndividual(int subPopNumber) {	// Random initialization of each individual 
		for(int x = 0; x < genes.length; x++) {
			attributeIndices[x] = Driver.groupedIndices.get(subPopNumber).get(x);
		}

		for(int x = 0; x < genes.length; x++) {
			if(Math.random() >= 0.5) {
				genes[x] = 1;
			}
			else {
				genes[x] = 0;
			}
		}
		return this;
	}
	
	public void setIndividuals(int[] ind, int[] att, int[] cv) {
		this.genes = ind;
		this.attributeIndices = att;
		for(int x = 0; x < Driver.NO_OF_SUBPOPULATION - 1; x++) {
			this.contextVectors[x] = cv;
		}
	}

	public void setContextGenes(int count, int[] cv) { contextVectors[count] = cv; }
	
	public void setContextIndices(int count, int[] cvatt) { contextVectorsIndices[count] = cvatt; }
	
	public int[] getContextGenes(int x) { return contextVectors[x]; }
	
	public int[] getContextIndices(int x) { return contextVectorsIndices[x]; }
	
	public void setGenes(int[] Genes, int[] Atts) {
		genes = Genes;
		attributeIndices = Atts;
	}
	
	public void setGene(int index, int gene, int att) {
        genes[index] = gene;
        attributeIndices[index] = att;
    }

	public int[] getGenes() {
		isFitnessChanged = true;
		return genes;
	}
	
	public int[] getAttributeIndices() { return attributeIndices; }
	
	public void setNumberofOnes(Double fnumberofOnes) { numberofOnes = fnumberofOnes; }
	
	public Double getNumberofOnes() { return numberofOnes; }
	
	public void setPrecision(Double prec) { precision = prec; }
	
	public Double getPrecision() { return precision; }
	
	public void setRecall(Double rec) { recall = rec; }
	
	public Double getRecall() { return recall; }
	
	public void setF1Score(Double f1s) { f1Score = f1s; }
	
	public Double getF1Score() { return f1Score; }

	public void setAccuracy(Double faccuracy) {	accuracy = faccuracy; }
	
	public Double getAccuracy() { return accuracy; }

	public void setSensitivity(Double sen) { sensitivity = sen; }
	
	public Double getSensitivity() { return sensitivity; }
	
	public void setSpecificity(Double spec) { specificity = spec; }
	
	public Double getSpecificity() { return specificity; }
	
	public void setCorrectlyClassified(Double correct) { correctlyClassified = correct; }
	
	public Double getCorrectlyClassified() { return correctlyClassified; }
	
	public void setStrSummary(String summary) { strSummary = summary; }
	
	public String getStrSummary() { return strSummary; }
	
	public void setStrClassDetails(String classDetails) { strClassDetails = classDetails; }
	
	public String getStrClassDetails() { return strClassDetails; }
	
	public void setStrConfusionMatrix(String confusionMatrix) { strConfusionMatrix = confusionMatrix; }
	
	public String getStrConfusionMatrix() { return strConfusionMatrix; }
	
	public void setMicroPrecision(Double miPrecision) { microPrecision = miPrecision; }
	
	public Double getMicroPrecision() { return microPrecision; }
	
	public void setMicroRecall(Double miRecall) { microRecall = miRecall; }
	
	public Double getMicroRecall() { return microRecall; }
	
	public void setMicroF1Score(Double miF1Score) { microF1Score = miF1Score; }
	
	public Double getMicroF1Score() { return microF1Score; }
	
	public void setMacroPrecision(Double maPrecision) { macroPrecision = maPrecision; }
	
	public Double getMacroPrecision() { return macroPrecision; }
	
	public void setMacroRecall(Double maRecall) { macroRecall = maRecall; }
	
	public Double getMacroRecall() { return macroRecall; }
	
	public void setMacroF1Score(Double maF1Score) { macroF1Score = maF1Score; }
	
	public Double getMacroF1Score() { return macroF1Score; }
	
	public void setWeightedPrecision(Double wePrecision) { weightedPrecision = wePrecision; }
	
	public Double getWeightedPrecision() { return weightedPrecision; }
	
	public void setWeightedRecall(Double weRecall) { weightedRecall = weRecall; }
	
	public Double getWeightedRecall() { return weightedRecall; }
	
	public void setWeightedF1Score(Double weF1Score) { weightedF1Score = weF1Score; }
	
	public Double getWeightedF1Score() { return weightedF1Score; }
	
	public void setMatthewsCorrelationCoefficient(Double mCorrCoeff) { matthewsCorrelationCoefficient = mCorrCoeff;}
	
	public Double getMatthewsCorrelationCoefficient() { return matthewsCorrelationCoefficient;}
	
	public void setAreaUnderROC(Double areaROC) { areaUnderROC = areaROC;}
	
	public Double getAreaUnderROC() { return areaUnderROC;}
	
	public void setAreaUnderPRC(Double areaPRC) { areaUnderPRC = areaPRC;}
	
	public Double getAreaUnderPRC() { return areaUnderPRC;}
	
	public void setWeightedMatthewsCorrelation(Double wMattCorrCoeff) { weightedMatthewsCorrelation = wMattCorrCoeff;}
	
	public Double getWeightedMatthewsCorrelation() { return weightedMatthewsCorrelation;}
	
	public void setWeightedAreaUnderROC(Double wAreaROC) { weightedAreaUnderROC = wAreaROC;}
	
	public Double getWeightedAreaUnderROC() { return weightedAreaUnderROC;}
	
	public void setWeightedAreaUnderPRC(Double wAreaPRC) { weightedAreaUnderPRC = wAreaPRC;}
	
	public Double getWeightedAreaUnderPRC() { return weightedAreaUnderPRC;}
	
	public int[] getSolutionAttributesIndex() { return solutionAttributesIndex; }
	
	public String[] getSolutionAttributesName() { return solutionAttributesName; }

	public Double countNumberofOnes() {
		int count = 0;
		int cv_gene = 0;
		
		for(int x = 0; x < genes.length; x++) {
			solutionVector[x] = genes[x];
			solutionVectorIndices[x] = attributeIndices[x];
		}
		
		cv_gene = genes.length;

		for(int x = 0; x < Driver.NO_OF_SUBPOPULATION - 1; x++) {
			for(int y = 0; y < contextVectors[x].length; y++) {
				solutionVector[cv_gene] = contextVectors[x][y];
				solutionVectorIndices[cv_gene] = contextVectorsIndices[x][y];
				cv_gene++;
			}
		}
		for(int x = 0; x < solutionVector.length; x++) {
			if(solutionVector[x] == 1)
			{
				count++;
			}
		}

		return Double.valueOf(count);
	}
	
	public void setSolutionAttributesIndexName() {
		int size = getNumberofOnes().intValue();
		int count = 0;
		solutionAttributesIndex = new int[size];
		solutionAttributesName = new String[size];
		for(int x = 0; x < solutionVector.length; x++) {
			if(solutionVector[x] == 1) {
				solutionAttributesIndex[count] = solutionVectorIndices[x]+1;
				solutionAttributesName[count] = trainData.attribute(solutionVectorIndices[x]).name();
				count++;
			}
		}
		
		int tempSolutionAttributesIndex;
		String tempSolutionAttributesName;
		
		for(int p = 0; p < count; p++) {
			for(int q = 0; q < count; q++) {
				for(int r = 0; r < count - q - 1; r++) {
					if(solutionAttributesIndex[r] > solutionAttributesIndex[r+1]) {
						tempSolutionAttributesIndex = solutionAttributesIndex[r];	tempSolutionAttributesName = solutionAttributesName[r];
						solutionAttributesIndex[r] = solutionAttributesIndex[r+1];	solutionAttributesName[r] = solutionAttributesName[r+1];
						solutionAttributesIndex[r+1] = tempSolutionAttributesIndex;	solutionAttributesName[r+1] = tempSolutionAttributesName;
					}
				}
			}
		}

	}
	
	public int[] toBeDeletedAttributes() {
		numofDeleteAttributes = 0;
		for(int x = 0; x < solutionVector.length; x++) {
			if(solutionVector[x] == 0) {
				attributes[numofDeleteAttributes] = solutionVectorIndices[x]+1;
				numofDeleteAttributes++;
			}
		}
		
		return attributes;
	}

	public void setFitness() {
			fitness = calculateFitness();
	}
	
	public Double getFitness() {
		return fitness;
	}
	
	public Double calculateFitness() {
		
		Double fCorrect, fnumberofOnes;
		fnumberofOnes = countNumberofOnes();
		deleteAttributes = toBeDeletedAttributes();

		EvaluateFeatures eTrainTest;
		EvaluateFeaturesCross eCross;
		List<Object> wekaEvaluation = null;
		if(Driver.classificationMode == "Train-Test") {
			eTrainTest = new EvaluateFeatures(deleteAttributes, numofDeleteAttributes);
			numofDeleteAttributes = 0;
			try {
				wekaEvaluation = eTrainTest.CorrectlyClassified();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		else if(Driver.classificationMode == "CV") {
			eCross = new EvaluateFeaturesCross(deleteAttributes, numofDeleteAttributes);
			numofDeleteAttributes = 0;
			try {
				wekaEvaluation = eCross.CorrectlyClassified();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		setPrecision((Double) wekaEvaluation.get(0) * 100);
		setRecall((Double) wekaEvaluation.get(1) * 100);
		setF1Score((Double) wekaEvaluation.get(2) * 100);
		setAccuracy((Double) wekaEvaluation.get(3));
		setSensitivity((Double) wekaEvaluation.get(4) * 100);
		setSpecificity((Double) wekaEvaluation.get(5) * 100);
		fCorrect = (Double) wekaEvaluation.get(6);
		setNumberofOnes(fnumberofOnes);
		setCorrectlyClassified(fCorrect);
		setStrSummary((String) wekaEvaluation.get(7));
		setStrClassDetails((String) wekaEvaluation.get(8));
		setStrConfusionMatrix((String) wekaEvaluation.get(9));
		
		setMicroPrecision((Double) wekaEvaluation.get(10));
		setMicroRecall((Double) wekaEvaluation.get(11));
		setMicroF1Score((Double) wekaEvaluation.get(12));
		setMacroPrecision((Double) wekaEvaluation.get(13));
		setMacroRecall((Double) wekaEvaluation.get(14));
		setMacroF1Score((Double) wekaEvaluation.get(15));
		setWeightedPrecision((Double) wekaEvaluation.get(16));
		setWeightedRecall((Double) wekaEvaluation.get(17));
		setWeightedF1Score((Double) wekaEvaluation.get(18));
		
		setMatthewsCorrelationCoefficient((Double) wekaEvaluation.get(19));
		setAreaUnderROC((Double) wekaEvaluation.get(20));
		setAreaUnderPRC((Double) wekaEvaluation.get(21));
		setWeightedMatthewsCorrelation((Double) wekaEvaluation.get(22));
		setWeightedAreaUnderROC((Double) wekaEvaluation.get(23));
		setWeightedAreaUnderPRC((Double) wekaEvaluation.get(24));
		
		setSolutionAttributesIndexName();	// to obtain the selected attribute's index number and attribute name
		
		Double f1 = 0.0, f2 = 0.0, ALPHA = 0.40;
		Double w1 = 0.60, w2 = 0.40;
		if(Driver.classificationMode == "Train-Test")
			f1 = Double.valueOf(fCorrect / Driver.NUMBER_OF_TEST_SAMPLES);
		else if(Driver.classificationMode == "CV") 
			f1 = Double.valueOf(fCorrect / Driver.NUMBER_OF_TRAIN_SAMPLES);

		f2 = Double.valueOf(fnumberofOnes / Driver.SOLUTION_SIZE);

		fitness = Driver.weight1 * f1 - Driver.weight2 * f2;

		return fitness;
	}
	
	public String toString() {
		return Arrays.toString(this.genes);
	}
	
	public String toCVString() {
		return Arrays.toString(this.contextVectors);
	}
	
	public String toCVString(int x) {
		return Arrays.toString(this.contextVectors[x]);
	}

}	// End of Class Individual
