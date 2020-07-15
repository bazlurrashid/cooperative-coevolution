package cceafs;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.neighboursearch.LinearNNSearch;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class EvaluateFeaturesCross {
	
	private int[] TO_BE_REMOVED;
	private String STR_DELETE_ATTR;	// used to remove the features or attributes that are not selected as part of the subset selection
	private int NUM_DETELE_ATTR;
	private Double precision = 0.0;	
	private Double recall = 0.0;
	private Double f1Score = 0.0;
	private Double accuracy = 0.0;
	private Double sensitivity = 0.0;
	private Double specificity = 0.0;
	private Double correctlyClassified = 0.0;
	
	private String strSummary;
	private String strClassDetails;
	private String strConfusionMatrix;
	
	private Double weightedPrecision;
	private Double weightedRecall;
	private Double weightedF1Score;
	private Double microPrecision;
	private Double microRecall;
	private Double microF1Score;
	private Double macroPrecision;
	private Double macroRecall;
	private Double macroF1Score;
	
	public EvaluateFeaturesCross(int[] deleteAttributes, int numofDeleteAttributes) {

		NUM_DETELE_ATTR = numofDeleteAttributes;
		TO_BE_REMOVED = new int[NUM_DETELE_ATTR];
		TO_BE_REMOVED = deleteAttributes;
		STR_DELETE_ATTR = "";
		for(int x = 0; x < NUM_DETELE_ATTR; x++) {
			if(x == NUM_DETELE_ATTR - 1) {
				STR_DELETE_ATTR += String.valueOf(TO_BE_REMOVED[x]);
			}
			else
				STR_DELETE_ATTR += String.valueOf(TO_BE_REMOVED[x]) + ",";
		}
	}	// end of EvaluateFeatures(int[] deleteAttributes, int numofDeleteAttributes)
	
	public List<Object> CorrectlyClassified() throws Exception {
	
//	weka.core.WekaPackageManager.loadPackages(false,true,false);	
	Instances trainData = Driver.trainData;
	trainData.setClassIndex(trainData.numAttributes()-1);
	Remove trainRemove = new Remove();
	trainRemove.setInvertSelection(false);
	trainRemove.setAttributeIndices(STR_DELETE_ATTR);
	trainRemove.setInputFormat(trainData);
	trainData = Filter.useFilter(trainData, trainRemove);
	
	List<Object> wekaEvaluation = null;
	if(Driver.classificationModel == "NB")
		wekaEvaluation = callNaiveBayesClassifier(trainData);			// evaluation using NaiveBayes classifier
	else if(Driver.classificationModel == "SVM")
		wekaEvaluation = callSVMClassifier(trainData);				// evaluation using SVM classifier
	else if(Driver.classificationModel == "J48")
		wekaEvaluation = callJ48Classifier(trainData);				// evaluation using J48 classifier
	else if(Driver.classificationModel == "RF")
		wekaEvaluation = callRandomForestClassifier(trainData);		// evaluation using RandomForest classifier
	else if(Driver.classificationModel == "LR")
		wekaEvaluation = callLogisticRegressionClassifier(trainData);	// evaluation using LogisticRegression classifier
	else if(Driver.classificationModel == "KNN")
		wekaEvaluation = callkNNClassifier(trainData);				// evaluation using k-Nearest Neighbor classifier

	precision = (Double) wekaEvaluation.get(0); 
	recall = (Double) wekaEvaluation.get(1);
	f1Score = (Double) wekaEvaluation.get(2);
	accuracy = (Double) wekaEvaluation.get(3);
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

	return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix, 
			weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score);
	}	// end of List<Object> CorrectlyClassified()

	public List<Object> callNaiveBayesClassifier(Instances trainData) throws Exception {
		
		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		int seed = 1;
		int folds = 10;
		Random rand = new Random(seed);
		Instances crossData = trainData;
		crossData.randomize(rand);
		if (crossData.classAttribute().isNominal())
			crossData.stratify(folds);
		Evaluation eval = new Evaluation(crossData);
		NaiveBayes nB = new NaiveBayes();
	    eval.crossValidateModel(nB, crossData, 10, new Random(seed));
		
		precision = eval.precision(0); 
		recall = eval.recall(0);
		f1Score = eval.fMeasure(0);
		accuracy = eval.pctCorrect();
		sensitivity = eval.numTruePositives(0) / (eval.numTruePositives(0) + eval.numFalseNegatives(0));
		specificity = eval.numTrueNegatives(0) / (eval.numTrueNegatives(0) + eval.numFalsePositives(0));
		correctlyClassified = eval.correct();
		strSummary = eval.toSummaryString();
		strClassDetails = eval.toClassDetailsString("====== Detailed Accuracy By Class ======");
		strConfusionMatrix = eval.toMatrixString("====== Confusion Matrix ======");
		
		Double weightedPrecision, weightedRecall, weightedF1Score;
		Double microPrecision, microRecall, microF1Score;
		Double macroPrecision, macroRecall, macroF1Score;
		Double TP = 0.0, FP =0.0, FN = 0.0, TN = 0.0, PRE = 0.0, RE = 0.0, F1 = 0.0;
		for(int x = 0; x < crossData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / crossData.numClasses();	macroRecall = RE / crossData.numClasses();	macroF1Score = F1 / crossData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix, 
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score);
	}	// end of callNaiveBayesClassifier(Instances trainData, Instances testData)
	
	public List<Object> callSVMClassifier(Instances trainData) throws Exception {
		
		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		int seed = 1;
		int folds = 10;
		Random rand = new Random(seed);
		Instances crossData = trainData;
		crossData.randomize(rand);
		if (crossData.classAttribute().isNominal())
			crossData.stratify(folds);
		Evaluation eval = new Evaluation(crossData);
		SMO svm = new SMO();
		eval.crossValidateModel(svm, crossData, 10, new Random(seed));
		
		precision = eval.precision(0); 
		recall = eval.recall(0);
		f1Score = eval.fMeasure(0);
		accuracy = eval.pctCorrect();
		sensitivity = eval.numTruePositives(0) / (eval.numTruePositives(0) + eval.numFalseNegatives(0));
		specificity = eval.numTrueNegatives(0) / (eval.numTrueNegatives(0) + eval.numFalsePositives(0));
		correctlyClassified = eval.correct();
		strSummary = eval.toSummaryString();
		strClassDetails = eval.toClassDetailsString("====== Detailed Accuracy By Class ======");
		strConfusionMatrix = eval.toMatrixString("====== Confusion Matrix ======");
		
		Double weightedPrecision, weightedRecall, weightedF1Score;
		Double microPrecision, microRecall, microF1Score;
		Double macroPrecision, macroRecall, macroF1Score;
		Double TP = 0.0, FP =0.0, FN = 0.0, TN = 0.0, PRE = 0.0, RE = 0.0, F1 = 0.0;
		for(int x = 0; x < crossData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / crossData.numClasses();	macroRecall = RE / crossData.numClasses();	macroF1Score = F1 / crossData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix, 
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score);
	}	// end of callSVMClassifier(Instances trainData, Instances testData)
	
	public List<Object> callJ48Classifier(Instances trainData) throws Exception {
		
		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		int seed = 1;
		int folds = 10;
		Random rand = new Random(seed);
		Instances crossData = trainData;
		crossData.randomize(rand);
		if (crossData.classAttribute().isNominal())
			crossData.stratify(folds);
		Evaluation eval = new Evaluation(trainData);
		J48 j48 = new J48();
		eval.crossValidateModel(j48, crossData, 10, new Random(seed));
		
		precision = eval.precision(0); 
		recall = eval.recall(0);
		f1Score = eval.fMeasure(0);
		accuracy = eval.pctCorrect();
		sensitivity = eval.numTruePositives(0) / (eval.numTruePositives(0) + eval.numFalseNegatives(0));
		specificity = eval.numTrueNegatives(0) / (eval.numTrueNegatives(0) + eval.numFalsePositives(0));
		correctlyClassified = eval.correct();
		strSummary = eval.toSummaryString();
		strClassDetails = eval.toClassDetailsString("====== Detailed Accuracy By Class ======");
		strConfusionMatrix = eval.toMatrixString("====== Confusion Matrix ======");
		
		Double weightedPrecision, weightedRecall, weightedF1Score;
		Double microPrecision, microRecall, microF1Score;
		Double macroPrecision, macroRecall, macroF1Score;
		Double TP = 0.0, FP =0.0, FN = 0.0, TN = 0.0, PRE = 0.0, RE = 0.0, F1 = 0.0;
		for(int x = 0; x < crossData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / crossData.numClasses();	macroRecall = RE / crossData.numClasses();	macroF1Score = F1 / crossData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix, 
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score);
	}	// end of callJ48Classifier(Instances trainData, Instances testData)
	
	public List<Object> callRandomForestClassifier(Instances trainData) throws Exception {

		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		int seed = 1;
		int folds = 10;
		Random rand = new Random(seed);
		Instances crossData = trainData;
		crossData.randomize(rand);
		if (crossData.classAttribute().isNominal())
			crossData.stratify(folds);
		Evaluation eval = new Evaluation(trainData);
		RandomForest rF = new RandomForest();
		eval.crossValidateModel(rF, crossData, folds, new Random(seed));
		
		precision = eval.precision(0); 
		recall = eval.recall(0);
		f1Score = eval.fMeasure(0);
		accuracy = eval.pctCorrect();
		sensitivity = eval.numTruePositives(0) / (eval.numTruePositives(0) + eval.numFalseNegatives(0));
		specificity = eval.numTrueNegatives(0) / (eval.numTrueNegatives(0) + eval.numFalsePositives(0));
		correctlyClassified = eval.correct();
		strSummary = eval.toSummaryString();
		strClassDetails = eval.toClassDetailsString("====== Detailed Accuracy By Class ======");
		strConfusionMatrix = eval.toMatrixString("====== Confusion Matrix ======");
		
		Double weightedPrecision, weightedRecall, weightedF1Score;
		Double microPrecision, microRecall, microF1Score;
		Double macroPrecision, macroRecall, macroF1Score;
		Double TP = 0.0, FP =0.0, FN = 0.0, TN = 0.0, PRE = 0.0, RE = 0.0, F1 = 0.0;
		for(int x = 0; x < crossData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / crossData.numClasses();	macroRecall = RE / crossData.numClasses();	macroF1Score = F1 / crossData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix, 
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score);
	}	// end of callRandomForestClassifier(Instances trainData, Instances testData)
	

	public List<Object> callLogisticRegressionClassifier(Instances trainData) throws Exception {

		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		int seed = 1;
		int folds = 10;
		Random rand = new Random(seed);
		Instances crossData = trainData;
		crossData.randomize(rand);
		if (crossData.classAttribute().isNominal())
			crossData.stratify(folds);
		Evaluation eval = new Evaluation(trainData);
		Logistic logistic = new Logistic();
		eval.crossValidateModel(logistic, crossData, folds, new Random(seed));
		
		precision = eval.precision(0); 
		recall = eval.recall(0);
		f1Score = eval.fMeasure(0);
		accuracy = eval.pctCorrect();
		sensitivity = eval.numTruePositives(0) / (eval.numTruePositives(0) + eval.numFalseNegatives(0));
		specificity = eval.numTrueNegatives(0) / (eval.numTrueNegatives(0) + eval.numFalsePositives(0));
		correctlyClassified = eval.correct();
		strSummary = eval.toSummaryString();
		strClassDetails = eval.toClassDetailsString("====== Detailed Accuracy By Class ======");
		strConfusionMatrix = eval.toMatrixString("====== Confusion Matrix ======");
		
		Double weightedPrecision, weightedRecall, weightedF1Score;
		Double microPrecision, microRecall, microF1Score;
		Double macroPrecision, macroRecall, macroF1Score;
		Double TP = 0.0, FP =0.0, FN = 0.0, TN = 0.0, PRE = 0.0, RE = 0.0, F1 = 0.0;
		for(int x = 0; x < crossData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / crossData.numClasses();	macroRecall = RE / crossData.numClasses();	macroF1Score = F1 / crossData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix, 
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score);
	}	// end of callLogisticRegressionClassifier(Instances trainData, Instances testData)
	
	public List<Object> callkNNClassifier(Instances trainData) throws Exception {

		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		int seed = 1;
		int folds = 10;
		Random rand = new Random(seed);
		Instances crossData = trainData;
		crossData.randomize(rand);
		if (crossData.classAttribute().isNominal())
			crossData.stratify(folds);
		Evaluation eval = new Evaluation(trainData);
		IBk knn = new IBk();
		eval.crossValidateModel(knn, crossData, folds, new Random(seed));
		
		precision = eval.precision(0); 
		recall = eval.recall(0);
		f1Score = eval.fMeasure(0);
		accuracy = eval.pctCorrect();
		sensitivity = eval.numTruePositives(0) / (eval.numTruePositives(0) + eval.numFalseNegatives(0));
		specificity = eval.numTrueNegatives(0) / (eval.numTrueNegatives(0) + eval.numFalsePositives(0));
		correctlyClassified = eval.correct();
		strSummary = eval.toSummaryString();
		strClassDetails = eval.toClassDetailsString("====== Detailed Accuracy By Class ======");
		strConfusionMatrix = eval.toMatrixString("====== Confusion Matrix ======");
		
		Double weightedPrecision, weightedRecall, weightedF1Score;
		Double microPrecision, microRecall, microF1Score;
		Double macroPrecision, macroRecall, macroF1Score;
		Double TP = 0.0, FP =0.0, FN = 0.0, TN = 0.0, PRE = 0.0, RE = 0.0, F1 = 0.0;
		for(int x = 0; x < crossData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / crossData.numClasses();	macroRecall = RE / crossData.numClasses();	macroF1Score = F1 / crossData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix, 
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score);
	}	// end of callkNNClassifier(Instances trainData, Instances testData)
	
}	// End of class EvaluateFeatures
