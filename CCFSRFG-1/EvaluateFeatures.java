package ccfsrfg1;

import java.util.Arrays;
import java.util.List;
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
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class EvaluateFeatures {
	
	private int[] TO_BE_REMOVED;
	private String STR_DELETE_ATTR;	
	private int NUM_DETELE_ATTR;
	private Double precision = 0.0, recall = 0.0, f1Score = 0.0, accuracy = 0.0, sensitivity = 0.0, specificity = 0.0, correctlyClassified = 0.0;
	private String strSummary, strClassDetails, strConfusionMatrix;
	private Double weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score;
	private Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
	
	public EvaluateFeatures(int[] deleteAttributes, int numofDeleteAttributes) {

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
	
	Instances testData = Driver.testData;
	testData.setClassIndex(testData.numAttributes()-1);
	Remove testRemove = new Remove();
	testRemove.setInvertSelection(false);
	testRemove.setAttributeIndices(STR_DELETE_ATTR);
	testRemove.setInputFormat(testData);
	testData = Filter.useFilter(testData, testRemove);
	
	List<Object> wekaEvaluation = null;
	if(Driver.classificationModel == "NB")
		wekaEvaluation = callNaiveBayesClassifier(trainData, testData);			// evaluation using NaiveBayes classifier
	else if(Driver.classificationModel == "SVM")
		wekaEvaluation = callSVMClassifier(trainData, testData);				// evaluation using SVM classifier
	else if(Driver.classificationModel == "J48")
		wekaEvaluation = callJ48Classifier(trainData, testData);				// evaluation using J48 classifier
	else if(Driver.classificationModel == "RF")
		wekaEvaluation = callRandomForestClassifier(trainData, testData);		// evaluation using RandomForest classifier
	else if(Driver.classificationModel == "LR")
		wekaEvaluation = callLogisticRegressionClassifier(trainData, testData);	// evaluation using LogisticRegression classifier
	else if(Driver.classificationModel == "KNN")
		wekaEvaluation = callkNNClassifier(trainData, testData);				// evaluation using k-Nearest Neighbor classifier

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

	matthewsCorrelationCoefficient = (Double) wekaEvaluation.get(19);
	areaUnderROC = (Double) wekaEvaluation.get(20);
	areaUnderPRC = (Double) wekaEvaluation.get(21);
	weightedMatthewsCorrelation = (Double) wekaEvaluation.get(22);
	weightedAreaUnderROC = (Double) wekaEvaluation.get(23);
	weightedAreaUnderPRC = (Double) wekaEvaluation.get(24);

	return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix,
			weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score,
			matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC);
	}	// end of List<Object> CorrectlyClassified()

	public List<Object> callNaiveBayesClassifier(Instances trainData, Instances testData) throws Exception {
		
		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		Evaluation eval = new Evaluation(trainData);
		NaiveBayes nB = new NaiveBayes();
		nB.buildClassifier(trainData);	// build the classifier using train data
		weka.core.SerializationHelper.write(Driver.MODEL_SAVE, nB);	// save the model
		NaiveBayes nB2 = (NaiveBayes) weka.core.SerializationHelper.read(Driver.MODEL_SAVE);	// load the model
		eval.evaluateModel(nB2, testData);	// test the classifier model using test data
		
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
		for(int x = 0; x < testData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / testData.numClasses();	macroRecall = RE / testData.numClasses();	macroF1Score = F1 / testData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
		matthewsCorrelationCoefficient = eval.matthewsCorrelationCoefficient(0);
		areaUnderROC = eval.areaUnderROC(0);
		areaUnderPRC = eval.areaUnderPRC(0);
		weightedMatthewsCorrelation = eval.weightedMatthewsCorrelation();
		weightedAreaUnderROC = eval.weightedAreaUnderROC();
		weightedAreaUnderPRC = eval.weightedAreaUnderPRC();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix,
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score,
				matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC);
	}	// end of callNaiveBayesClassifier(Instances trainData, Instances testData)
	
	public List<Object> callSVMClassifier(Instances trainData, Instances testData) throws Exception {
		
		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		Evaluation eval = new Evaluation(trainData);
		SMO svm = new SMO();
		svm.buildClassifier(trainData);	// build the classifier using train data
		weka.core.SerializationHelper.write(Driver.MODEL_SAVE, svm);	// save the model
		SMO svm2 = (SMO) weka.core.SerializationHelper.read(Driver.MODEL_SAVE);	// load the model
		eval.evaluateModel(svm2, testData);	// test the classifier model using test data
		
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
		for(int x = 0; x < testData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / testData.numClasses();	macroRecall = RE / testData.numClasses();	macroF1Score = F1 / testData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
		matthewsCorrelationCoefficient = eval.matthewsCorrelationCoefficient(0);
		areaUnderROC = eval.areaUnderROC(0);
		areaUnderPRC = eval.areaUnderPRC(0);
		weightedMatthewsCorrelation = eval.weightedMatthewsCorrelation();
		weightedAreaUnderROC = eval.weightedAreaUnderROC();
		weightedAreaUnderPRC = eval.weightedAreaUnderPRC();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix,
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score,
				matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC);
	}	// end of callSVMClassifier(Instances trainData, Instances testData)
	
	public List<Object> callJ48Classifier(Instances trainData, Instances testData) throws Exception {
		
		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		Evaluation eval = new Evaluation(trainData);
		J48 j48 = new J48();
		j48.buildClassifier(trainData);	// build the classifier using train data
		weka.core.SerializationHelper.write(Driver.MODEL_SAVE, j48);	// save the model
		J48 j482 = (J48) weka.core.SerializationHelper.read(Driver.MODEL_SAVE);	// load the model
		eval.evaluateModel(j482, testData);	// test the classifier model using test data
		
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
		for(int x = 0; x < testData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / testData.numClasses();	macroRecall = RE / testData.numClasses();	macroF1Score = F1 / testData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
		matthewsCorrelationCoefficient = eval.matthewsCorrelationCoefficient(0);
		areaUnderROC = eval.areaUnderROC(0);
		areaUnderPRC = eval.areaUnderPRC(0);
		weightedMatthewsCorrelation = eval.weightedMatthewsCorrelation();
		weightedAreaUnderROC = eval.weightedAreaUnderROC();
		weightedAreaUnderPRC = eval.weightedAreaUnderPRC();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix,
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score,
				matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC);
	}	// end of callJ48Classifier(Instances trainData, Instances testData)
	
	public List<Object> callRandomForestClassifier(Instances trainData, Instances testData) throws Exception {

		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		Evaluation eval = new Evaluation(trainData);
		RandomForest rf = new RandomForest();
		rf.buildClassifier(trainData);	// build the classifier using train data
		weka.core.SerializationHelper.write(Driver.MODEL_SAVE, rf);	// save the model
		RandomForest rf2 = (RandomForest) weka.core.SerializationHelper.read(Driver.MODEL_SAVE);	// load the model
		eval.evaluateModel(rf2, testData);	// test the classifier model using test data
		
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
		for(int x = 0; x < testData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / testData.numClasses();	macroRecall = RE / testData.numClasses();	macroF1Score = F1 / testData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
		matthewsCorrelationCoefficient = eval.matthewsCorrelationCoefficient(0);
		areaUnderROC = eval.areaUnderROC(0);
		areaUnderPRC = eval.areaUnderPRC(0);
		weightedMatthewsCorrelation = eval.weightedMatthewsCorrelation();
		weightedAreaUnderROC = eval.weightedAreaUnderROC();
		weightedAreaUnderPRC = eval.weightedAreaUnderPRC();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix,
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score,
				matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC);
	}	// end of callRandomForestClassifier(Instances trainData, Instances testData)
	

	public List<Object> callLogisticRegressionClassifier(Instances trainData, Instances testData) throws Exception {

		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		Evaluation eval = new Evaluation(trainData);
		Logistic logistic = new Logistic();
		logistic.buildClassifier(trainData);	// build the classifier using train data
		weka.core.SerializationHelper.write(Driver.MODEL_SAVE, logistic);	// save the model
		Logistic logistic2 = (Logistic) weka.core.SerializationHelper.read(Driver.MODEL_SAVE);	// load the model
		eval.evaluateModel(logistic2, testData);	// test the classifier model using test data
		
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
		for(int x = 0; x < testData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / testData.numClasses();	macroRecall = RE / testData.numClasses();	macroF1Score = F1 / testData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
		matthewsCorrelationCoefficient = eval.matthewsCorrelationCoefficient(0);
		areaUnderROC = eval.areaUnderROC(0);
		areaUnderPRC = eval.areaUnderPRC(0);
		weightedMatthewsCorrelation = eval.weightedMatthewsCorrelation();
		weightedAreaUnderROC = eval.weightedAreaUnderROC();
		weightedAreaUnderPRC = eval.weightedAreaUnderPRC();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix,
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score,
				matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC);
	}	// end of callLogisticRegressionClassifier(Instances trainData, Instances testData)
	
	public List<Object> callkNNClassifier(Instances trainData, Instances testData) throws Exception {

		Double precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified;
		String strSummary, strClassDetails, strConfusionMatrix;
		
		Evaluation eval = new Evaluation(trainData);
		IBk knn = new IBk();
		knn.buildClassifier(trainData);	// build the classifier using train data
		weka.core.SerializationHelper.write(Driver.MODEL_SAVE, knn);	// save the model
		IBk knn2 = (IBk) weka.core.SerializationHelper.read(Driver.MODEL_SAVE);	// load the model
		eval.evaluateModel(knn2, testData);	// test the classifier model using test data
		
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
		for(int x = 0; x < testData.numClasses(); x++) {
			TP += eval.numTruePositives(x);
			FP += eval.numFalsePositives(x);
			FN += eval.numFalseNegatives(x);
			PRE += eval.precision(x);
			RE += eval.recall(x);
			F1 += eval.fMeasure(x);
		}
		microPrecision = TP / (TP + FP);	microRecall = TP / (TP + FN);	microF1Score = 2 / (1 / microPrecision + 1 / microRecall);
		macroPrecision = PRE / testData.numClasses();	macroRecall = RE / testData.numClasses();	macroF1Score = F1 / testData.numClasses();
		weightedPrecision = eval.weightedPrecision();	weightedRecall = eval.weightedRecall(); weightedF1Score = eval.weightedFMeasure();

		Double matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC;
		matthewsCorrelationCoefficient = eval.matthewsCorrelationCoefficient(0);
		areaUnderROC = eval.areaUnderROC(0);
		areaUnderPRC = eval.areaUnderPRC(0);
		weightedMatthewsCorrelation = eval.weightedMatthewsCorrelation();
		weightedAreaUnderROC = eval.weightedAreaUnderROC();
		weightedAreaUnderPRC = eval.weightedAreaUnderPRC();

		return Arrays.asList(precision, recall, f1Score, accuracy, sensitivity, specificity, correctlyClassified, strSummary, strClassDetails, strConfusionMatrix,
				weightedPrecision, weightedRecall, weightedF1Score, microPrecision, microRecall, microF1Score, macroPrecision, macroRecall, macroF1Score,
				matthewsCorrelationCoefficient, areaUnderROC, areaUnderPRC, weightedMatthewsCorrelation, weightedAreaUnderROC, weightedAreaUnderPRC);
	}	// end of callkNNClassifier(Instances trainData, Instances testData)
	
}	// End of class EvaluateFeatures
