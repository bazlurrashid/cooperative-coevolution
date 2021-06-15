package fglcc;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

public class fglcc {
	
	public static Double OriginalAccuracy;
	public static final String TRAIN_DATASET = "D:/feature-selection/security-dataset/For_Supervised_Detection/UNSW_NB15_Two_Classes_Randomized.arff";	// Dataset from which feature selection is to be performed
	public static final String csvFileName = "D:/feature-selection/matrix_corr_UNSW.csv";
	public static final String arffFileName = "D:/feature-selection/matrix_corr_UNSW.arff";
	public static final String dsRTFFileName = "D:/feature-selection/matrix_corr_UNSW.rtf";
	public static Instances data;
	public static double[] m_correlations;
	public static int[] m_correlations_index;
	public static double[][] m_correlations_data_index;
	public static double[][] m_corr_data_index_without_sort;
	public static double[][] fitness;
	public static List<Integer> selected_features;
	public static List<Double> selected_features_corr;
	public static Double[][] matrix_corr;
	public static int groups = 14; // number of clusters for k-means clustering
	
	public static FileWriter csvWriter, dsRTFWriter;
	public static String COMMA_DELIMITER = ",", NEW_LINE_SEPARATOR = "\n";	//Delimiters used in CSV file

	public fglcc(){

	}
	
	public static void main(String[] args) throws Exception {	// Start of main() function
		DataSource trainSource = new DataSource(TRAIN_DATASET);
		data = trainSource.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		dsRTFWriter = new FileWriter(dsRTFFileName);
		try {
			dsRTFWriter.append("Dataset Name: " + String.valueOf(TRAIN_DATASET));	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			data.deleteWithMissingClass();
			ReplaceMissingValues rmv = new ReplaceMissingValues();
			rmv.setInputFormat(data);
			data = Filter.useFilter(data, rmv);

			int numClasses = data.classAttribute().numValues();
			int classIndex = data.classIndex();
			int numInstances = data.numInstances();
			dsRTFWriter.append("Number of attributes including class: " + String.valueOf(data.numAttributes()));	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			m_correlations = new double[data.numAttributes()];
			m_correlations_index = new int[data.numAttributes() - 1];
			m_correlations_data_index = new double[data.numAttributes() - 1][2];
			m_corr_data_index_without_sort = new double[data.numAttributes() - 1][2];
			selected_features = new ArrayList<Integer>();
			selected_features_corr = new ArrayList<Double>();
			matrix_corr = new Double[data.numAttributes() - 2][data.numAttributes() - 2];
			fitness = new double[data.numAttributes() - 2][2];;
	    
			List<Integer> numericIndexes = new ArrayList<Integer>();
			List<Integer> nominalIndexes = new ArrayList<Integer>();
			double[][][] nomAtts = new double[data.numAttributes()][][];
			for (int i = 0; i < data.numAttributes(); i++) {
				if (data.attribute(i).isNominal() && i != classIndex) {
					nomAtts[i] = new double[data.attribute(i).numValues()][data.numInstances()];
					Arrays.fill(nomAtts[i][0], 1.0); // set zero index for this att to all
	                                         // 1's
					nominalIndexes.add(i);
				} 
				else if (data.attribute(i).isNumeric() && i != classIndex) {
					numericIndexes.add(i);
				}
			}
			for(int x = 0; x < numericIndexes.size(); x++) {
				m_correlations_index[x] = numericIndexes.get(x);
			}

			int xx = 0;
			for(int x = numericIndexes.size(); x < (numericIndexes.size() + nominalIndexes.size()); x++) {
				m_correlations_index[x] = nominalIndexes.get(xx);
				xx++;
			}
			if (nominalIndexes.size() > 0) {
				for (int i = 0; i < data.numInstances(); i++) {
					Instance current = data.instance(i);
					for (int j = 0; j < current.numValues(); j++) {
						if (current.attribute(current.index(j)).isNominal() && current.index(j) != classIndex) {
						// Will need to check for zero in case this isn't a sparse
						// instance (unless we add 1 and subtract 1)
							nomAtts[current.index(j)][(int) current.valueSparse(j)][i] += 1;
							nomAtts[current.index(j)][0][i] -= 1;
						}
					}
				}
			}
			if (data.classAttribute().isNumeric()) {
				double[] classVals = data.attributeToDoubleArray(classIndex);
				// do the numeric attributes
				for (Integer i : numericIndexes) {
					double[] numAttVals = data.attributeToDoubleArray(i);
					m_correlations[i] = Utils.correlation(numAttVals, classVals, numAttVals.length);
					if (m_correlations[i] == 1.0) {
					// check for zero variance (useless numeric attribute)
						if (Utils.variance(numAttVals) == 0) {
							m_correlations[i] = 0;
						}
					}
				}
				// do the nominal attributes
				if (nominalIndexes.size() > 0) {
				// now compute the correlations for the binarized nominal attributes
					for (Integer i : nominalIndexes) {
					double sum = 0;
					double corr = 0;
					double sumCorr = 0;
					double sumForValue = 0;
					for (int j = 0; j < data.attribute(i).numValues(); j++) {
						sumForValue = Utils.sum(nomAtts[i][j]);
						corr = Utils.correlation(nomAtts[i][j], classVals, classVals.length);
						// useless attribute - all instances have the same value
						if (sumForValue == numInstances || sumForValue == 0) {
							corr = 0;
						}
						if (corr < 0.0) {
							corr = -corr;
						}
						sumCorr += sumForValue * corr;
						sum += sumForValue;
					}
					m_correlations[i] = (sum > 0) ? sumCorr / sum : 0;
					}
				}
			} 
			else {
				// class is nominal
				double[][] binarizedClasses = new double[data.classAttribute().numValues()][data.numInstances()];
				// this is equal to the number of instances for all inst weights = 1
				double[] classValCounts = new double[data.classAttribute().numValues()];
				for (int i = 0; i < data.numInstances(); i++) {
					Instance current = data.instance(i);
					binarizedClasses[(int) current.classValue()][i] = 1;
				}
				for (int i = 0; i < data.classAttribute().numValues(); i++) {
					classValCounts[i] = Utils.sum(binarizedClasses[i]);
				}
				double sumClass = Utils.sum(classValCounts);
				// do numeric attributes first
				if (numericIndexes.size() > 0) {
					for (Integer i : numericIndexes) {
					double[] numAttVals = data.attributeToDoubleArray(i);
					double corr = 0;
					double sumCorr = 0;
					for (int j = 0; j < data.classAttribute().numValues(); j++) {
						corr = Utils.correlation(numAttVals, binarizedClasses[j], numAttVals.length);
						if (corr < 0.0) {
							corr = -corr;
						}
					if (corr == 1.0) {
						// check for zero variance (useless numeric attribute)
						if (Utils.variance(numAttVals) == 0) {
							corr = 0;
						}
					}
					sumCorr += classValCounts[j] * corr;
					}
					m_correlations[i] = sumCorr / sumClass;
					}
				}
				if (nominalIndexes.size() > 0) {
					for (Integer i : nominalIndexes) {
						double sumForAtt = 0;
						double corrForAtt = 0;
						for (int j = 0; j < data.attribute(i).numValues(); j++) {
						double sumForValue = Utils.sum(nomAtts[i][j]);
						double corr = 0;
						double sumCorr = 0;
						double avgCorrForValue = 0;
						sumForAtt += sumForValue;
						for (int k = 0; k < numClasses; k++) {
							// corr between value j and class k
							corr = Utils.correlation(nomAtts[i][j], binarizedClasses[k], binarizedClasses[k].length);
							// useless attribute - all instances have the same value
							if (sumForValue == numInstances || sumForValue == 0) {
								corr = 0;
							}
							if (corr < 0.0) {
								corr = -corr;
							}
							sumCorr += classValCounts[k] * corr;
						}
						avgCorrForValue = sumCorr / sumClass;
						corrForAtt += sumForValue * avgCorrForValue;
						}
						// the weighted average corr for att i as
						// a whole (wighted by value frequencies)
						m_correlations[i] = (sumForAtt > 0) ? corrForAtt / sumForAtt : 0;
					}
				}
			}  
			for (int x = 0; x < m_correlations_index.length; x++){
				m_correlations_data_index[x][0] = m_correlations_index[x]+1;
			}
			for (int x = 0; x < m_correlations_index.length; x++){
				m_correlations_data_index[x][1] = m_correlations[x];
			}
			m_corr_data_index_without_sort = m_correlations_data_index;
			Arrays.sort(m_corr_data_index_without_sort, (a, b) -> Double.compare(a[0], b[0]));
			dsRTFWriter.append("Correlation values of all atrributes in relation to class value");
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			for (int i = 0; i < m_corr_data_index_without_sort.length; i++) {
				for (int j = 0; j < 2; j++) {
					dsRTFWriter.append(String.valueOf(m_corr_data_index_without_sort[i][j] + " "));
				}
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
			}
			System.out.println();
			Arrays.sort(m_correlations_data_index, (a, b) -> Double.compare(b[1], a[1]));
			dsRTFWriter.append(String.valueOf("Sorting based on maximum correlation values with the class"));
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			for (int i = 0; i < m_correlations_index.length; i++) {
				for (int j = 0; j < 2; j++) {
					dsRTFWriter.append(String.valueOf(m_correlations_data_index[i][j] + " "));
				}
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
			}
			selected_features.add((int) m_correlations_data_index[0][0]);
			selected_features_corr.add((double) m_correlations_data_index[0][1]);
			System.out.println("selected_features: " + Arrays.toString(selected_features.toArray()));
			dsRTFWriter.append(String.valueOf("selected_features: " + Arrays.toString(selected_features.toArray())));
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			dsRTFWriter.append(String.valueOf("selected_features_corr: " + Arrays.toString(selected_features_corr.toArray())));
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
		
			String toRemoveSelectedFeature = selected_features.get(0).toString();
			Remove dataRemove = new Remove();
			dataRemove.setInvertSelection(false);
			dataRemove.setAttributeIndices(toRemoveSelectedFeature);
			dataRemove.setInputFormat(data);
			data = Filter.useFilter(data, dataRemove);
			dsRTFWriter.append(String.valueOf("Computing correlation of each attribute with other attributes"));
			dsRTFWriter.append(NEW_LINE_SEPARATOR);
			for(int x = 0; x < (data.numAttributes() - 1); x++) {
				if (data.attribute(x).isNumeric()) {
					double[] classVals = data.attributeToDoubleArray(data.attribute(x).index());
					// do the numeric attributes
					for (Integer i : numericIndexes) {
						double[] numAttVals = data.attributeToDoubleArray(i);
						m_correlations[i] = Utils.correlation(numAttVals, classVals, numAttVals.length);
						if (m_correlations[i] == 1.0) {
							// check for zero variance (useless numeric attribute)
							if (Utils.variance(numAttVals) == 0) {
								m_correlations[i] = 0;
							}
						}
					}
					// do the nominal attributes
					if (nominalIndexes.size() > 0) {
						// now compute the correlations for the binarized nominal attributes
						for (Integer i : nominalIndexes) {
						double sum = 0;
						double corr = 0;
						double sumCorr = 0;
						double sumForValue = 0;
						for (int j = 0; j < data.attribute(i).numValues(); j++) {
							sumForValue = Utils.sum(nomAtts[i][j]);
							corr = Utils.correlation(nomAtts[i][j], classVals, classVals.length);
							// useless attribute - all instances have the same value
							if (sumForValue == numInstances || sumForValue == 0) {
								corr = 0;
							}
							if (corr < 0.0) {
								corr = -corr;
							}
							sumCorr += sumForValue * corr;
							sum += sumForValue;
						}
						m_correlations[i] = (sum > 0) ? sumCorr / sum : 0;
						}
					}
				} 
				else {
					// class is nominal
					double[][] binarizedClasses = new double[data.attribute(x).numValues()][data.numInstances()];
					// this is equal to the number of instances for all inst weights = 1
					double[] classValCounts = new double[data.attribute(x).numValues()];
					for (int i = 0; i < data.numInstances(); i++) {
						Instance current = data.instance(i);
						binarizedClasses[(int) current.classValue()][i] = 1;
					}
					for (int i = 0; i < data.attribute(x).numValues(); i++) {
						classValCounts[i] = Utils.sum(binarizedClasses[i]);
					}
					double sumClass = Utils.sum(classValCounts);
					// do numeric attributes first
					if (numericIndexes.size() > 0) {
						for (Integer i : numericIndexes) {
							double[] numAttVals = data.attributeToDoubleArray(i);
							double corr = 0;
							double sumCorr = 0;
							for (int j = 0; j < data.attribute(x).numValues(); j++) {
								corr = Utils.correlation(numAttVals, binarizedClasses[j], numAttVals.length);
								if (corr < 0.0) {
									corr = -corr;
								}
							if (corr == 1.0) {
								// check for zero variance (useless numeric attribute)
								if (Utils.variance(numAttVals) == 0) {
									corr = 0;
								}
							}
							sumCorr += classValCounts[j] * corr;
							}
							m_correlations[i] = sumCorr / sumClass;
						}
					}
					if (nominalIndexes.size() > 0) {
						for (Integer i : nominalIndexes) {
							double sumForAtt = 0;
							double corrForAtt = 0;
							for (int j = 0; j < data.attribute(i).numValues(); j++) {
								double sumForValue = Utils.sum(nomAtts[i][j]);
								double corr = 0;
								double sumCorr = 0;
								double avgCorrForValue = 0;
								sumForAtt += sumForValue;
								for (int k = 0; k < numClasses; k++) {
									// corr between value j and class k
									corr = Utils.correlation(nomAtts[i][j], binarizedClasses[k], binarizedClasses[k].length);
									// useless attribute - all instances have the same value
									if (sumForValue == numInstances || sumForValue == 0) {
										corr = 0;
									}
									if (corr < 0.0) {
										corr = -corr;
									}
									sumCorr += classValCounts[k] * corr;
								}
								avgCorrForValue = sumCorr / sumClass;
								corrForAtt += sumForValue * avgCorrForValue;
							}
							// the weighted average corr for att i as
							// a whole (wighted by value frequencies)
							m_correlations[i] = (sumForAtt > 0) ? corrForAtt / sumForAtt : 0;
						}
					}
				}
				for(int y = 0; y < (data.numAttributes() - 1); y++) {
					matrix_corr[x][y] = m_correlations[y];
				}
			}
		
			for(int x = 0; x < matrix_corr.length; x++) {
				for(int y = 0; y < matrix_corr.length; y++) {
					dsRTFWriter.append(String.valueOf(matrix_corr[x][y] + "\t"));
				}
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
			}
			csvWriter = new FileWriter(csvFileName);
			try {
				for(int x = 0; x < matrix_corr.length; x++) {
	    			 csvWriter.append(String.valueOf("A" + (x+1)));	
	    			 if(x < matrix_corr.length - 1)
	    				 csvWriter.append(COMMA_DELIMITER);
				}
				csvWriter.append(NEW_LINE_SEPARATOR);
				for(int x = 0; x < matrix_corr.length; x++) {
					for(int y = 0; y < matrix_corr.length; y++) {
						csvWriter.append(String.valueOf(matrix_corr[x][y]));	
						if(y < matrix_corr.length - 1)
							csvWriter.append(COMMA_DELIMITER);
					}
					csvWriter.append(NEW_LINE_SEPARATOR);
				}
				System.out.println("CSV file written successfully at " + csvFileName);
				dsRTFWriter.append("CSV file written successfully at " + csvFileName);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			}	catch (Exception e) {
					System.out.println("Error in CsvFileWriters !!!");
					dsRTFWriter.append("Error in CsvFileWriters !!!");		dsRTFWriter.append(NEW_LINE_SEPARATOR);
					e.printStackTrace();
			}	// end of catch 	
			finally {
				try {
					csvWriter.flush();		
					csvWriter.close();		
				}	catch (IOException e) {
						System.out.println("Error while flushing/closing Writers !!!");
						dsRTFWriter.append("Error while flushing/closing Writers !!!");	dsRTFWriter.append(NEW_LINE_SEPARATOR);
						e.printStackTrace();
				}	// End of 2nd catch block
			}	// End of finally
		
			Instances allData = DataSource.read(csvFileName);
			ArffSaver saver = new ArffSaver();
	        saver.setInstances(allData);
	        saver.setFile(new File(arffFileName));
	        saver.writeBatch();
	        System.out.println ("CSV file has been converted into ARRF file and saved at " + arffFileName);
	        dsRTFWriter.append("CSV file has been converted into ARRF file and saved at " +  arffFileName);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			Instances arffData;
			DataSource arffDataSource = new DataSource(arffFileName);
			arffData = arffDataSource.getDataSet();

			SimpleKMeans kMeans = new SimpleKMeans();
			kMeans.setPreserveInstancesOrder(true);
			kMeans.setNumClusters(groups);
			kMeans.buildClusterer(arffData);
			dsRTFWriter.append(String.valueOf(kMeans));	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			Arrays.sort(m_corr_data_index_without_sort, (a, b) -> Double.compare(a[0], b[0]));
	
			int selFeature = selected_features.get(0);
			double sum_sel_corr = 0.0;
			for(int x = 0; x < groups; x++) {
				for(int y = 0; y < (data.numAttributes() - 1); y++) {
					for(int z = 0; z < selected_features.size(); z++) {
						sum_sel_corr += selected_features_corr.get(z);
					}
					if(y >= selFeature - 1) {
						fitness[y][0] = y + 2;
						fitness[y][1] = m_corr_data_index_without_sort[y+1][1] - 0.3 * (Double.valueOf(kMeans.getClusterCentroids().get(x).toString(y)) + Double.valueOf(sum_sel_corr));
					}
					else {
						fitness[y][0] = y + 1;
						fitness[y][1] = m_corr_data_index_without_sort[y][1] - 0.3 * (Double.valueOf(kMeans.getClusterCentroids().get(x).toString(y)) + Double.valueOf(sum_sel_corr));
					}
					sum_sel_corr = 0.0;
				}
				dsRTFWriter.append(String.valueOf("Before sorting based on maximum fitness values for cluster " + x));
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
				for (int i = 0; i < fitness.length; i++) {
					for (int j = 0; j < 2; j++) {
						dsRTFWriter.append(String.valueOf(fitness[i][j] + " "));
					}
					dsRTFWriter.append(NEW_LINE_SEPARATOR);
				}
				Arrays.sort(fitness, (a, b) -> Double.compare(b[1], a[1]));
				dsRTFWriter.append(String.valueOf("Sorting based on maximum fitness values for cluster " + x));
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
				for (int i = 0; i < fitness.length; i++) {
					for (int j = 0; j < 2; j++) {
						dsRTFWriter.append(String.valueOf(fitness[i][j] + " "));
					}
					dsRTFWriter.append(NEW_LINE_SEPARATOR);
				}
				int check = 0, ind = 0;
				for(int zz = 0; zz < matrix_corr.length; zz++) {
					for(int yy = 0; yy < matrix_corr.length; yy++) {
						if(Double.compare(Double.valueOf(kMeans.getClusterCentroids().get(x).toString(yy)), matrix_corr[zz][yy]) == 0) {
							check = 1;
							ind = yy + 1;
						}
					}
				}
				if(check == 1) {
					selected_features.add(ind);
				}
				else {
					selected_features.add((int) fitness[0][0]);
				}
				int fit = 1; int afit = 0;
				for(int a = 0; a < selected_features.size(); a++) {
					for(int b = 0; b < a; b++) {
						if(a!=b) {
							if(selected_features.get(a) == selected_features.get(b)) {
							int repeat_chk=0;
								for(int m = 1; m < fitness.length; m++) {
									repeat_chk=0;
									for(int n = 0; n < selected_features.size(); n++) {
										if(selected_features.get(n) == fitness[m][0]) {
											repeat_chk=1;
											break;
										}
		    				
									}
									if (repeat_chk==0) {
										afit=m;
										break;
									}
								}
								selected_features.remove(a);
								selected_features.add((int) fitness[afit][0]);
							}
						}
					}
				}
	    
				selected_features_corr.add((double) fitness[0][1]);
				System.out.println("selected_features: " + Arrays.toString(selected_features.toArray()));
				dsRTFWriter.append(String.valueOf("selected_features: " + Arrays.toString(selected_features.toArray())));
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
				dsRTFWriter.append(String.valueOf("Top_fitness: " + Arrays.toString(selected_features_corr.toArray())));
				dsRTFWriter.append(NEW_LINE_SEPARATOR);
			}
      
			Collections.sort(selected_features);
			System.out.println("Final selected_features: " + Arrays.toString(selected_features.toArray()));
			dsRTFWriter.append("Final selected_features: " + Arrays.toString(selected_features.toArray()));	dsRTFWriter.append(NEW_LINE_SEPARATOR);
			System.out.println("RTF file written successfully at " + dsRTFFileName);
			dsRTFWriter.append("RTF file written successfully at " + dsRTFFileName);	dsRTFWriter.append(NEW_LINE_SEPARATOR);
		}	
		catch (Exception e) {
				System.out.println("Error in RTFFileWriters!!!");
				dsRTFWriter.append("Error in RTFFileWriters!!!");	dsRTFWriter.append(NEW_LINE_SEPARATOR);
				e.printStackTrace();
		}	// end of catch 	
	    finally {
			try {
				dsRTFWriter.flush();
				dsRTFWriter.close();
	    	}	
			catch (IOException e) {
				System.out.println("Error while flushing/closing Writers !!!");
	    		dsRTFWriter.append("Error while flushing/closing Writers !!!");	dsRTFWriter.append(NEW_LINE_SEPARATOR);
	    		e.printStackTrace();
	    	}	// End of 2nd catch block
	    }	// End of finally
	}	// End of Main function
	
}
