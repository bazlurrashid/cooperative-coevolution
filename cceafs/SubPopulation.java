package cceafs;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import com.google.common.collect.ComparisonChain;
import java.lang.Double; 
public class SubPopulation {
	private Individual[] individuals;
	
	public SubPopulation(int length) {
		individuals = new Individual[length];
	}
	
	public SubPopulation initializeSubPopulation() throws Exception {
		for(int x = 0; x < individuals.length; x++) {
			individuals[x] = new Individual(Driver.NO_OF_GENES).initializeIndividual();
		}
		return this;
	}
	
	public Individual[] getIndividuals() {
		return individuals;
	}
	
	public void sortIndividualesByFitness() throws Exception {	
		Double originalAccuracy = Driver.accuracy;
		Individual temp;
		for(int x = 0; x < individuals.length; x++) {
			for(int y = 0; y < individuals.length; y++) {
				for(int z = 0; z < individuals.length - y - 1; z++) {
					Double d1 = Math.abs(individuals[z].getAccuracy() - originalAccuracy);
					Double d2 = Math.abs(individuals[z+1].getAccuracy() - originalAccuracy);
					if(Double.compare(individuals[z].getFitness(), individuals[z+1].getFitness()) < 0) {
						temp = individuals[z];
						individuals[z] = individuals[z+1];
						individuals[z+1] = temp;
					}
					else if(Double.compare(individuals[z].getFitness(), individuals[z+1].getFitness()) == 0) {
						if(Double.compare(individuals[z].getNumberofOnes(), individuals[z+1].getNumberofOnes()) < 0) {
							temp = individuals[z];
							individuals[z] = individuals[z+1];
							individuals[z+1] = temp;
						}
					}
				}
			}
		}
	}
	

	public void setIndividuals(int x, int[] ind) {
		individuals[x].setGenes(ind);
	}
	
	public void setContextVector(int x, int count, int[] cv) {
		individuals[x].setContextGenes(count, cv);
	}

}	// End of Class SubPopulation

