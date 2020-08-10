package ccfsrfg1;

import java.lang.Double; 
public class SubPopulation {
	private Individual[] individuals;
	private int NO_OF_GENES;	// local variable to hold the size of each subpopulation and is not equivalent to the Driver.NO_OF_GENES
	
	public SubPopulation(int length) {
		individuals = new Individual[length];
	}
	
	public SubPopulation initializeSubPopulation(int subPopNumber, int sizeofEachSubPop) throws Exception {
		NO_OF_GENES = sizeofEachSubPop;
		for(int x = 0; x < individuals.length; x++) {
			individuals[x] = new Individual(NO_OF_GENES).initializeIndividual(subPopNumber);
		}
		return this;
	}
	
	public Individual[] getIndividuals() {
		return individuals;
	}
	
	public void sortIndividualesByFitness() throws Exception {	// Fitness sorting based on classification accuracy only
		Individual temp;
		for(int x = 0; x < individuals.length; x++) {
			for(int y = 0; y < individuals.length; y++) {
				for(int z = 0; z < individuals.length - y - 1; z++) {
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
	
	public void setIndividuals(int x, int[] ind, int[] att) {
		individuals[x].setGenes(ind, att);
	}
	
	public void setContextVector(int x, int count, int[] cv, int[] cvatt) {
		individuals[x].setContextGenes(count, cv);
		individuals[x].setContextIndices(count, cvatt);
	}

}	// End of Class SubPopulation

