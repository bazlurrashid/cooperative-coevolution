package ccfsrfg2;

import java.util.Arrays;
import java.util.Random;

public class ContextVector {
	
	private Individual individuals;
	private Individual contextVector;
	
	public ContextVector() {

	}
	
	public ContextVector findRandomCollaborators(SubPopulation[] subPopulation) {
		
		int count = 0;
		for(int x = 0; x < Driver.NO_OF_SUBPOPULATION; x++) {	
			for(int y = 0; y < Driver.SUBPOP_SIZE; y++) {
				individuals = subPopulation[x].getIndividuals()[y];
				subPopulation[x].setIndividuals(y, individuals.getGenes(), individuals.getAttributeIndices());
				for(int z = 0; z < Driver.NO_OF_SUBPOPULATION; z++) {
					if(z != x) {
						int randomInt = new Random().nextInt(Driver.SUBPOP_SIZE);
						contextVector = subPopulation[z].getIndividuals()[randomInt];
						subPopulation[x].setContextVector(y, count, contextVector.getGenes(), contextVector.getAttributeIndices());
						count++;
					}
				}
				count = 0;
			}
		}
		count = 0;
		return this;
	}
	
	public ContextVector findRandomCollaboratorsGenerations(SubPopulation[] subPopulation) {
		int Min = 1, Max = Driver.SUBPOP_SIZE - 1;
		int count = 0;
		for(int x = 0; x < Driver.NO_OF_SUBPOPULATION; x++) {	
			individuals = subPopulation[x].getIndividuals()[0];
			subPopulation[x].setIndividuals(0, individuals.getGenes(), individuals.getAttributeIndices());
			for(int xx = 0; xx < Driver.NO_OF_SUBPOPULATION; xx++) {
				if(xx != x) {
					contextVector = subPopulation[xx].getIndividuals()[0];
					subPopulation[x].setContextVector(0, count, contextVector.getGenes(), contextVector.getAttributeIndices());
					count++;
				}
			}
			count = 0;
			for(int y = 1; y < Driver.SUBPOP_SIZE; y++) {
				individuals = subPopulation[x].getIndividuals()[y];
				subPopulation[x].setIndividuals(y, individuals.getGenes(), individuals.getAttributeIndices());
				for(int z = 0; z < Driver.NO_OF_SUBPOPULATION; z++) {
					if(z != x) {
						int SI = Min + (int)(Math.random() * ((Max - Min) + 1));
						contextVector = subPopulation[z].getIndividuals()[SI];
						subPopulation[x].setContextVector(y, count, contextVector.getGenes(), contextVector.getAttributeIndices());
						count++;
					}
				}
				count = 0;
			}
		}
		count = 0;
		return this;
	}
	
	public ContextVector findBestCollaborators(SubPopulation[] subPopulation) {
		
		int count = 0;
		for(int x = 0; x < Driver.NO_OF_SUBPOPULATION; x++) {	
			for(int y = 0; y < Driver.SUBPOP_SIZE; y++) {
				individuals = subPopulation[x].getIndividuals()[y];
				subPopulation[x].setIndividuals(y, individuals.getGenes(), individuals.getAttributeIndices());
				for(int z = 0; z < Driver.NO_OF_SUBPOPULATION; z++) {
					if(z != x) {
						contextVector = subPopulation[z].getIndividuals()[0];
						subPopulation[x].setContextVector(y, count, contextVector.getGenes(), contextVector.getAttributeIndices());
						count++;
					}
				}
				count = 0;
			}
		}
		count = 0;
		return this;
	}

}	// End of Class Collaboration