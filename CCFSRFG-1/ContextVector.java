package ccfsrfg1;

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