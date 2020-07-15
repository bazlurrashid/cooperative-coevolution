package cceafs;

public class SortBestIndividualesByFitness {
	private Individual[] toSortIndividual;
	public SortBestIndividualesByFitness(Individual[] ind) {
		toSortIndividual = ind;
	}
	public Individual[] sortIndividuales() {
		Double originalAccuracy = Driver.accuracy;
		Individual temp;
		for(int x = 0; x < toSortIndividual.length; x++) {
			for(int y = 0; y < toSortIndividual.length; y++) {
				for(int z = 0; z < toSortIndividual.length - y - 1; z++) {
					Double d1 = Math.abs(toSortIndividual[z].getAccuracy() - originalAccuracy);
					Double d2 = Math.abs(toSortIndividual[z+1].getAccuracy() - originalAccuracy);
					if(Double.compare(toSortIndividual[z].getFitness(), toSortIndividual[z+1].getFitness()) < 0) {
						temp = toSortIndividual[z];
						toSortIndividual[z] = toSortIndividual[z+1];
						toSortIndividual[z+1] = temp;
					}
					else if(Double.compare(toSortIndividual[z].getFitness(), toSortIndividual[z+1].getFitness()) == 0) {
						if(Double.compare(toSortIndividual[z].getNumberofOnes(), toSortIndividual[z+1].getNumberofOnes()) < 0) {
							temp = toSortIndividual[z];
							toSortIndividual[z] = toSortIndividual[z+1];
							toSortIndividual[z+1] = temp;
						}
					}
				}
			}
		}
		return toSortIndividual;
	}
}
