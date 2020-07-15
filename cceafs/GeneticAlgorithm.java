package cceafs;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class GeneticAlgorithm 
{
	public GeneticAlgorithm() throws IOException{

	}

	public SubPopulation evolve(SubPopulation subPopulation, Individual ind) throws Exception {
		return mutateSubPopulation(crossoverSubPopulation(subPopulation));
	}
	
	private SubPopulation crossoverSubPopulation(SubPopulation subPopulation) throws Exception {
		SubPopulation crossoverSubPopulation = new SubPopulation(subPopulation.getIndividuals().length);
		for(int x = 0; x < Driver.NUMB_OF_ELITE_CHROMOSOMES; x++) {
			crossoverSubPopulation.getIndividuals()[x] = subPopulation.getIndividuals()[x];
		}
		
		for(int x = Driver.NUMB_OF_ELITE_CHROMOSOMES; x < subPopulation.getIndividuals().length; x++) {
			Individual individual1 = selectTournamentSubPopulation(subPopulation).getIndividuals()[0];
			Individual individual2 = selectTournamentSubPopulation(subPopulation).getIndividuals()[0];
			crossoverSubPopulation.getIndividuals()[x] = crossoverIndividual(individual1, individual2);
		}
		return crossoverSubPopulation;
	}
	
	private Individual crossoverIndividual(Individual individual1, Individual individual2) throws Exception {
		Individual crossoverIndividual = new Individual(Driver.NO_OF_GENES);
		// Performing single-point crossover
		Random random = new Random();
		int crossOverPoint = random.nextInt(individual1.getGenes().length);
/		for(int x = 0; x < individual1.getGenes().length; x++) {
			if(x >= crossOverPoint) {
				crossoverIndividual.setGene(x, individual1.getGenes()[x]);
			}
			else
				crossoverIndividual.setGene(x, individual2.getGenes()[x]);
		}
		return crossoverIndividual;
	}
	
	private SubPopulation mutateSubPopulation(SubPopulation subPopulation) throws Exception {
		SubPopulation mutateSubPopulation = new SubPopulation(subPopulation.getIndividuals().length);
		for(int x = 0; x < Driver.NUMB_OF_ELITE_CHROMOSOMES; x++) {
			mutateSubPopulation.getIndividuals()[x] = subPopulation.getIndividuals()[x];
		}
		
		for(int x = Driver.NUMB_OF_ELITE_CHROMOSOMES; x < subPopulation.getIndividuals().length; x++) {
			mutateSubPopulation.getIndividuals()[x] = mutateIndividual(subPopulation.getIndividuals()[x]);
		}
		return mutateSubPopulation;
	}
	
	private Individual mutateIndividual(Individual individual) throws Exception {
		Individual mutateIndividual = new Individual(Driver.NO_OF_GENES);
		for(int x = 0; x < individual.getGenes().length; x++) {
			Random random = new Random();
			Double rand = Math.random();
			if(rand < Driver.MUTATION_RATE) {
				int mutationPoint = random.nextInt(individual.getGenes().length);
					if(individual.getGenes()[mutationPoint] == 1)
						mutateIndividual.setGene(mutationPoint, 0);
					else
						mutateIndividual.setGene(mutationPoint, 1);
			}
			else
				mutateIndividual.setGene(x, individual.getGenes()[x]);
		}
		
		return mutateIndividual;
	}
	
	private SubPopulation selectTournamentSubPopulation(SubPopulation subPopulation) throws Exception {
		SubPopulation tournamentSubPopulation = new SubPopulation(Driver.TOURNAMENT_SELECTION_SIZE);
		for(int x = 0; x < Driver.TOURNAMENT_SELECTION_SIZE; x++) {
			tournamentSubPopulation.getIndividuals()[x] = subPopulation.getIndividuals()[(int)(Math.random() * subPopulation.getIndividuals().length)];
		}
		tournamentSubPopulation.sortIndividualesByFitness();
		return tournamentSubPopulation;
	}

}	// End of Class GeneticAlgorithm


