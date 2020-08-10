package ccfsrfg2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import com.google.common.collect.Lists;

public class RandomGrouping {

	private int numberofGroups;
	private int[] sizeofEachGroup;
	
	public RandomGrouping() {

        for (int x = 0; x < Driver.SOLUTION_SIZE; x++) {
        	Driver.attributIndices.add(x);
        }
        
        Driver.groupedIndices = Lists.partition(Driver.attributIndices, Driver.NO_OF_GENES);
		numberofGroups = Driver.groupedIndices.size();
		sizeofEachGroup = new int[numberofGroups];
		int y = 0;
		for(List<Integer> sublist: Driver.groupedIndices) {
			sizeofEachGroup[y] = sublist.size();
			y++;
		}
	}
	
	public RandomGrouping randomizeAttributeIndices() {

		Collections.shuffle(Driver.attributIndices, new Random(System.nanoTime()));
		Driver.groupedIndices = Lists.partition(Driver.attributIndices, Driver.NO_OF_GENES);

		return this;
	}
	
	public RandomGrouping randomizeAttributeSubIndices() {
		
		Collections.shuffle(Driver.attributIndices, new Random(System.nanoTime()));
		Driver.groupedIndices = Lists.partition(Driver.attributIndices, Driver.NO_OF_GENES);

		for (List<Integer> sublist: Driver.groupedIndices) {
        	Collections.shuffle(sublist);
        }
		
		return this;
	}
	
	public List<List<Integer> > getGroupedIndices() {
		return Driver.groupedIndices;
	}
	
	public int getNumberofGroups() {
		return numberofGroups;
	}
	
	public int[] getSizeofEachGroup() {
		return sizeofEachGroup;
	}
}
