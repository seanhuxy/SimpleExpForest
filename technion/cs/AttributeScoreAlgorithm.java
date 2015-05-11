package technion.cs;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;

import java.util.Enumeration;

/**
 * AttributeScoreAlgorithm is defined as abstract class
 * rather than an Interface, to allow integration into
 * Weka's mechanism for choosing a subclass from the GUI
 * User: Arik Friedman
 * Date: 20/05/2009 
 */
public abstract class AttributeScoreAlgorithm {
       /**
        * Score a split of the data with the given attribute
        * @param data the dataset according to which scores are calculated
        * @param att the attribute for which the score is calculated
        * @return a score for splitting the given data with the given attribute
        */
       public abstract double Score(Instances data,C45Attribute att);

       /**
        * Get the sensitivity of the scorer
        * The sensitivity is the maximal change possible in a score
        * when a single data point in added to or removed from a dataset
        * @return scorer sensitivity
        */
       public abstract double GetSensitivity();

       /**
        * Provide an upper bound on the number of instances in a dataset
        * Used by information gain scorer to set the sensitivity
        * @param num the maximal allowed number of instances
        */
       public abstract void InitializeMaxNumInstances(int num);

       /**
        * Score a binary split (used for evaluation splits on numeric attributes)
        * @param leftDist class counts in the first partition
        * @param rightDist class counts in the second partition
        * @return a score for the given split
        */
       public abstract double Score(int[] leftDist, int[] rightDist);

       protected static Instances[] SplitData(Instances data, C45Attribute att) {
              if (att.isNumeric())
                     return SplitDataNumeric(data,att);

              Instances[] splitData = new Instances[att.numValues()];
              for (int j = 0; j < splitData.length; j++) {
                     splitData[j] = new Instances(data, data.numInstances());
              }
              Enumeration instEnum = data.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     splitData[(int) inst.value(att.WekaAttribute())].add(inst);
              }
              for (Instances aSplitData : splitData) {
                     aSplitData.compactify();
              }
              return splitData;
       }

       protected static Instances[] SplitDataNumeric(Instances data, C45Attribute att)
       {
              double splitPoint= att.getSplitPoint();
              Instances[] splitData = new Instances[2];
              splitData[0] = new Instances(data, data.numInstances());
              splitData[1] = new Instances(data, data.numInstances());

              Enumeration instEnum = data.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     if (inst.value(att.WekaAttribute())<splitPoint)
                            splitData[0].add(inst);
                     else
                            splitData[1].add(inst);
              }
              for (Instances aSplitData : splitData) {
                     aSplitData.compactify();
              }
              return splitData;

       }
}
