package technion.cs.Scorer;

import technion.cs.*;
import technion.cs.PrivacyAgents.PrivacyAgentBudget;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.Instance;

import java.util.Enumeration;
import java.util.List;
import java.util.LinkedList;
import java.io.Serializable;
import java.io.Reader;
import java.io.FileReader;
import java.math.BigDecimal;

/**
 * User: Arik Friedman
 * Date: 19/10/2009
 * Time: 00:25:57
 */
public class GiniScorer extends AttributeScoreAlgorithm implements Serializable {

       public double Score(Instances data, C45Attribute att) {
              Instances[] splitData = SplitData(data,att);
              double score=0;
              for (int j = 0; j < splitData.length; j++) {
                     // Add up Gini indices of new nodes, weighted according to number of instances in each node
                     // Actually we compute -GiniIndex (we want to minimize GiniIndex, while the returned
                     // score will be maximized)
                     if (splitData[j].numInstances() > 0) {
                            score -= ((double) splitData[j].numInstances())*GiniScore(splitData[j]);
                     }
              }
              return score;
       }

       /**
        * Calculate Gini index of a leaf
        * @param leaf the leaf node for which Gini index should be calculated
        * @return 1-sum(p_i)^2, where p_i is the frequency of class value i
        */
       public double GiniScore(Instances leaf)
       {
              double score=1;
              double [] classCounts = new double[leaf.numClasses()];
              Enumeration instEnum = leaf.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     classCounts[(int) inst.classValue()]++;
              }
              for (int j = 0; j < leaf.numClasses(); j++) {
                     //System.out.print(classCounts[j] + "; ");
                     if (classCounts[j] > 0) {
                            score -= Math.pow(classCounts[j]/leaf.numInstances(),2);
                     }
              }
              return score;
       }

       /**
        * Calculate Gini index of a leaf (partition)
        * @param classCounts the counts of class values in the leaf
        * @return 1-sum(p_i)^2, where p_i is the frequency of class value i
        */
       public double GiniScore(int[] classCounts)
       {
              double score=1;
              double total=0;

              for (int j = 0; j < classCounts.length; j++)
                     total+=classCounts[j];

              for (int j = 0; j < classCounts.length; j++) {
                     //System.out.print(classCounts[j] + "; ");
                     if (classCounts[j] > 0) {
                            score -= Math.pow(classCounts[j]/total,2);
                     }
              }
              return score;
       }

       public double GetSensitivity() {
              return 2;
       }

       public void InitializeMaxNumInstances(int maxNumInstances) {
              // NumMaxInstances is irrelevant to this scorer
       }

       /**
        * Score a binary split (used for evaluation splits on numeric attributes)
        *
        * @param leftDist  class counts in the first partition
        * @param rightDist class counts in the second partition
        * @return a score for the given split
        */
       @Override
       public double Score(int[] leftDist, int[] rightDist) {
              double score=0;              
              double totalLeft=0, totalRight=0;
              for (int count: leftDist)
                     totalLeft+=count;
              for (int count: rightDist)
                                   totalRight+=count;

              score -= totalLeft*GiniScore(leftDist);
              score -= totalRight*GiniScore(rightDist);

              return score;

       }

       /**
        * The main method is used to test the GiniScorer
        * @param args program parameters, unused
        */
       @SuppressWarnings("unchecked")
       public static void main(String[] args)
       {
              final String INPUT_FILE="contact-lenses.arff";
              //final String INPUT_FILE="adult.merged.arff";
              Instances data;
              try {
                     Reader reader = new FileReader(INPUT_FILE);
                     data = new Instances(reader);
              } catch (Exception e) {
                     System.out.println("Failed to read input file " + INPUT_FILE + ", exiting");
                     return;
              }
              data.setClassIndex(data.numAttributes()-1);

              System.out.println("Acquired data from file " + INPUT_FILE);
              System.out.println("Got " + data.numInstances() + " instances with " + data.numAttributes() + " attributes, class attribute is " + data.classAttribute().name());
              AttributeScoreAlgorithm scorer = new GiniScorer();
              Enumeration<Attribute> atts=(Enumeration<Attribute>)data.enumerateAttributes();
              while (atts.hasMoreElements())
              {
                     C45Attribute att=new C45Attribute(atts.nextElement());
                     System.out.println("Score for attribute " + att.WekaAttribute().name() + " is " + scorer.Score(data,att));
              }

              List<C45Attribute> candidateAttributes = new LinkedList<C45Attribute>();
              Enumeration attEnum = data.enumerateAttributes();
              while (attEnum.hasMoreElements())
                     candidateAttributes.add(new C45Attribute((Attribute)attEnum.nextElement()));
              BigDecimal epsilon=new BigDecimal(0.1, DiffPrivacyClassifier.MATH_CONTEXT);
              PrivacyAgent agent = new PrivacyAgentBudget(epsilon);

              PrivateInstances privData = new PrivateInstances(agent,data);
              privData.setDebugMode(true);
              try {
                     C45Attribute res=privData.privateChooseAttribute(new GiniScorer(), candidateAttributes,epsilon);
                     System.out.println("Picked attribute " + res.WekaAttribute().name());
              } catch(Exception e) {System.out.println(e.getMessage());}

       }
}
