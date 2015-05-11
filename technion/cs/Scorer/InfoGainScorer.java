package technion.cs.Scorer;

import technion.cs.*;
import weka.core.Instances;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Utils;

import java.util.Enumeration;
import java.util.List;
import java.util.LinkedList;
import java.io.Serializable;
import java.io.Reader;
import java.io.FileReader;
import java.math.BigDecimal;

import technion.cs.PrivacyAgents.PrivacyAgentBudget;

/**
 * User: Arik Friedman
 * Date: 20/05/2009
 * Time: 00:25:57
 */
public class InfoGainScorer extends AttributeScoreAlgorithm implements Serializable {

       private int m_maxNumInstances=0;

       public double Score(Instances data, C45Attribute att) {
              if (data.numInstances()>m_maxNumInstances)
                     throw new IllegalArgumentException("Number of instances is larger than the allowed limit (" + m_maxNumInstances + ")");
              Instances[] splitData = SplitData(data,att);
              double score=0;
              for (int j = 0; j < splitData.length; j++) {
                     if (splitData[j].numInstances() > 0) {
                            score -= ((double) splitData[j].numInstances() /
                                          (double) data.numInstances()) *computeEntropy(splitData[j]);
                     }
              }
              return score*data.numInstances();
       }

       public double GetSensitivity() {
              return Utils.log2(m_maxNumInstances)+1;
       }

       public void InitializeMaxNumInstances(int maxNumInstances) {
              m_maxNumInstances=maxNumInstances;
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

              double totalLeft=0, totalRight=0;
              for (int count: leftDist)
                     totalLeft+=count;
              for (int count: rightDist)
                                   totalRight+=count;

              if (totalLeft+totalRight>m_maxNumInstances)
                     throw new IllegalArgumentException("Number of instances is larger than the allowed limit (" + m_maxNumInstances + ")");

              double score=0;
              if (totalLeft>0)
                     score -= totalLeft *computeEntropy(leftDist);
              if (totalRight>0)
                     score -= totalRight *computeEntropy(rightDist);
              return score;

       }
       
       private double computeEntropy(int[] classCounts)
       {
              double total=0;
              for (int count:classCounts)
                     total+=count;

              double entropy = 0;
              for (int j = 0; j < classCounts.length; j++) {
                     if (classCounts[j] > 0) {
                            entropy -= classCounts[j] * Utils.log2(classCounts[j]);
                     }
              }
              entropy /= total;
              return entropy + Utils.log2(total);
       }

       private double computeEntropy(Instances data)  {

              double [] classCounts = new double[data.numClasses()];
              Enumeration instEnum = data.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     classCounts[(int) inst.classValue()]++;
              }
              double entropy = 0;
              //System.out.print("(classes: ");
              for (int j = 0; j < data.numClasses(); j++) {
                     //System.out.print(classCounts[j] + "; ");
                     if (classCounts[j] > 0) {
                            entropy -= classCounts[j] * Utils.log2(classCounts[j]);
                     }
              }
              //System.out.println(")");
              entropy /= (double) data.numInstances();
              return entropy + Utils.log2(data.numInstances());
       }

       /**
        * The main method is used to test the InfoGainScorer
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
              AttributeScoreAlgorithm scorer = new InfoGainScorer();
              scorer.InitializeMaxNumInstances((int) Math.pow(2,Math.ceil(Utils.log2(data.numInstances()))));
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
              scorer = new InfoGainScorer();
              scorer.InitializeMaxNumInstances((int) Math.pow(2,Math.ceil(Utils.log2(data.numInstances()))));
              try {
                     C45Attribute res=privData.privateChooseAttribute(scorer, candidateAttributes,epsilon);
                     System.out.println("Picked attribute " + res.WekaAttribute().name());
              } catch(Exception e) { System.out.println(e.getMessage());}

       }
}
