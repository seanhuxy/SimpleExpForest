package technion.cs;

import technion.cs.PrivacyAgents.PrivacyAgentBudget;
import technion.cs.Scorer.GiniScorer;
import technion.cs.Scorer.InfoGainScorer;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.*;
import java.math.BigDecimal;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Scanner;

/**
 * An attribute used for constructing differential-private C4.5 trees
 * Author: Arik Friedman
 * Date: 22/01/2010
 */
public class C45Attribute implements Serializable {

       private Attribute m_Attribute;// the encapsulated attribute
       private double m_LowerBound;
       private double m_UpperBound;
       private double m_SplitPoint;       // relevant only for numeric attributes

       public C45Attribute(Attribute att)
       {
              m_Attribute=att;
              if (m_Attribute.isNumeric())
              {
                     m_LowerBound=att.getLowerNumericBound();
                     m_UpperBound=att.getUpperNumericBound();
              }
              else
              {
                     m_LowerBound=0;
                     m_UpperBound=m_Attribute.numValues()-1;
              }
       }

       public C45Attribute(Attribute att, double lowerBound, double upperBound)
       {
              m_Attribute=att;
              m_LowerBound=lowerBound;
              m_UpperBound=upperBound;
       }

       public C45Attribute(C45Attribute att, double lowerBound, double upperBound)
       {
              m_Attribute=att.m_Attribute;
              m_LowerBound=lowerBound;
              m_UpperBound=upperBound;
       }

       public C45Attribute(Attribute att, String configFile)
       {
              m_Attribute=att;
              Scanner scanner=null;
              boolean found=false;
              try {
                     scanner=new Scanner(new File(configFile));
                     while (scanner.hasNext())
                     {
                            String attributeName=scanner.next();
                            Double lowerBound=scanner.nextDouble();
                            Double upperBound=scanner.nextDouble();
                            if (attributeName.equals(m_Attribute.name()))
                            {
                                   m_LowerBound=lowerBound;
                                   m_UpperBound=upperBound;
                                   found=true;
                                   break;
                            }
                     }
              } catch (FileNotFoundException e) {
                     System.out.println("Could not find file " + configFile + " : " + e.getMessage());
              }
              finally {
                     if (scanner!=null)
                            scanner.close();

                     if (!found)
                            throw new IllegalArgumentException("Attribute Bounds were not initialized");
              }
       }


       /**
        * Return the number of values (i.e., number of resulting partitions when splitting)
        * for the attribute
        * @return number of split values
        */
       public int numValues()
       {
              if (isNumeric())     // we use binary splits with numeric attributes
                            return 2;
              return m_Attribute.numValues();
       }

       public void InitBounds()
       {
              // open definition file and obtain lower and upper bounds
       }

       public double lowerBound()
       {
              return m_LowerBound;
       }

       public double upperBound()
       {
              return m_UpperBound;
       }

       public boolean isNumeric()
       {
              return m_Attribute.isNumeric();
       }

       public void setSplitPoint(double splitPoint)
       {
              m_SplitPoint=splitPoint;
       }

       public double getSplitPoint()
       {
              return m_SplitPoint;
       }



/**
        * This method returns an array of split points of an attribute,
        * with the score of each split point according to a given scorer
        * For each split point, there are three numbers given:
        *     (lower bound (lb), upper bound (ub) , score (s))
        * This should be interpreted as: for     lb<= x < ub,   score =s
        *
        * The C4.5 limitation on minimal number of objects in each partition
        * is not enforced here (the evaluation will be inaccurate anyway)
        *
        * Missing values are ignored in this function
        *
        * @param data   the data according to which the splits are calculated
        * @param att the attribute used for splitting
        * @param scorer the scorer used to grade the splits
        * @return a double array containing pairs of split point and the respective score
        */
       public static double[][] GetSplitPoints(Instances data, C45Attribute att, AttributeScoreAlgorithm scorer)
       {
              data.sort(att.m_Attribute);

              // keep track of class distribution for values above and
              // below split point, and get the resulting score for each such
              // split point
              int [] lowClassCounts = new int[data.numClasses()];
              int [] highClassCounts = new int[data.numClasses()];

              // Initialize class distribution and known value weights
              Enumeration enu = data.enumerateInstances();
              while (enu.hasMoreElements())
              {
                     Instance inst=(Instance) enu.nextElement();
                     highClassCounts[(int) inst.classValue()]++;
              }

              double[][] results = new double[data.numInstances()+1][3];
              // Move instances from the high class counts to the low class counts
              double prevValue=att.lowerBound();
              enu = data.enumerateInstances();
              int i=0;
              while (enu.hasMoreElements())
              {
                     Instance inst = (Instance) enu.nextElement();

                     double currValue=inst.value(att.m_Attribute);
                     // this result will provide the score for (prevValue<= att < currValue)
                     //PrintCounts(lowClassCounts,highClassCounts);
                     results[i][0]=prevValue;
                     results[i][1]=currValue;
                     results[i][2]=scorer.Score(lowClassCounts,highClassCounts);

                     // if a new instance has the same value as the former one,
                     // the next split point will override the current split point
                     if (prevValue==currValue)
                            i--;
                     highClassCounts[(int) inst.classValue()]--;
                     lowClassCounts[(int) inst.classValue()]++;

                     i++;
                     prevValue=currValue;
              }

              //PrintCounts(lowClassCounts,highClassCounts);
              results[i][0]=prevValue;
              results[i][1]=att.upperBound();            // this result will provide the score for (prevValue<= att < infinity)
              results[i][2]=scorer.Score(lowClassCounts,highClassCounts);

              // return only a subset containing the actual split points
              return Arrays.copyOf(results,i+1);
       }


       public Attribute WekaAttribute()
       {
              return m_Attribute;
       }

       private static void PrintCounts(int[] lowCounts, int[] highCounts)
       {
              System.out.print("left: (");
              for (int i=0;i<lowCounts.length;i++)
                     System.out.print(lowCounts[i] + ";");

              System.out.print(") \tright (");
              for (int i=0;i<highCounts.length;i++)
                     System.out.print(highCounts[i] + ";");
              System.out.println(")");
       }

       private static void TestGetSplitPoints(String dataFile, String attributeName)
       {
              try {
                     Instances trainInstances=new Instances(new BufferedReader(new FileReader(dataFile)));
                     trainInstances.setClassIndex(trainInstances.numAttributes()-1);
                     AttributeScoreAlgorithm scorer = new InfoGainScorer();
                     scorer.InitializeMaxNumInstances(trainInstances.numInstances());
                     double[][] results = GetSplitPoints(trainInstances,new C45Attribute(trainInstances.attribute(attributeName)),scorer);
                     System.out.println("Results retrieved");
                     for (int i=0;i<results.length;i++)
                            System.out.println("Split point: " + results[i][0] + " <= x < " + results[i][1] + " - score: " + results[i][2]);

              } catch (Exception e)
              {
                     System.out.println(e.getMessage());
              }
       }

       private static void TestPrivateGetSplitPoint(String dataFile, String attributeName)
       {
              try {
                     BigDecimal epsilon = new BigDecimal(10);
                     Instances trainInstances=new Instances(new BufferedReader(new FileReader(dataFile)));
                     trainInstances.setClassIndex(trainInstances.numAttributes()-1);
                     PrivateInstances pInst = new PrivateInstances(new PrivacyAgentBudget(epsilon),trainInstances);
                     pInst.setDebugMode(true);
                     AttributeScoreAlgorithm scorer = new GiniScorer();
                     scorer.InitializeMaxNumInstances(trainInstances.numInstances());
                     double splitPoint=pInst.privateChooseNumericSplitPoint(new C45Attribute(pInst.attribute(attributeName)),epsilon,scorer);
                     System.out.println("Chose split point " + splitPoint);
              } catch (Exception e)
              {
                     System.out.println(e.getMessage());
              }
       }


       public static void main(String[] args)
       {
              String dataFile=args[0];
              String attributeName=args[1];
              System.out.println("Running test with data file " + dataFile + " and attribute " + attributeName);
              TestGetSplitPoints(dataFile, attributeName);
              TestPrivateGetSplitPoint(dataFile,attributeName);
       }

}
