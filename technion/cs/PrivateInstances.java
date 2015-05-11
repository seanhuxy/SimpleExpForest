package technion.cs;

import technion.cs.PrivacyAgents.CommonBigDecimal;
import technion.cs.Scorer.RandomScorer;
import weka.core.*;

import java.io.IOException;
import java.io.Reader;
import java.io.Serializable;
import java.util.*;
import java.math.BigDecimal;

import technion.cs.PrivacyAgents.PrivacyAgentPartition;

/**
 * Author: Arik Friedman
 * Date: 16/11/2009
 *
 * PrivateInstances is a wrapper to the Weka Instances class.
 * It provides access to the Instances, while maintaining differential privacy
 * with a PrivacyAgent that manages the privacy budget
 */
public class PrivateInstances implements Serializable, RevisionHandler {

       private Instances m_Data;
       private PrivacyAgent m_Agent;

       // debug mode
       private boolean m_Debug=false;

       /** for serialization */
       private static final long serialVersionUID = 2175398004986986596L;

       /** Random number generator,
        * can be initialized with a seed to allow repeating experiments */
       protected Random m_Random = new Random();

       /**
        * Returns the revision string.
        * @return the revision
        */
       public String getRevision() {
              return RevisionUtils.extract("$Revision: 2.0 $");
       }

       /* ** Constructors (backward compatibility) **/

       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/Reader reader) throws IOException {
              m_Data = new Instances(reader); m_Agent =agent;}
       @Deprecated public PrivateInstances(PrivacyAgent agent,/*@non_null@*/Reader reader, int capacity) throws IOException {//noinspection deprecation
              m_Data = new Instances(reader,capacity);
              m_Agent =agent;}
       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/Instances dataset) {
              m_Data = new Instances(dataset); m_Agent =agent;}
       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/Instances dataset, int capacity) {
              m_Data = new Instances(dataset,capacity); m_Agent =agent;}
       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/Instances source, int first, int toCopy) {
              m_Data = new Instances(source,first,toCopy); m_Agent =agent;}
       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/String name,/*@non_null@*/FastVector attInfo, int capacity) {
              m_Data = new Instances(name,attInfo,capacity);
              m_Agent =agent;}


       /*     ** General public methods **/
       /**
        * Change the debug mode
        * @param mode debug mode (set to true for debug message output)
        */
       public void setDebugMode(boolean mode)
       {
              m_Debug=mode;
       }

       /**
        * Set the seed for the psuedo random number generator
        * @param seed seed for random numbers
        */
       public void setSeed(int seed)
       {
              m_Random=new Random(seed);
       }

       /*       * New constructors, get PrivateInstances dataset       */

       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/PrivateInstances dataset) {
              m_Data = new Instances(dataset.m_Data); m_Agent =agent;}
       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/PrivateInstances dataset, int capacity) {
              m_Data = new Instances(dataset.m_Data,capacity); m_Agent =agent;}
       public PrivateInstances(PrivacyAgent agent,/*@non_null@*/PrivateInstances source, int first, int toCopy) {
              m_Data = new Instances(source.m_Data,first,toCopy); m_Agent =agent;}

       /* * (Public) Data access methods **/
       public /*@pure@*/ int numAttributes() {return m_Data.numAttributes();}
       public /*@pure@*/ int numClasses() {return m_Data.numClasses();}
       public /*@pure@*/ Attribute classAttribute() {return m_Data.classAttribute();}

       public void add(/*@non_null@*/ Instance instance) {m_Data.add(instance);}
       public void compactify() {m_Data.compactify();}

       public /*@pure@*/ Attribute attribute(int index) {return m_Data.attribute(index);}
       public /*@pure@*/ Attribute attribute(String name) {return m_Data.attribute(name);}
       
       /**
        * Get an array with the noisy distribution of class values within the dataset.
        * The array contains noisy counts of the occurrences of each class value in
        * the dataset
        * @param epsilon privacy parameter
        * @return array with noisy counts for each class value
        * @throws PrivacyBudgetExhaustedException thrown if epsilon is more than
        * the available privacy budget
        */
       public double[] getNoisyDistribution(BigDecimal epsilon) throws PrivacyBudgetExhaustedException
       {
              if (epsilon.signum()<0) // BigDecimal equivalent to (epsilon<0)
                     throw new IllegalArgumentException("Negative values of epsilon are illegal");

              if (!m_Agent.Request(epsilon))
                     throw new PrivacyBudgetExhaustedException("getNoisyDistribution: privacy budget exhausted - Requested budget: " + epsilon + ", existing budget: " + m_Agent.RemainingBudget());

              if (m_Debug)
                     System.out.println("Getting distribution for a leaf with " + m_Data.numInstances()+ " instances, with epsilon =  " + epsilon);
              // A partition operation to divide the instances by class value,
              // followed by a noisy count on each partition, would give us the results.
              // Instead, we make a short cut, and add noisy directly over the distribution,
              double[] distribution = getDistribution();
              for (int i=0;i<distribution.length;i++)
                   distribution[i]+=laplace(BigDecimal.ONE.divide(epsilon,DiffPrivacyClassifier.MATH_CONTEXT));
              return distribution;
       }

       /**
        * This function checks whether the count of instances is above a threshold
        * It uses the exponential mechanism to determine this, rather than a noisy count.
        * to make more efficient use of the privacy budget.
        * If we are above the threshold, the query function has the value (numInstances-threshold).
        * If we are below the threshold, the query function has the value -(numInstances-threshold).
        * @param threshold we want to check whether the number of instances exceeds the threshold
        * @param epsilon the privacy paramter
        * @return (hopefully) true if number of instances larger than threshold (depending on noise)
        * @throws PrivacyBudgetExhaustedException in case budget is exhausted
        */
       public boolean IsCountMoreThan(double threshold, BigDecimal epsilon) throws PrivacyBudgetExhaustedException
       {
              if (!m_Agent.Request(epsilon))
                     throw new PrivacyBudgetExhaustedException("IsCountMoreThan: privacy budget exhausted - Requested budget: " + epsilon + ", existing budget: " + m_Agent.RemainingBudget());

              double[] scores = new double[2];
              scores[0]=m_Data.numInstances()-threshold; //above threshold
              scores[1]=threshold-m_Data.numInstances(); // below threshold
              boolean result=(drawFromScores(scores, 1, epsilon)==0);
              if (m_Debug)
                     System.out.println("Checking whether there are more than " + threshold + " instances. Real count = " + m_Data.numInstances() + " continuation decision: " + result);
              return result;
       }

       /**
        * Perform a NoisyCount operation to get an estimate on the number
        * of instances within the dataset
        * (The method was named NoisyNumInstances rather than NoisyCount
        * to maintain naming consistency with Weka's NumInstances method)
        * 
        * @param epsilon the privacy parameter
        * @return a noisy estimate of the number of instances
        * @throws PrivacyBudgetExhaustedException thrown if epsilon is above the
        * available privacy budget
        */
       public double NoisyNumInstances(BigDecimal epsilon) throws PrivacyBudgetExhaustedException
       {
              if (!m_Agent.Request(epsilon))
                     throw new PrivacyBudgetExhaustedException("NoisyNumInstances: privacy budget exhausted - Requested budget: " + epsilon + ", existing budget: " + m_Agent.RemainingBudget());
              return m_Data.numInstances()+laplace(BigDecimal.ONE.divide(epsilon,DiffPrivacyClassifier.MATH_CONTEXT));
       }

       /**
        * PartitionByAttribute gets an attribute, and splits the
        * PrivateInstances to distinct sets of PrivateInstances.
        * The splitting is applied according to the attribute type
        * @param att the attribute used to split the instances
        * @return A set of PrivateInstances, one per attribute value
        */
       public PrivateInstances[] PartitionByAttribute(C45Attribute att)
       {
              if (att.isNumeric())
                     return PartitionByNumericAttribute(att);
              return PartitionByNominalAttribute(att);
       }

       public PrivacyAgent GetPrivacyAgent()
       {
              return m_Agent;
       }

        /**
        * PartitionByNominalAttribute gets an attribute, and splits the
        * PrivateInstances to distinct sets of PrivateInstances, one
        * per attribute value. The PartitionByNominalAttribute operation
        * uses the Partition privacy agent, to reflect that queries
        * performed on one set of items do not affect other sets of
        * items
        * @param att the attribute used to split the instances
        * @return A set of PrivateInstances, one per attribute value
        */
       public PrivateInstances[] PartitionByNominalAttribute(C45Attribute att)
       {              
              PrivateInstances[] splitData = new PrivateInstances[att.numValues()];
              Map<Object,BigDecimal> budgetMap = new HashMap<Object,BigDecimal>();

              CommonBigDecimal common = new CommonBigDecimal(new BigDecimal(0.0, DiffPrivacyClassifier.MATH_CONTEXT));
              for (int j = 0; j <splitData.length; j++) {
                     splitData[j] = new PrivateInstances(new PrivacyAgentPartition(m_Agent,budgetMap,j, common),
                                   m_Data, m_Data.numInstances());                                         
                     splitData[j].setDebugMode(m_Debug);
                     splitData[j].setSeed(m_Random.nextInt());
              }

              Enumeration instEnum = m_Data.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     splitData[(int) inst.value(att.WekaAttribute())].add(inst);
              }
              for (PrivateInstances aSplitData : splitData) {
                     aSplitData.compactify();
              }
              return splitData;
       }

       /**
        * PartitionByNumericAttribute gets an attribute, and splits the
        * PrivateInstances to two distinct sets of PrivateInstances, one
        * for points left of the split point, and for points right of the split
        * point. . The PartitionByNumericAttribute operation
        * uses the Partition privacy agent, to reflect that queries
        * performed on one set of items do not affect other sets of
        * items
        * @param att the attribute used to split the instances
        * @return A set of PrivateInstances, one per attribute value
        */
       public PrivateInstances[] PartitionByNumericAttribute(C45Attribute att)
       {
              PrivateInstances[] splitData = new PrivateInstances[2];
              Map<Object,BigDecimal> budgetMap = new HashMap<Object,BigDecimal>();

              CommonBigDecimal common = new CommonBigDecimal(new BigDecimal(0.0, DiffPrivacyClassifier.MATH_CONTEXT));
              for (int j = 0; j < 2; j++) {
                     splitData[j] = new PrivateInstances(new PrivacyAgentPartition(m_Agent,budgetMap,j, common),
                                   m_Data, m_Data.numInstances());
                     splitData[j].setDebugMode(m_Debug);
                     splitData[j].setSeed(m_Random.nextInt());
              }

              double splitPoint=att.getSplitPoint();
              Enumeration instEnum = m_Data.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     if (inst.value(att.WekaAttribute())<splitPoint)
                            splitData[0].add(inst);
                     else
                            splitData[1].add(inst);                     
              }
              for (PrivateInstances aSplitData : splitData) {
                     aSplitData.compactify();
              }
              return splitData;
       }

       /**
        * Choose an attribute using the exponential mechanism
        * @param scorer the scorer according to which the attributes are evaluated
        * @param attList the list of attributes to consider
        * @param epsilon differential privacy parameter
        * @return the chosen attribute, picked randomly with probability proportional to the attribute score
        * @throws PrivacyBudgetExhaustedException thrown if the privacy budget is exhausted
        */
       public C45Attribute privateChooseAttribute(AttributeScoreAlgorithm scorer, List<C45Attribute> attList, BigDecimal epsilon) throws PrivacyBudgetExhaustedException
       {
              if (scorer.GetSensitivity()>0 && !m_Agent.Request(epsilon))
                     throw new PrivacyBudgetExhaustedException("PrivateChooseAttribute: privacy budget exhausted - Requested budget: " + epsilon + ", existing budget: " + m_Agent.RemainingBudget());

              int index;

              // shortcut: don't go through the exponential mechanism just for a random selection
              if (scorer.getClass().equals(RandomScorer.class))
                  index=m_Random.nextInt(attList.size());
              else
              {
                     if (m_Debug)
                            System.out.println("Private Choose Attribute, going to evaluate " + attList.size() + " attributes");

                     double[] scores= new double[attList.size()];
                     int i=0;
                     for (C45Attribute att:attList)
                     {
                            scores[i]=scorer.Score(m_Data,att);
                            if (m_Debug)
                                   System.out.println("\tAttribute " + i + " (" + att.WekaAttribute().name() +
                                                 ((att.isNumeric())  ?    ("(split point " + att.getSplitPoint() + ")")  :  ""   )
                                                 + ") Score: " + scores[i]);
                            i++;
                     }

                     index=drawFromScores(scores,scorer.GetSensitivity(), epsilon);
              }

              if (m_Debug)
                     System.out.println("Attribute " + index + " was picked");

              return attList.get(index);
       }

       /**
        * A method for choosing a split point for a numeric attribute while preserving
        * differential privacy
        * @param att a numeric attribute for splitting (non numeric attribute will cause an exception)
        * @param epsilon differential privacy parameter
        * @param scorer the scorer according to which the attributes are evaluated
        * @return a split point for the numeric attribute
        * @throws PrivacyBudgetExhaustedException thrown if the privacy budget is exhausted
        */
       public double privateChooseNumericSplitPoint(C45Attribute att, BigDecimal epsilon, AttributeScoreAlgorithm scorer) throws PrivacyBudgetExhaustedException
       {
              if (!att.isNumeric())
                     throw new IllegalArgumentException("Numeric split point can only be chosen for a numeric attribute.");

              if (scorer.GetSensitivity()>0 && !m_Agent.Request(epsilon))
                     throw new PrivacyBudgetExhaustedException("PrivateChooseNumericSplitPoint: privacy budget exhausted - Requested budget: " + epsilon + ", existing budget: " + m_Agent.RemainingBudget());

              //Short cut: don't go through the exponential mechanism just for random selection
              if (scorer.getClass().equals(RandomScorer.class))
                     return (att.lowerBound()+m_Random.nextDouble()*(att.upperBound()-att.lowerBound()));

              double[][] splitPoints=C45Attribute.GetSplitPoints(m_Data,att,scorer);
              double[] scores=new double[splitPoints.length];
              double[] weights=new double[splitPoints.length];
              // Calculate the overall score for each interval
              // (= <interval size> * <score in each point>)
              for (int i=0;i<splitPoints.length;i++)
              {
                     scores[i]=splitPoints[i][2];
                     weights[i]=splitPoints[i][1]-splitPoints[i][0];
                     //if (m_Debug)
                     //       System.out.println("Range x:["+splitPoints[i][0] + ", " + splitPoints[i][1] +"), score " + scores[i] + ", weight is " + weights[i]);
              }

              // Given the scores choose an interval with the exponential mechanism
              int index=drawFromWeightedScores(scores,weights,scorer.GetSensitivity(),epsilon);

              // Uniformly pick a point within the interval
              double splitPoint=splitPoints[index][0]+(splitPoints[index][1]-splitPoints[index][0])*m_Random.nextDouble();
              return splitPoint;              
       }

       /**
        * Pick up a value of an attribute (typically the class attribute) in a privacy preserving way,
        * giving precedence to most frequent value
        * @param att the attribute that we need to pick a value for (typically the class attribute)
        * @param epsilon the differential privacy parameter for the operation
        * @return the most frequent value (given privacy restrictions)
        * @throws PrivacyBudgetExhaustedException thrown if the privacy  budget is exhausted
        */
       public int privateChooseFrequentValue(Attribute att, BigDecimal epsilon)  throws PrivacyBudgetExhaustedException
       {
              if (!m_Agent.Request(epsilon))
                     throw new PrivacyBudgetExhaustedException("PrivateChooseFrequentValue: privacy budget exhausted - Requested budget: " + epsilon + ", existing budget: " + m_Agent.RemainingBudget());

              double[] distribution = new double[att.numValues()];
              Enumeration instEnum = m_Data.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     distribution[(int) inst.value(att)]++;
              }

              int index=drawFromScores(distribution, 1.0, epsilon);
              if (m_Debug)
                     System.out.println("Choose class value: picked the value " + att.value(index)  + "(index " + index + ")");
              return index;
       }


       /**
        *
        * @param scores the set of attribute scores, according to which an attribute should be picked
        * @param scorerSensitivity sensitivity of the scoring function
        *          (delta Q to use when setting the probability distribution)
        * @param epsilon the epsilon value to use for privacy
        * @throws PrivacyBudgetExhaustedException if privacy budget is exhausted
        * @return index of chosen attribute
        */
       public int drawFromScores(double[] scores, double scorerSensitivity, BigDecimal epsilon) throws PrivacyBudgetExhaustedException
       {
              // adjust scores so the largest one will be zero
              // (this will ensure that after exponent, the scores
              // are in the range 0-1)
              double maxScore=scores[Utils.maxIndex(scores)];
              for (int i=0;i<scores.length;i++)
                     scores[i]-=maxScore;

              // add differential privacy factor to scores
              for (int i=0;i<scores.length;i++)
              {
                     if (scorerSensitivity!=0)
                            scores[i]=Math.exp(scores[i]*(epsilon.doubleValue()/(2*scorerSensitivity)));                     
                     if (m_Debug)
                            System.out.println("\tOption " + i + " Score: " + scores[i]);
              }

              // sample a value based on the distribution implied by the scores
              return drawFromDistribution(scores);
       }       

        /**
        *
        * @param scores the set of attribute scores, according to which an attribute should be picked
        * @param scorerSensitivity sensitivity of the scoring function
        *          (delta Q to use when setting the probability distribution)
         * @param weights holds the weight for each score (the size of the interval
         *    in which the score applies) 
        * @param epsilon the epsilon value to use for privacy
        * @throws PrivacyBudgetExhaustedException if privacy budget is exhausted
        * @return index of chosen attribute
        */
       public int drawFromWeightedScores(double[] scores, double[] weights, double scorerSensitivity, BigDecimal epsilon) throws PrivacyBudgetExhaustedException
       {
              // adjust scores so the largest one will be zero
              // (this will ensure that after exponent, the scores
              // are in the range 0-1)
              double maxScore=scores[Utils.maxIndex(scores)];
              for (int i=0;i<scores.length;i++)
                     scores[i]-=maxScore;

              // add differential privacy factor and weight factor to scores
              for (int i=0;i<scores.length;i++)
              {
                     scores[i]=weights[i]*Math.exp(scores[i]*(epsilon.doubleValue()/(2*scorerSensitivity)));
                     //if (m_Debug)
                     //       System.out.println("\tOption " + i + " original score " + scores[i]/weights[i] + ", weighted score: " + scores[i]);
              }

              // sample a value based on the distribution implied by the scores
              return drawFromDistribution(scores);
       }

       /**
        * Pick a number given a distribution of weights
        * the distribution is not assumed to be normalized, and
        * the normalize method is invoked first
        * For example, given the array [3 5 2], the method will return:
        *     0 with probability 0.3
        *     1 with probability 0.5
        *     2 with probability 0.2
        * @param dist the distribution of weights to draw from
        * @return a sample from the distribution according to the proportional weights
        */
       protected int drawFromDistribution(double[] dist)
       {
              return drawFromNormalizedDistribution(normalize(dist));
       }

       /**
        * Pick a number given a distribution of weights
        * It is assumed that the distribution is normalized (i.e.,
        * the sum of weights is 1).
        * For example, given the array [0.3 0.5 0.2], the method will return:
        *     0 with probability 0.3
        *     1 with probability 0.5
        *     2 with probability 0.2
        * @param dist the distribution to draw from
        * @return a sample from the distribution according to the weights
        */
       protected int drawFromNormalizedDistribution(double[] dist)
       {
              double rnd=m_Random.nextDouble();              
              double curr=0;
              for (int i = 0; i < dist.length; i++) {
                     curr += dist[i];
                     if (curr > rnd) {
                            return i;
                     }
              }
              return dist.length-1;
       }

       /**
        * Normalize an array of weights such that the
        * weights sum to 1
        * @param weights the weights to normalize
        * @return an array of the weights normalized so they sum to 1
        */
       protected double[] normalize(double[] weights)
       {
              double sum=0;
              for (double d: weights)
                     sum+=d;
              double[] dist = new double[weights.length];
              for (int i=0;i< weights.length;i++)
              {
                     if (sum!=0)
                            dist[i]= weights[i]/sum;
                     else
                            dist[i]=1.0/weights.length;
              }
              return dist;
       }

       /* * (Internal) Data access helper methods **/

       /**
        * get the accurate number of instances in the dataset
        * @return number of instances
        */
       protected /*@pure@*/ int numInstances() {return m_Data.numInstances();}

       /**
        * Get the distribution of class values within the dataset
        * The method counts the number of instances having each class value,
        * and returns an array containing the counts
        *
        * @return an array with distribution of class values 
        */
       private double[] getDistribution()
       {
              double[] distribution = new double[m_Data.numClasses()];
              Enumeration instEnum = m_Data.enumerateInstances();
              while (instEnum.hasMoreElements()) {
                     Instance inst = (Instance) instEnum.nextElement();
                     distribution[(int) inst.classValue()]++;
              }
              return distribution;
       }

       /** Privacy helper functions **/

       /**
        * Sample a number from Laplace distribution with location parameter miu
        * and with scale parameter beta.
        * Using inverse transform sampling, we draw a number u based on the uniform distribution,
        * and get the sample x = miu-(lambda*sign(u)*ln(1-2|u|))
        * @param miu  location parameter for laplace distribution
        * @param bigBeta scale parameter for laplace distribution (in BigDecimal format)
        * @return a value sampled from Laplace(miu, beta)
        */
       private double laplace(double miu, BigDecimal bigBeta)
       {
              double beta=bigBeta.doubleValue();
              double uniform= m_Random.nextDouble()-0.5;
              return miu-beta*((uniform>0) ? -Math.log(1.-2*uniform) : Math.log(1.+2*uniform));
       }

       /**
        * Sample a number from Laplace distribution with location parameter 0
        * and with scale parameter beta.
        * @param beta scale parameter for laplace distribution
        * @return a value sampled from Laplace(0, beta)
        */
       private double laplace(BigDecimal beta)
       {
              return laplace(0,beta);
       }


       public static double Stddev(double epsilon)
       {
              return Math.sqrt(2.0)/epsilon;
       }


       public void setRandomSeed(int seed)
       {
              m_Random=new Random(seed);
       }



}
