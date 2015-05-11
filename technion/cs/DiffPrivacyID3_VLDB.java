package technion.cs;

import technion.cs.Scorer.InfoGainScorer;
import technion.cs.PrivacyAgents.PrivacyAgentBudget;
import weka.classifiers.Sourcable;

import java.util.Enumeration;
import java.util.Vector;
import java.util.List;
import java.util.LinkedList;
import java.math.BigDecimal;

import weka.core.*;

/**
 * Created by IntelliJ IDEA.
 * User: Arik Friedman
 * Date: 27/04/2009
  * DiffPrivacyID3_VLDB implements Id3 while conforming to the privacy constraints of differential privacy.
 */

/**
 <!-- globalinfo-start -->
 * Class for constructing an unpruned decision tree based on the ID3 algorithm. Can only deal with nominal attributes. No missing values allowed. Empty leaves may result in unclassified instances. For more information see: <br/>
 * <br/>
 * R. Quinlan (1986). Induction of decision trees. Machine Learning. 1(1):81-106.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Quinlan1986,
 *    author = {R. Quinlan},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {81-106},
 *    title = {Induction of decision trees},
 *    volume = {1},
 *    year = {1986}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 *
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 *
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 1.23 $
 */
public class DiffPrivacyID3_VLDB
              extends DiffPrivacyClassifier
              implements Sourcable{

       static {
              try {
                     java.beans.PropertyEditorManager.registerEditor(
                                   AttributeScoreAlgorithm.class,
                                   weka.gui.GenericObjectEditor.class);
              } catch (Throwable t) {
                     // ignore
              }
       }

       /** The node's successors. */
       protected DiffPrivacyID3_VLDB[] m_Successors;

       /** Attribute used for splitting. */
       protected C45Attribute m_Attribute;

       /** Class value if node is leaf. */
       protected double m_ClassValue;

       /** Class distribution if node is leaf. */
       protected double[] m_Distribution;

       /** Class attribute of dataset. */
       protected Attribute m_ClassAttribute;

       /** The value of epsilon required for deciding whether to continue splitting */
       protected BigDecimal m_PrivacyBudgetForStoppingCriterion;

       /** The value of epsilon required to pick an attribute */
       protected BigDecimal m_PrivacyBudgetForAttributeChoice;

       /** The total value of epsilon required per node */
       protected BigDecimal m_PrivacyBudgetForNodes;

       /** The value of epsilon for noisy count operations in leaves */
       protected BigDecimal m_PrivacyBudgetForLeaves;

       /** Maximal number of instances allowed for the data set
        * (used to determine sensitivity for info gain  */
       protected int m_maxNumInstances;

       /**
        * Determine whether the checks for number of instances should be skipped when
        * inducing the tree (depth of tree will be fixed)
        */
       protected boolean m_skipNumInstancesChecks =false;

       /**
        * Maximal allowed depth for the induced decision tree
        */
       private int m_MaxDepth =DEFAULT_MAX_DEPTH;

/**
        * Granularity requirement for differential private count operation in leaves, with 0.95 confidence
        */
       private int m_Granularity = DEFAULT_GRANULARITY;

       final private static int DEFAULT_MAX_DEPTH=10;
       final public  static String MAX_DEPTH_OPTION="d";
       final public  static String GRANULARITY_OPTION ="g";
       final static public String ID3_SCORER_OPTION="O";
       final static public String MAX_NUM_INSTANCES_OPTION ="I";
       final static public int DEFAULT_MAX_NUM_INSTANCES =0;
       final static public String SKIP_NUM_INSTANCES_CHECK_OPTION="S";
       final static public int DEFAULT_GRANULARITY =500;
       final static public double CONFIDENCE_BOUNDS=0.95;

       /** ID3 scorer to use (default: info gain) */
       protected AttributeScoreAlgorithm m_Scorer =new InfoGainScorer();
       final private static String DEFAULT_SCORER_CLASS="technion.dsl.datamining.scorer.InfoGainScorer";

       protected DiffPrivacyID3_VLDB CreateDiffPrivacyId3_VLDB()
       {
              DiffPrivacyID3_VLDB newTree = new DiffPrivacyID3_VLDB();
              newTree.m_Debug=m_Debug;
              newTree.m_Scorer = m_Scorer;
              newTree.m_PrivacyBudgetForNodes = m_PrivacyBudgetForNodes;
              newTree.m_PrivacyBudgetForAttributeChoice=m_PrivacyBudgetForAttributeChoice;
              newTree.m_PrivacyBudgetForStoppingCriterion = m_PrivacyBudgetForStoppingCriterion;
              newTree.m_PrivacyBudgetForLeaves=m_PrivacyBudgetForLeaves;
              newTree.m_MaxDepth = m_MaxDepth -1; // not really used (other than being received as a parameter from the UI,
              newTree.m_skipNumInstancesChecks=m_skipNumInstancesChecks;
              // max depth is later passed between trees through the call to makeTree)

              return newTree;
       }

       /**
        * Returns default capabilities of the classifier.
        *
        * @return      the capabilities of this classifier
        */
       public Capabilities getCapabilities() {
              Capabilities result = super.getCapabilities();

              // attributes
              result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);

              // class
              result.enable(Capabilities.Capability.NOMINAL_CLASS);
              result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

              // instances
              result.setMinimumNumberInstances(0);

              return result;
       }

       private int maxNumValues(List<C45Attribute> attList)
       {
              int max=0;
              for (C45Attribute att:attList)
                     if (att.numValues()>max)
                            max=att.numValues();
              return max;
       }

       private double avgNumValues(List<C45Attribute> attList)
       {
              double total=0;
              for (C45Attribute att:attList)
                     total+=att.numValues();
              return total/attList.size();
       }
       /**
        * Builds Id3 decision tree classifier.
        *
        * @param data the training data
        * @exception Exception if classifier can't be built successfully
        */
       public void buildClassifier(Instances data) throws Exception {
              // can classifier handle the data?
              getCapabilities().testWithFail(data);

              // adapt the differential privacy parameter from a global
              // parameter to a per-operation parameter
              // Due to partition operation, a quota should be given per level of depth
              // for each level of depth (node), we make two operations:
              // 1. noisy count on num instances (to decide whether to turn to leaf or split
              // 2. choose class (for leaf)   or choose splitting attribute (for node)
              if (m_Debug)
                     System.out.println("Total number of instances: " + data.numInstances());
              if (m_maxNumInstances==0)
                     m_maxNumInstances =  (int) Math.pow(2,Math.ceil(Utils.log2(data.numInstances())));
              m_Scorer.InitializeMaxNumInstances(m_maxNumInstances);
              if (m_Debug)
                     System.out.println("MaxNumInstances: " + m_maxNumInstances);

              PrivacyAgent privacyAgent = new PrivacyAgentBudget(m_Epsilon);

              // Preallocate the required budget to provide the granularity level in the count queries in the leaves
              m_PrivacyBudgetForLeaves=new BigDecimal(-Math.log(1-Math.pow(CONFIDENCE_BOUNDS,data.classAttribute().numValues()))/ m_Granularity, DiffPrivacyClassifier.MATH_CONTEXT);

              // Preallocate the required budget to check the stopping rule (whether there are enough instances left)
              if (m_skipNumInstancesChecks)
                     m_PrivacyBudgetForStoppingCriterion =new BigDecimal(0,DiffPrivacyClassifier.MATH_CONTEXT);
              else
                     m_PrivacyBudgetForStoppingCriterion =new BigDecimal(Math.log(CONFIDENCE_BOUNDS/(1-CONFIDENCE_BOUNDS))/(2* m_Granularity),DiffPrivacyClassifier.MATH_CONTEXT);              

              // Divide the remaining budget among the attribute choice decisions
              BigDecimal budget=m_Epsilon.subtract(m_PrivacyBudgetForLeaves);
              if (budget.signum()>0)
              {
                     if (m_PrivacyBudgetForStoppingCriterion.signum()>0)
                     {
                            int depth=(int) Math.floor(budget.divide(m_PrivacyBudgetForStoppingCriterion,DiffPrivacyClassifier.MATH_CONTEXT).doubleValue());
                            if (m_MaxDepth>depth)
                                   m_MaxDepth=depth;
                            budget=budget.subtract(m_PrivacyBudgetForStoppingCriterion.multiply(BigDecimal.valueOf(m_MaxDepth)));
                     }
                     if (m_MaxDepth>0)
                            m_PrivacyBudgetForAttributeChoice=budget.divide(BigDecimal.valueOf(m_MaxDepth),DiffPrivacyClassifier.MATH_CONTEXT);
                     else
                            m_PrivacyBudgetForAttributeChoice=BigDecimal.ZERO;
              }
              else if (!m_skipNumInstancesChecks || m_Scorer.GetSensitivity()>0)
              // if we need to check for stopping criterion or choose attributes, there's no budget for that
              {
                     m_MaxDepth=0;
                     m_PrivacyBudgetForStoppingCriterion=BigDecimal.ZERO;
                     m_PrivacyBudgetForAttributeChoice=BigDecimal.ZERO;
              }


              /*if (m_Scorer.GetSensitivity()==0)
                     m_PrivacyBudgetForAttributeChoice=new BigDecimal(0,DiffPrivacyClassifier.MATH_CONTEXT);
              else
                     m_PrivacyBudgetForAttributeChoice=new BigDecimal(-2*m_Scorer.GetSensitivity()*Math.log((1-CONFIDENCE_BOUNDS)/data.numAttributes())/ m_Granularity,DiffPrivacyClassifier.MATH_CONTEXT);
                */
              //m_PrivacyBudgetForNodes=m_PrivacyBudgetForAttributeChoice.add(m_PrivacyBudgetForStoppingCriterion,DiffPrivacyClassifier.MATH_CONTEXT);

              // calibrate the budget to match the available epsilon
              // if the allocated epsilon is smaller than the overall budget required
              // according to granularity, we keep the calculated budget levels (if we run out of budget,
              // the tree will be shallower)
              // However, if we have more budget than the minimum required, we stretch it
              // over all the queries so we can take more precise decisions
              if (m_Debug)
              {
                     System.out.println("Overall epsilon is " + m_Epsilon);
                     System.out.println("Depth is " + m_MaxDepth);
                     System.out.println("Granularity is " + m_Granularity);
                     System.out.println("epsilon per leaf " +m_PrivacyBudgetForLeaves);
                     System.out.println("epsilon per continuation choice is " + m_PrivacyBudgetForStoppingCriterion);
                     System.out.println("epsilon per attribute choice is " + m_PrivacyBudgetForAttributeChoice);
                     //System.out.println("epsilon per node " +m_PrivacyBudgetForNodes);
              }
              // Commented - better results obtained when extra epsilon goes to the counts
              /*
              BigDecimal totalBudget=m_PrivacyBudgetForLeaves.add(m_PrivacyBudgetForNodes.multiply(BigDecimal.valueOf(m_MaxDepth)));
              if (m_Epsilon.compareTo(totalBudget)>0)
              {
                     if (m_Debug)
                            System.out.println("Given Epsilon " + m_Epsilon + " is larger than needed budget " + totalBudget +", increasing budget allocation");
                     BigDecimal leavesRatio=m_PrivacyBudgetForLeaves.divide(totalBudget,DiffPrivacyClassifier.MATH_CONTEXT);
                     BigDecimal contChoiceRatio=m_PrivacyBudgetForStoppingCriterion.divide(m_PrivacyBudgetForNodes,DiffPrivacyClassifier.MATH_CONTEXT);
                     m_PrivacyBudgetForLeaves=m_Epsilon.multiply(leavesRatio,DiffPrivacyClassifier.MATH_CONTEXT);
                     m_PrivacyBudgetForNodes=m_Epsilon.subtract(m_PrivacyBudgetForLeaves).divide(BigDecimal.valueOf(m_MaxDepth),DiffPrivacyClassifier.MATH_CONTEXT);                     
                     m_PrivacyBudgetForStoppingCriterion=m_PrivacyBudgetForNodes.multiply(contChoiceRatio,DiffPrivacyClassifier.MATH_CONTEXT);
                     m_PrivacyBudgetForAttributeChoice=m_PrivacyBudgetForNodes.subtract(m_PrivacyBudgetForStoppingCriterion);

                     if (m_Debug)
                                   {
                                          // Sanity check
                                          BigDecimal test=m_Epsilon;
                                          for (int i=0;i<m_MaxDepth;i++)
                                          {
                                                 test=test.subtract(m_PrivacyBudgetForAttributeChoice);
                                                 test=test.subtract(m_PrivacyBudgetForStoppingCriterion);
                                          }
                                          System.out.println("After " + m_MaxDepth + " subtractions, remaining budget is " + test + ", and required budget is " + m_PrivacyBudgetForLeaves);

                                          System.out.println("After calibartion:");
                                          System.out.println("Overall epsilon is " + m_Epsilon);
                                          System.out.println("Depth is " + m_MaxDepth);
                                          System.out.println("Granularity is " + m_Granularity);
                                          System.out.println("epsilon per leaf " +m_PrivacyBudgetForLeaves);
                                          System.out.println("epsilon per continuation choice is " +m_PrivacyBudgetForStoppingCriterion);
                                          System.out.println("epsilon per attribute choice is " + m_PrivacyBudgetForAttributeChoice);
                                          System.out.println("epsilon per node " +m_PrivacyBudgetForNodes);
                                   }

              }
                */

              // remove instances with missing class
              data.deleteWithMissingClass();

              PrivateInstances privateData= new PrivateInstances(privacyAgent,data);
              privateData.setDebugMode(m_Debug);
              privateData.setSeed(getSeed());

              List<C45Attribute> candidateAttributes = new LinkedList<C45Attribute>();
              Enumeration attEnum = data.enumerateAttributes();
              while (attEnum.hasMoreElements())
                     candidateAttributes.add(new C45Attribute((Attribute)attEnum.nextElement()));              

              makeTree(privateData,candidateAttributes, m_MaxDepth);
       }

       /**
        * Method for building an Id3 tree.
        *
        * @param data the training data
        * @param candidateAttributes the attributes to check for splitting the tree
        * @param depth the maximal allowed depth for the induced sub-tree
        * @exception Exception
        * if decision tree can't be built successfully
        */
       protected void makeTree(PrivateInstances data,List<C45Attribute> candidateAttributes, int depth) throws Exception {
              // Check if no instances have reached this node.
              int maxNumAttributeValues=maxNumValues(candidateAttributes);
              double avgNumAttributeValues=avgNumValues(candidateAttributes);
              m_ClassAttribute = data.classAttribute();              
              // Check whether there are no more attributes available for splits,
              // whether there are enough instances to split the node, or whether no further splits are allowed
              if (depth<=0 || candidateAttributes.size()==0  ||
                            //data.GetPrivacyAgent().RemainingBudget().compareTo(m_PrivacyBudgetForLeaves.add(m_PrivacyBudgetForStoppingCriterion).add(m_PrivacyBudgetForAttributeChoice))<0  ||
                            (!m_skipNumInstancesChecks && // is the remaining budget smaller than required for continuation and attribute choices?
                            !EnoughInstancesToSplit(data, m_PrivacyBudgetForStoppingCriterion, avgNumAttributeValues)))
              {
                     //Distribution with exponential mechanism
                     /*m_ClassValue = data.privateChooseFrequentValue(data.classAttribute(), m_PrivacyBudgetForNodes); // budget cost 2
                     m_Distribution = new double[data.numClasses()];
                     m_Distribution[(int) m_ClassValue]=1.0;  // setting the distribution like this losses a lot of information (we only keep information about the
                     // class attribute, but it won't affect the classification results, and it saves data queries, meaning better privacy
                     */
                     // Distribution with noisy count mechanism
                     m_Distribution = data.getNoisyDistribution(data.GetPrivacyAgent().RemainingBudget());   // privacy budget use 2a
                     m_ClassValue = Utils.maxIndex(m_Distribution);
                     m_Distribution = turnToDistribution(m_Distribution);
                     return;
              }

              //  If we got here, then we split the node
              // Choose attribute with maximum information gain.
              m_Attribute=data.privateChooseAttribute(m_Scorer, candidateAttributes, m_PrivacyBudgetForAttributeChoice); // budget cost 2

              // Choose attribute with maximum majority class.
              //m_Attribute=data.privateChooseAttribute(new NumHitsScorer(), candidateAttributes);
              if (m_Debug)
                     System.out.println("Splitting with attribute " + m_Attribute.WekaAttribute().name());

              PrivateInstances[] splitData = data.PartitionByAttribute(m_Attribute);

              m_Successors = new DiffPrivacyID3_VLDB[m_Attribute.numValues()];
              candidateAttributes.remove(m_Attribute); // attribute will not be available in subtrees
              for (int j = 0; j < m_Successors.length; j++) {
                     m_Successors[j] = CreateDiffPrivacyId3_VLDB();
                     m_Successors[j].makeTree(splitData[j],candidateAttributes,depth-1);
              }
              candidateAttributes.add(m_Attribute);// attribute will be available for next successors
       }


       /**
        * Given noisy counts, turn them into a distribution, and make sure it "behaves",
        * i.e., there are no negative probabilities, and that the elements sum up to 1
        * The fixed distribution must have the same dominant class value as in the counts
        * @param counts a (possibly noisy) count of instances per class value
        * @return a fixed version of the distribution
        */
       protected double[] turnToDistribution(double counts[])
       {
              double[] distribution = new double[counts.length];
              // in the new distribution, make sure there are no negative values
              double sum=0;
              for (int i=0;i<counts.length;i++)
              {
                     distribution[i]=(counts[i]<0) ? 0 : counts[i]; // the minimum between 0 and counts[i]
                     sum+=distribution[i];

              }

              // If the new max has changed (this may only happen if the max value was negative as well,
              // and now all values are zero), then make sure that the original max value
              // will still be chosen by setting it to be larger than the current max value.
              int originalMax= Utils.maxIndex(counts);
              int newMax= Utils.maxIndex(distribution);
              if (newMax!=originalMax)
              {
                     distribution[originalMax]=distribution[newMax]+Utils.SMALL;
                     sum+=Utils.SMALL;
              }

              // ensure that the distribution elements some up to 1.0
              if (sum==0)
                     distribution[originalMax]=1;
              else
                     for (int i=0;i<distribution.length;i++)
                            distribution[i]/=sum;

              return distribution;
       }

       private boolean EnoughInstancesToSplit(PrivateInstances data, BigDecimal privacyBudget, double avgNumAttributeValues)
                 throws PrivacyBudgetExhaustedException
       {
              int numClassValues=data.classAttribute().numValues();
              double noiseStddev=PrivateInstances.Stddev(m_PrivacyBudgetForLeaves.doubleValue());
              //return data.IsCountMoreThan((noiseStddev*maxNumAttributeValues*numClassValues), privacyBudget);
              //return data.IsCountMoreThan((noiseStddev*avgNumAttributeValues*numClassValues), privacyBudget);
              return data.IsCountMoreThan((noiseStddev*numClassValues), privacyBudget);
       }

       /**
        * Classifies a given test instance using the decision tree.
        *
        * @param instance the instance to be classified
        * @return the classification
        * @throws NoSupportForMissingValuesException if instance has missing values
        */
       public double classifyInstance(Instance instance)
                     throws NoSupportForMissingValuesException {

              if (instance.hasMissingValue()) {
                     throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                   + "please.");
              }
              if (m_Attribute == null) {
                     return m_ClassValue;
              } else {
                     return m_Successors[(int) instance.value(m_Attribute.WekaAttribute())].
                                   classifyInstance(instance);
              }
       }

       /**
        * Computes class distribution for instance using decision tree.
        *
        * @param instance the instance for which distribution is to be computed
        * @return the class distribution for the given instance
        * @throws NoSupportForMissingValuesException if instance has missing values
        */
       public double[] distributionForInstance(Instance instance)
                     throws NoSupportForMissingValuesException {

              if (instance.hasMissingValue()) {
                     throw new NoSupportForMissingValuesException("Id3: no missing values, "
                                   + "please.");
              }
              if (m_Attribute == null) {
                     return m_Distribution;
              } else {
                     return m_Successors[(int) instance.value(m_Attribute.WekaAttribute())].
                                   distributionForInstance(instance);
              }
       }

       /**
        * Prints the decision tree using the private toString method from below.
        * Function altered with respect to original in Id3 - epsilon parameter is output as well
        *
        * @return a textual description of the classifier
        */
       public String toString() {

              if ((m_Distribution == null) && (m_Successors == null)) {
                     return m_Epsilon.toString() + "-Differential Privacy Id3: No model built yet.\n\n";
              }
              return m_Epsilon.toString() + "-Differential Privacy Id3\n\n" + "\n\n" + toString(0);
       }
       /**
        * Outputs a tree at a certain level.
        *
        * @param level the level at which the tree is to be printed
        * @return the tree as string at the given level
        */
       protected String toString(int level) {

              StringBuffer text = new StringBuffer();

              if (m_Attribute == null) {
                     //if (Instance.isMissingValue(m_ClassValue)) {
                     if ( Double.isNaN(m_ClassValue) ) {
                            text.append(": null");
                     } else {
                            text.append(": ").append(m_ClassAttribute.value((int) m_ClassValue));
                     }
              } else {
                     for (int j = 0; j < m_Attribute.numValues(); j++) {
                            text.append("\n");
                            for (int i = 0; i < level; i++) {
                                   text.append("|  ");
                            }
                            text.append(m_Attribute.WekaAttribute().name()).append(" = ").append(m_Attribute.WekaAttribute().value(j));
                            text.append(m_Successors[j].toString(level + 1));
                     }
              }
              return text.toString();
       }

       /**
        * Adds this tree recursively to the buffer.
        *
        * @param id          the unqiue id for the method
        * @param buffer      the buffer to add the source code to
        * @return            the last ID being used
        * @throws Exception  if something goes wrong
        */
       protected int toSource(int id, StringBuffer buffer) throws Exception {
              int                 result;
              int                 i;
              int                 newID;
              StringBuffer[]      subBuffers;

              buffer.append("\n");
              buffer.append("  protected static double node").append(id).append("(Object[] i) {\n");

              // leaf?
              if (m_Attribute == null) {
                     result = id;
                     if (Double.isNaN(m_ClassValue))
                            buffer.append("    return Double.NaN;");
                     else
                            buffer.append("    return ").append(m_ClassValue).append(";");
                     if (m_ClassAttribute != null)
                            buffer.append(" // ").append(m_ClassAttribute.value((int) m_ClassValue));
                     buffer.append("\n");
                     buffer.append("  }\n");
              }
              else {
                     buffer.append("    // ").append(m_Attribute.WekaAttribute().name()).append("\n");

                     // subtree calls
                     subBuffers = new StringBuffer[m_Attribute.numValues()];
                     newID      = id;
                     for (i = 0; i < m_Attribute.numValues(); i++) {
                            newID++;

                            buffer.append("    ");
                            if (i > 0)
                                   buffer.append("else ");
                            buffer.append("if (((String) i[").append(m_Attribute.WekaAttribute().index()).append("]).equals(\"").append(m_Attribute.WekaAttribute().value(i)).append("\"))\n");
                            buffer.append("      return node");
                            buffer.append(newID);
                            buffer.append("(i);\n");

                            subBuffers[i] = new StringBuffer();
                            newID         = m_Successors[i].toSource(newID, subBuffers[i]);
                     }
                     buffer.append("    else\n");
                     buffer.append("      throw new IllegalArgumentException(\"Value '\" + i[").append(m_Attribute.WekaAttribute().index()).append("] + \"' is not allowed!\");\n");
                     buffer.append("  }\n");

                     // output subtree code
                     for (i = 0; i < m_Attribute.numValues(); i++) {
                            buffer.append(subBuffers[i].toString());
                     }
                     //noinspection UnusedAssignment
                     subBuffers = null;

                     result = newID;
              }

              return result;
       }

       /**
        * Returns a string that describes the classifier as source. The
        * classifier will be contained in a class with the given name (there may
        * be auxiliary classes),
        * and will contain a method with the signature:
        * <pre><code>
        * public static double classify(Object[] i);
        * </code></pre>
        * where the array <code>i</code> contains elements that are either
        * Double, String, with missing values represented as null. The generated
        * code is public domain and comes with no warranty. <br/>
        * Note: works only if class attribute is the last attribute in the dataset.
        *
        * @param className the name that should be given to the source class.
        * @return the object source described by a string
        * @throws Exception if the souce can't be computed
        */
       public String toSource(String className) throws Exception {
              StringBuffer        result;
              int                 id;

              result = new StringBuffer();

              result.append("class ").append(className).append(" {\n");
              result.append("  public static double classify(Object[] i) {\n");
              id = 0;
              result.append("    return node").append(id).append("(i);\n");
              result.append("  }\n");
              toSource(id, result);
              result.append("}\n");

              return result.toString();
       }

       /**
        * Returns the revision string.
        *
        * @return		the revision
        */
       public String getRevision() {
              return RevisionUtils.extract("$Revision: 1.0 $");
       }

       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String granularityTipText() {
              return "The granularity level required for differential privacy queries.";
       }

       /**
        * Set the granularity level
        *
        * @param gran the granularity required for differential privacy queries
        */
       public void setGranularity(int gran)
       {
              m_Granularity =gran;
       }

       /**
        * Get the granularity level
        *
        * @return the the granularity level required for differential privacy queries
        */
       public int getGranularity()
       {
              return m_Granularity;
       }

       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String scorerTipText() {
              return "The scorer to use to score attributes when spliting nodes for decision tree induction.";
       }

       /**
        * Set the attribute scorer
        *
        * @param newScorer the new scorer to use.
        */
       public void setScorer(AttributeScoreAlgorithm newScorer)
       {
              m_Scorer = newScorer;
       }

       /**
        * Get the used scorer
        *
        * @return the used attribute scorer
        */
       public AttributeScoreAlgorithm getScorer()
       {
              return m_Scorer;
       }


       public void setMaxNumInstances(int num)
       {
              m_maxNumInstances=num;
       }

       public int getMaxNumInstances()
       {
              return m_maxNumInstances;
       }

       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String maxNumInstancesTipText() {
              return "The maximal number of instances that the training set can get (0 to round up to nearest log2 from above).";
       }


       public void setSkipNumInstancesChecks(boolean setDepth)
       {
              m_skipNumInstancesChecks =setDepth;
       }

       public boolean getSkipNumInstancesChecks()
       {
              return m_skipNumInstancesChecks;
       }

       /**
               * Returns the tip text for this property
               * @return tip text for this property suitable for
               * displaying in the explorer/experimenter gui
               */
              public String maxDepthTipText() {
                     return "The maximal depth allowed for the induced decision tree.";
              }

              /**
               * Set the privacy policy
               *
               * @param depth the maximal depth allowed for the induced decision tree
               */
              public void setMaxDepth(int depth)
              {
                     m_MaxDepth =depth;
              }

              /**
               * Get the used privacy policy
               *
               * @return the maximal depth allowed for the induced decision tree
               */
              public int getMaxDepth()
              {
                     return m_MaxDepth;
              }


       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String skipNumInstancesChecksTipText() {
              return "Determines whether the number of instances checks for tree depth should be skipped when inducing the tree.";
       }

       /**
        * Returns an enumeration of all the available options..
        *
        * @return an enumeration of all available options.
        */
       public Enumeration listOptions() {
              Vector<Option> newVector = new Vector<Option>(6);
              newVector.addElement(new Option("\tThe required granularity level for differential private operations (default: " +
                            DEFAULT_GRANULARITY + ").", GRANULARITY_OPTION, 1,
                            "-" + GRANULARITY_OPTION));

              newVector.addElement(new Option("\tMaximal allowed depth for the induced decision tree (default: " +
                            DEFAULT_MAX_DEPTH + ").", MAX_DEPTH_OPTION, 1,
                            "-" + MAX_DEPTH_OPTION ));

              newVector.addElement(new Option(
                            "\tFull class name of attribute scorer.\n"
                                          + "\t(default: " + DEFAULT_SCORER_CLASS +")",
                            ID3_SCORER_OPTION, 1, "-" + ID3_SCORER_OPTION));

              newVector.addElement(new Option(
                            "\tMaximal number of allowed training instances, 0 will automatically round up to nearest log 2 based on given training set.\n"
                                          + "\t(default: " + DEFAULT_MAX_NUM_INSTANCES+")",
                            MAX_NUM_INSTANCES_OPTION, 1, "-" + MAX_NUM_INSTANCES_OPTION));

              newVector.addElement(new Option(
                            "\tMaximal number of allowed training instances, 0 will automatically round up to nearest log 2 based on given training set.\n"
                                          + "\t(default:  true)",
                            MAX_NUM_INSTANCES_OPTION, 1, "-" + MAX_NUM_INSTANCES_OPTION));

              newVector.addElement(new Option("\tWhether number of instances checks are skipped",
                            SKIP_NUM_INSTANCES_CHECK_OPTION, 0,"-" + SKIP_NUM_INSTANCES_CHECK_OPTION));

              return newVector.elements();
       }

       /**
        * Sets the OptionHandler's options using the given list. All options
        * will be set (or reset) during this call (i.e. incremental setting
        * of options is not possible).
        *
        * @param options the list of options as an array of strings
        * @throws Exception if an option is not supported
        */
//@ requires options != null;
//@ requires \nonnullelements(options);
       public void setOptions(String[] options) throws Exception {
              super.setOptions(options);
              String paramString = Utils.getOption(GRANULARITY_OPTION, options);
              if (paramString.length() != 0)
                     setGranularity(Integer.parseInt(paramString));
              else
                     setGranularity(DEFAULT_GRANULARITY);

              paramString = Utils.getOption(MAX_DEPTH_OPTION, options);
              if (paramString.length() != 0)
                     setMaxDepth(Integer.parseInt(paramString));
              else
                     setMaxDepth(DEFAULT_MAX_DEPTH);


              String scorerString = Utils.getOption(ID3_SCORER_OPTION, options);

              if (scorerString.length() > 0) {
                     setScorer((AttributeScoreAlgorithm) Utils.forName(AttributeScoreAlgorithm.class,scorerString,null));
              } else {
                     setScorer((AttributeScoreAlgorithm) Utils.forName(AttributeScoreAlgorithm.class,DEFAULT_SCORER_CLASS,null));
              }

              String maxNumInstancesString = Utils.getOption(MAX_NUM_INSTANCES_OPTION,options);
              if (maxNumInstancesString.length()>0)
                     setMaxNumInstances(Integer.parseInt(maxNumInstancesString));
              else
                     setMaxNumInstances(DEFAULT_MAX_NUM_INSTANCES);

              m_skipNumInstancesChecks = Utils.getFlag(SKIP_NUM_INSTANCES_CHECK_OPTION, options);
       }

       /**
        * Gets the current option settings for the OptionHandler.
        *
        * @return the list of current option settings as an array of strings
        */
//@ ensures \result != null;
//@ ensures \nonnullelements(\result);
/*@pure@*/
       public String[] getOptions()
       {
              String [] superOptions=super.getOptions();

              String [] options = new String [9+superOptions.length];
              int current = 0;

              options[current++] = "-" + MAX_DEPTH_OPTION; options[current++] = "" + m_MaxDepth;

              options[current++] = "-" + GRANULARITY_OPTION; options[current++] = "" + m_Granularity;

              options[current++] = "-" + ID3_SCORER_OPTION;
              options[current++] = getScorer().getClass().getName();

              options[current++] = "-" + MAX_NUM_INSTANCES_OPTION;
              options[current++] = Integer.toString(getMaxNumInstances());

              if (m_skipNumInstancesChecks)
                        options[current++] = "-"+SKIP_NUM_INSTANCES_CHECK_OPTION;

              for (String superOption : superOptions) options[current++] = superOption;
              while (current < options.length) {
                     options[current++] = "";
              }

              return options;
       }

}
