package technion.cs;

import technion.cs.PrivacyAgents.PrivacyAgentBudget;
import technion.cs.Scorer.MaxScorer;
import weka.classifiers.Sourcable;
import java.util.Enumeration;
import java.util.Vector;
import java.util.List;
import java.util.LinkedList;
import java.math.BigDecimal;

import weka.classifiers.trees.j48.Stats;
import weka.core.*;

/**
 * Created by IntelliJ IDEA.
 * User: Arik Friedman
 * Date: 27/04/2009
  * DiffPrivacyC45 implements C4.5 while conforming to the privacy constraints of differential privacy.
 */

/**
 <!-- globalinfo-start -->
 * Class for constructing an unpruned decision tree based on the C4.5 algorithm. Can deal with both nominal and numeric attributes. No missing values allowed. For more information see: <br/>
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
 * @author Arik Friedman
 * @version $Revision: 1.0 $
 */
public class DiffPrivacyC45
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
       protected DiffPrivacyC45[] m_Successors;

       /** Attribute used for splitting. */
       protected C45Attribute m_Attribute;

       /** Split point used for splitting with a numeric attribute */
       protected double m_SplitPoint;

       /** Class value if node is leaf. */
       protected double m_ClassValue;

       /** Class distribution if node is leaf. */
       protected double[] m_Distribution;

       /** instance (approximate) counts - the (noisy) number of instances
        *   in the subtree/leaf per class value */
       protected double[] m_Counts;


       /** instance (approximate) counts and counts  per class value, calibrated
        * according to the (more accurate) number of  instances
        * computed in higher level nodes */
       protected double m_fixedNumInstances;
       protected double[] m_fixedCounts;
       
       /** Class attribute of dataset. */
       protected Attribute m_ClassAttribute;

       /** The value of epsilon for each differential privacy operation */
       protected BigDecimal m_PrivacyBudgetPerAction;

       /** Maximal number of instances allowed for the data set
        * (used to determine sensitivity for info gain  */
       protected int m_maxNumInstances;

       /**
        * Determine whether the checks for number of instances should be skipped when
        * inducing the tree (depth of tree will be fixed)
        */
       protected boolean m_skipNumInstancesChecks =false;

       /** The (approximate) number of instances in the current node */
       protected double m_approxNumInstances;

       /** Confidence level */
       private float m_CF = DEFAULT_CONFIDENCE_FACTOR;
       
       /**
        * Maximal allowed depth for the induced decision tree
        */
       private int m_MaxDepth =DEFAULT_MAX_DEPTH;

       /** Denote whether to post-prune the tree */
       private boolean m_Unpruned = false;

       protected String m_numericAttributesFile=DEFAULT_NUMERIC_ATTRIBUTES_FILE;

       final private static int DEFAULT_MAX_DEPTH=5;
       final public  static String MAX_DEPTH_OPTION="d";
       final static public String C45_SCORER_OPTION="O";
       final static public String MAX_NUM_INSTANCES_OPTION ="I";
       final static public int DEFAULT_MAX_NUM_INSTANCES =0;
       final static public String NUMERIC_ATTRIBUTES_FILE_OPTION="F";
       final static public String DEFAULT_NUMERIC_ATTRIBUTES_FILE="c:\\numericAtts.txt";
       final static public String CONFIDENCE_FACTOR_OPTION="C";
       final static public String UNPRUNED_TREE_OPTION="U";
       final static public float DEFAULT_CONFIDENCE_FACTOR=0.25f;
       final static public String SKIP_NUM_INSTANCES_CHECK_OPTION="S";

       /** C45 scorer to use (default: max scorer) */
       protected AttributeScoreAlgorithm m_Scorer = new MaxScorer();
       final private static String DEFAULT_SCORER_CLASS="technion.dsl.datamining.scorer.MaxScorer";

       protected DiffPrivacyC45 CreateDiffPrivacyC45()
       {
              DiffPrivacyC45 newTree = new DiffPrivacyC45();
              newTree.m_Debug=m_Debug;
              newTree.m_Scorer = m_Scorer;
              newTree.m_PrivacyBudgetPerAction=m_PrivacyBudgetPerAction;
              newTree.m_MaxDepth = m_MaxDepth -1; // not really used (other than being received as a parameter from the UI,
              newTree.m_CF=m_CF;
              newTree.m_Unpruned=m_Unpruned;
              newTree.m_ClassAttribute=m_ClassAttribute;
              newTree.m_skipNumInstancesChecks=m_skipNumInstancesChecks;
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

       /**
        * Builds C45 decision tree classifier.
        *
        * @param data the training data
        * @exception Exception if classifier can't be built successfully
        */
       public void buildClassifier(Instances data) throws Exception {
              // can classifier handle the data?
              getCapabilities().testWithFail(data);
                            
              PrivacyAgent privacyAgent = new PrivacyAgentBudget(m_Epsilon);

              // remove instances with missing class
              data.deleteWithMissingClass();

              PrivateInstances privateData= new PrivateInstances(privacyAgent,data);
              privateData.setDebugMode(m_Debug);
              privateData.setSeed(getSeed());

              if (m_Debug)
                     System.out.println("Total number of instances: " + data.numInstances());
              if (m_maxNumInstances==0)
                     m_maxNumInstances =  (int) Math.pow(2,Math.ceil(Utils.log2(data.numInstances())));
              m_Scorer.InitializeMaxNumInstances(m_maxNumInstances);
              if (m_Debug)
                     System.out.println("MaxNumInstances: " + m_maxNumInstances);

              List<C45Attribute> candidateAttributes = new LinkedList<C45Attribute>();
              int numNumericAtts=0;
              Enumeration attEnum = data.enumerateAttributes();
              while (attEnum.hasMoreElements())
              {
                     Attribute att=(Attribute)attEnum.nextElement();
                     if (att.isNumeric())
                     {
                            candidateAttributes.add(new C45Attribute(att,m_numericAttributesFile));
                            numNumericAtts++;
                     }
                     else
                            candidateAttributes.add(new C45Attribute(att));
              }
              // adapt the differential privacy parameter from a global
              // parameter to a per-operation parameter
              // Due to partition operation, a quota should be given per level of depth
              // for each level of depth (node), we make three operations:
              // 1. noisy count on num instances (to decide whether to turn to leaf or split
              // 2. Determine split points for numeric attributes
              // 3. choose class (for leaf)   or choose splitting attribute (for node)
              int budgetForAttributeSelection=(m_Scorer.GetSensitivity()>0)? 1 : 0;
              int budgetForNumInstancesChecks=m_skipNumInstancesChecks?0:1;
              if(m_Debug){
                System.out.println("Epsilon is " + m_Epsilon);
                System.out.println("budget for attribute selection is " + budgetForAttributeSelection);
                System.out.println("Number of numeric attrbiutes is " + numNumericAtts);
                System.out.println("budget for checking number of instances " + budgetForNumInstancesChecks);
                System.out.println("Depth is " + m_MaxDepth);
                System.out.println("Divisor is " + ((budgetForNumInstancesChecks+budgetForAttributeSelection+numNumericAtts)*(m_MaxDepth)+budgetForNumInstancesChecks+1));
              }
              m_PrivacyBudgetPerAction =m_Epsilon.divide(BigDecimal.valueOf((budgetForNumInstancesChecks+budgetForAttributeSelection+numNumericAtts)*(m_MaxDepth)+budgetForNumInstancesChecks+1),DiffPrivacyClassifier.MATH_CONTEXT); // +2 accounts for leaf's budget              

              if (m_Debug)
                     System.out.println("epsilon per action is " + m_PrivacyBudgetPerAction);
              makeTree(privateData,candidateAttributes, m_MaxDepth);

               // Calibrate noisy counts in the tree to match each other
              fixNumInstancesTopDown(m_approxNumInstances);
              fixClassCountsBottomUp();

              if (!m_skipNumInstancesChecks && !m_Unpruned)
              {
                     if (m_Debug)
                     {
                            System.out.println("\n\nTree before pruning:");
                            System.out.println(toString());
                     }

                     prune();

                     if (m_Debug)
                     {
                            System.out.println("\n\nTree after pruning:");
                            System.out.println(toString());
                     }
              }

       }

       /**
        * Turn a node into a leaf.
        * This method chooses a class value by taking the value
        * that maximizes the noisy count
        *
        * @param data the data in the node
        * @throws PrivacyBudgetExhaustedException thrown if the privacy budget is exhausted
        */
       protected void turnToLeaf(PrivateInstances data) throws PrivacyBudgetExhaustedException
       {
              //Distribution with exponential mechanism
              /*m_ClassValue = data.privateChooseFrequentValue(data.classAttribute(), m_PrivacyBudgetPerAction); // budget cost 2
              m_Distribution = new double[data.numClasses()];
              m_Distribution[(int) m_ClassValue]=1.0;  // setting the distribution like this losses a lot of information (we only keep information about the
              // class attribute, but it won't affect the classification results, and it saves data queries, meaning better privacy
              */

              // Distribution with noisy count mechanism
              m_Counts = data.getNoisyDistribution(m_PrivacyBudgetPerAction);
              m_ClassValue = Utils.maxIndex(m_Counts);
              //m_Distribution = new double[data.numClasses()];
              //m_Distribution[(int) m_ClassValue]=1.0;
              m_Distribution= turnToDistribution(m_Counts);
              return;
       }

       /**
        * Turn a subtree into a leaf.
        *
        * @param counts the distribution of class values in the subtree
        */
       protected void turnSubtreeToLeaf(double[] counts)
       {
              m_Counts = counts;
              m_Distribution=turnToDistribution(m_Counts);
              m_ClassValue = Utils.maxIndex(m_Distribution);
              m_Attribute=null;
              m_Successors=null;
              return;
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

       /**
        * Method for building a C45 tree.
        *
        * @param data the training data
        * @param candidateAttributes the attributes to check for splitting the tree
        * @param depth the maximal allowed depth for the induced sub-tree
        * @exception Exception
        * if decision tree can't be built successfully
        */
       protected void makeTree(PrivateInstances data,List<C45Attribute> candidateAttributes,int depth) throws Exception
       {
              int maxNumAttributeValues=maxNumValues(candidateAttributes);
              int numClassValues=data.classAttribute().numValues();
              m_ClassAttribute = data.classAttribute();

              m_approxNumInstances=0;
              if (!m_skipNumInstancesChecks)
                     m_approxNumInstances = data.NoisyNumInstances(m_PrivacyBudgetPerAction);                   // potential budget cost 1
              if (m_approxNumInstances<0)
                     m_approxNumInstances=0;
              
              // Check whether there are no more attributes available for splits,
              // whether no further splits are allowed, or whether there are enough instances to split the node,
              if (depth<=0 || candidateAttributes.size()==0 || (!m_skipNumInstancesChecks &&
                     !EnoughInstancesToSplit(m_approxNumInstances, maxNumAttributeValues, numClassValues)))
              {
                     turnToLeaf(data);                  // privacy budget use 2a
                     return;
              }

              //  If we got here, then we split the node

              // Determine split point for numeric attributes              
              for (C45Attribute att:candidateAttributes)                        // budget cost 2 (x number of numeric attributes)
              {
                     if (att.isNumeric())
                     {
                            double splitPoint=data.privateChooseNumericSplitPoint(att,m_PrivacyBudgetPerAction,m_Scorer);
                            att.setSplitPoint(splitPoint);
                     }
              }

              // Choose attribute with maximum score              
              m_Attribute=data.privateChooseAttribute(m_Scorer, candidateAttributes, m_PrivacyBudgetPerAction); // budget cost 3
              if (m_Debug)
                     System.out.println("Splitting with attribute " + m_Attribute.WekaAttribute().name());

              PrivateInstances[] splitData = data.PartitionByAttribute(m_Attribute);
              candidateAttributes.remove(m_Attribute); // attribute will not be available in sub-trees unless it is numeric

              if (m_Attribute.isNumeric())
              {
                     m_Successors = new DiffPrivacyC45[2];// numeric attributes use binary splits
                     m_SplitPoint=m_Attribute.getSplitPoint();                      

                     C45Attribute left=new C45Attribute(m_Attribute,m_Attribute.lowerBound(),m_SplitPoint);
                     candidateAttributes.add(left);
                     m_Successors[0] = CreateDiffPrivacyC45();
                     m_Successors[0].makeTree(splitData[0],candidateAttributes,depth-1);
                     candidateAttributes.remove(left);

                     C45Attribute right=new C45Attribute(m_Attribute,m_SplitPoint,m_Attribute.upperBound());
                     candidateAttributes.add(right);
                     m_Successors[1] = CreateDiffPrivacyC45();
                     m_Successors[1].makeTree(splitData[1],candidateAttributes,depth-1);
                     candidateAttributes.remove(right);
              }
              else {
                     m_Successors = new DiffPrivacyC45[m_Attribute.numValues()];
                     for (int j = 0; j < m_Successors.length; j++) {
                            m_Successors[j] = CreateDiffPrivacyC45();
                            m_Successors[j].makeTree(splitData[j],candidateAttributes,depth-1);
                     }
              }
              candidateAttributes.add(m_Attribute);// make sure that the attribute will be available for next successors
       }

       private boolean EnoughInstancesToSplit(double estimatedNumInstances, double maxNumAttributeValues, double numClassValues)
       {
              double instancesPerClass=estimatedNumInstances/(maxNumAttributeValues*numClassValues);
              double noiseStddev=PrivateInstances.Stddev(m_PrivacyBudgetPerAction.doubleValue());
              if (m_Debug)
                     System.out.println("instances per class: " + instancesPerClass + ", noiseStddev: " + noiseStddev + " , enoughInstances: " + (instancesPerClass>noiseStddev));
              return instancesPerClass>noiseStddev;
       }


       /**
        * This method makes top down fixes to the (approximate) number of instances in each node.
        * The estimation of the number of instances is more accurate in the upper levels of the
        * decision tree, since the noise is smaller with respect to the number of instances.
        * This method calibrates in a top-down manner all the estimations for number of instances
        * such that they would total to the amount calculated in the top of the tree.
        * For a given node, we use the estimated numbers of instances in each of the sub-trees
        * to determine the ratio of instances within each subtree, and then we divide the
        * estimated number of instances among the sub-trees according to this estimation.
        * @param fixedNumInstances the new estimated number of instances, according to the
        * calculations performed by the parent node
        */
       protected void fixNumInstancesTopDown(double fixedNumInstances)
       {
              m_fixedNumInstances =fixedNumInstances;

              // no further fixes to do in a leaf node
              if (m_Successors==null || m_Successors.length==0)
                     return;

              // use the approximated number of instances within the sub-trees
              // to determine the ratio of instances in each sub-node (taken from
              // the fixed number of instances)
              double total=0;
              for (DiffPrivacyC45 node:m_Successors)
                     total+=node.m_approxNumInstances;

              for (DiffPrivacyC45 node:m_Successors)
                     node.fixNumInstancesTopDown((node.m_approxNumInstances/total)*m_fixedNumInstances);
       }


       protected void fixClassCountsBottomUp()
       {
              // In a leaf node, the fixed counts are obtained by applying
              // the distribution of class counts to the fixed number of instances
              if (m_Attribute==null)
              {
                     m_fixedCounts = new double[m_Distribution.length];
                     for (int i=0;i< m_fixedCounts.length;i++)
                            m_fixedCounts[i]=m_Distribution[i]*m_fixedNumInstances;
                     return;
              }

              // For any other node, the fixed counts are obtained by summing over
              // all the counts in the sub-trees. Execution of fixNumInstancesTopDown
              // prior to running fixClassCountsBottomUp ensures that the counts
              // will sum to the node's fixed number of instances.
              for (DiffPrivacyC45 node:m_Successors)
              {
                     node.fixClassCountsBottomUp();
                     if (m_fixedCounts==null)
                            m_fixedCounts=new double[node.m_fixedCounts.length];
                     for (int i=0;i< m_fixedCounts.length;i++)
                            m_fixedCounts[i]+=node.m_fixedCounts[i];
              }
       }

       /**
        * Prunes a tree using C4.5's pruning procedure.
        */
       protected void prune()
       {
              if (m_Attribute==null)      // indication for a leaf
                     return;

              // Prune all sub-trees
              for (DiffPrivacyC45 node:m_Successors)
                     node.prune();

              // Compute error if this Tree would be leaf
              double[] counts= m_fixedCounts;
              double errorsLeaf = getLeafEstimatedErrors(counts);

              // Compute error for the whole sub-tree
              double errorsTree = getSubtreeEstimatedErrors();
              if (m_Debug)
                     System.out.println("prune(): errors in leaf: " + errorsLeaf
                                   + ", errors in subtree: " + errorsTree);
              // Decide if leaf is best choice.
              if (Utils.smOrEq(errorsLeaf,errorsTree+0.1))
              {
                     turnSubtreeToLeaf(counts);
                     if (m_Debug) System.out.println("Pruning node ");
              }
              else
                     if (m_Debug) System.out.println("Node not pruned.");
       }

       /**
        * Get the (approximately) largest branch of the current subtree
        * @return return the branch for which there is the maximal (noisy) number of instances
        */
       /*
       private int getLargestBranch()
       {
              if (m_Attribute==null || m_Successors==null || m_Successors.length==0)
                     throw new IllegalArgumentException("Cannot get largest branch for a leaf");

              double max=Double.MIN_VALUE;
              int index=-1;
              for (int i=0;i<m_Successors.length;i++)
                     if (m_Successors[i].m_approxNumInstances>max)
                            index=i;
              return index;
       }
         */
       
       /**
        * Computes estimated errors for this node, when it is considered
        * as a leaf.
        *@param counts the distribution of class values within the subtree
        * @return estimated error of the subtree if it would be turned into a leaf
        */
       private double getLeafEstimatedErrors(double[] counts){
              if (m_fixedNumInstances<=0)
                     return 0;
              double approxNumErrors = numErrors(counts);

              //if (m_Debug) System.out.println("numErrors for leaf " + name() + " is " + numErrors);
              return approxNumErrors +
                            Stats.addErrs(m_fixedNumInstances,approxNumErrors ,m_CF);
       }

       /**
        * Get the weight of erroneous classifications in a given data set
        * (The total weight of instances in minority classes) within a leaf
        * @param counts the distribution of class values within the subtree or leaf
        * @return the approximate number of errors within the subtree or leaf
        */
       private double numErrors(double[] counts)
       {
              if (m_fixedNumInstances<=0)
                     return 0;

               int index=Utils.maxIndex(counts);
               double error = m_fixedNumInstances-counts[index];
               return (error<0) ? 0 : error;
        }
               /**
         * Computes estimated errors for tree.
         * @return estimated errors for tree
         */
        private double getSubtreeEstimatedErrors()
        {
                if (m_Attribute==null)
                        return getLeafEstimatedErrors(m_fixedCounts);

               if (m_fixedNumInstances<=0)
                      return 0;

                double errors = 0;

                for (DiffPrivacyC45 node:m_Successors)
                        errors+=node.getSubtreeEstimatedErrors();
               
                //if (m_Debug) System.out.println("numErrors for subtree " + name() + " is " + errors);
                return errors;
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
                     throw new NoSupportForMissingValuesException("C4.5: no missing values, "
                                   + "please.");
              }
              if (m_Attribute == null) {
                     return m_ClassValue;
              } else {
                     // treat non-numeric attribute
                     if (!m_Attribute.isNumeric())
                            return m_Successors[(int) instance.value(m_Attribute.WekaAttribute())].
                                          classifyInstance(instance);

                     // treat numeric attribute
                     if (instance.value(m_Attribute.WekaAttribute())<m_SplitPoint)
                            return m_Successors[0].classifyInstance(instance);
                     else
                            return m_Successors[1].classifyInstance(instance);
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
                     throw new NoSupportForMissingValuesException("C4.5: no missing values, "
                                   + "please.");
              }
              if (m_Attribute == null) {
                     return m_Distribution;
              } else {
                     // treat non-numeric attribute
                     if (!m_Attribute.isNumeric())
                            return m_Successors[(int) instance.value(m_Attribute.WekaAttribute())].
                                          distributionForInstance(instance);

                     // treat numeric attribute
                     if (instance.value(m_Attribute.WekaAttribute())<m_SplitPoint)
                            return m_Successors[0].distributionForInstance(instance);
                     else
                            return m_Successors[1].distributionForInstance(instance);
              }
       }

       /**
        * Prints the decision tree using the private toString method from below.
        * Function altered with respect to original in C4.5 - epsilon parameter is output as well
        *
        * @return a textual description of the classifier
        */
       public String toString() {

              if ((m_Distribution == null) && (m_Successors == null)) {
                     return m_Epsilon.toString() + "-Differential Privacy C4.5: No model built yet.\n\n";
              }
              return m_Epsilon.toString() + "-Differential Privacy C4.5\n\n" + "\n\n" + toString(0);
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
                            text.append("  [" + m_fixedNumInstances + "]");
                            text.append(": ").append(m_ClassAttribute.value((int) m_ClassValue));                            
                            text.append("   Counts  " + distributionToString(m_fixedCounts) + "   Distribution " + distributionToString(m_Distribution));
                     }
              } else {
                     text.append("  [" + m_fixedNumInstances + "]");
                     for (int j = 0; j < m_Successors.length; j++) {
                            text.append("\n");
                            for (int i = 0; i < level; i++) {
                                   text.append("|  ");
                            }
                            if (m_Attribute.isNumeric())
                                   text.append(m_Attribute.WekaAttribute().name()).append(j==0?" < ":" >= ").append(m_SplitPoint);
                            else
                                   text.append(m_Attribute.WekaAttribute().name()).append(" = ").append(m_Attribute.WekaAttribute().value(j));
                            text.append(m_Successors[j].toString(level + 1));
                     }
              }
              return text.toString();
       }

       private String distributionToString(double[] distribution)
       {
              StringBuffer text = new StringBuffer();
              text.append("[");
              for (double d:distribution)
                     text.append(d + "; ");
              text.append("]");
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
                     subBuffers = new StringBuffer[m_Successors.length];
                     newID      = id;
                     for (i = 0; i < m_Successors.length; i++) {
                            newID++;

                            buffer.append("    ");
                            if (i > 0)
                                   buffer.append("else ");
                            if (m_Attribute.isNumeric())
                                   buffer.append("if (((Double) i[").append(m_Attribute.WekaAttribute().index()).append("]).doubleValue()").append(i==0?" < ":" >= ").append(m_SplitPoint).append(")\n");
                            else
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

       public void setNumericAttributesFile(String file)
       {
              m_numericAttributesFile=file;
       }

       public String getNumericAttributesFile()
       {
              if (m_numericAttributesFile==null || m_numericAttributesFile.length()==0)
                     return DEFAULT_NUMERIC_ATTRIBUTES_FILE;

              return m_numericAttributesFile;
       }

       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String numericAttributesFileTipText() {
              return "The path and name for a text file containing upper and lower bounds of numeric attributes";
       }

       /**
        * Get the value of unpruned.
        *
        * @return Value of unpruned.
        */
       public boolean getUnpruned() {

               return m_Unpruned;
       }


       /**
        * Set the value of unpruned. Turns reduced-error pruning
        * off if set.
        * @param v  Value to assign to unpruned.
        */
       public void setUnpruned(boolean v) {
               m_Unpruned = v;
       }

       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String unprunedTipText() {
               return "Whether to prune the tree";
       }

       /**
        * Get the value of CF.
        *
        * @return Value of CF.
        */
       public float getConfidenceFactor() {

               return m_CF;
       }

       /**
        * Set the value of CF.
        *
        * @param v  Value to assign to CF.
        */
       public void setConfidenceFactor(float v) {

               m_CF = v;
       }


       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String confidenceFactorTipText() {
               return "The confidence factor used for pruning (smaller values incur "
                           + "more pruning).";
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
       public String skipNumInstancesChecksTipText() {
              return "Determines whether the number of instances checks for tree depth should be skipped when inducing the tree. Setting this to true disables pruning.";
       }

       /**
        * Returns an enumeration of all the available options..
        *
        * @return an enumeration of all available options.
        */
       public Enumeration listOptions() {
              Vector<Option> newVector = new Vector<Option>(7);
              newVector.addElement(new Option("\tMaximal allowed depth for the induced decision tree (default: " +
                            DEFAULT_MAX_DEPTH + ").", MAX_DEPTH_OPTION, 1,
                            "-" + MAX_DEPTH_OPTION ));

              newVector.addElement(new Option(
                            "\tFull class name of attribute scorer.\n"
                                          + "\t(default: " + DEFAULT_SCORER_CLASS +")",
                            C45_SCORER_OPTION, 1, "-" + C45_SCORER_OPTION));

              newVector.addElement(new Option(
                            "\tPath and name of file with upper and lower bounds for numeric attributes.\n"
                                          + "\t(default: " + DEFAULT_NUMERIC_ATTRIBUTES_FILE +")",
                            NUMERIC_ATTRIBUTES_FILE_OPTION, 1, "-" + NUMERIC_ATTRIBUTES_FILE_OPTION));

              newVector.addElement(new Option(
                            "\tMaximal number of allowed training instances, 0 will automatically round up to nearest log 2 based on given training set.\n"
                                          + "\t(default: " + DEFAULT_MAX_NUM_INSTANCES+")",
                            MAX_NUM_INSTANCES_OPTION, 1, "-" + MAX_NUM_INSTANCES_OPTION));

              newVector.addElement(new Option("\tWhether pruning is performed",
                            UNPRUNED_TREE_OPTION, 0,"-" + UNPRUNED_TREE_OPTION));

                newVector.addElement(new Option("\tSet confidence threshold for pruning.\n" +
                            "\t(default 0.25)",
                            CONFIDENCE_FACTOR_OPTION, 1, "-" + CONFIDENCE_FACTOR_OPTION + " <pruning confidence>"));

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
              String paramString = Utils.getOption(MAX_DEPTH_OPTION, options);
              if (paramString.length() != 0)
                     setMaxDepth(Integer.parseInt(paramString));
              else
                     setMaxDepth(DEFAULT_MAX_DEPTH);

              String scorerString = Utils.getOption(C45_SCORER_OPTION, options);

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

              paramString = Utils.getOption(NUMERIC_ATTRIBUTES_FILE_OPTION, options);
              if (paramString.length()>0)
                     setNumericAttributesFile(paramString);
              else
                     setNumericAttributesFile(DEFAULT_NUMERIC_ATTRIBUTES_FILE);

              // Pruning option
              m_Unpruned = Utils.getFlag(UNPRUNED_TREE_OPTION, options);

              m_skipNumInstancesChecks = Utils.getFlag(SKIP_NUM_INSTANCES_CHECK_OPTION, options);

              // Confidence factor for pruning
              paramString= Utils.getOption(CONFIDENCE_FACTOR_OPTION, options);
                if (paramString.length() != 0)
                {
                        if (m_Unpruned)
                                throw new IllegalArgumentException("Doesn't make sense to change confidence for unpruned "
                                            +"tree!");
                        else
                        {
                                setConfidenceFactor(Float.parseFloat(paramString));
                                if ((m_CF <= 0) || (m_CF >= 1))
                                        throw new IllegalArgumentException("Confidence has to be greater than zero and smaller " +
                                                    "than one!");
                        }
                }
                else
                        setConfidenceFactor(DEFAULT_CONFIDENCE_FACTOR);

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

              String [] options = new String [10+superOptions.length];
              int current = 0;

              options[current++] = "-" + MAX_DEPTH_OPTION; options[current++] = "" + m_MaxDepth;

              options[current++] = "-" + C45_SCORER_OPTION;
              options[current++] = getScorer().getClass().getName();

              options[current++]="-" + NUMERIC_ATTRIBUTES_FILE_OPTION;
              options[current++]=getNumericAttributesFile();

              options[current++]="-" + CONFIDENCE_FACTOR_OPTION;
              options[current++]=Float.toString(getConfidenceFactor());
              
              if (m_Unpruned)
                        options[current++] = "-"+UNPRUNED_TREE_OPTION;

              if (m_skipNumInstancesChecks)
                        options[current++] = "-"+SKIP_NUM_INSTANCES_CHECK_OPTION;

              for (String superOption : superOptions) options[current++] = superOption;
              while (current < options.length) {
                     options[current++] = "";
              }

              return options;
       }

}
