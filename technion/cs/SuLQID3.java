package technion.cs;

import technion.cs.DiffPrivacyClassifier;
import technion.cs.PrivacyAgent;
import technion.cs.PrivacyAgents.PrivacyAgentBudget;
import technion.cs.PrivacyBudgetExhaustedException;
import technion.cs.PrivateInstances;
import weka.classifiers.Sourcable;
import java.util.Enumeration;
import java.util.Vector;
import java.util.List;
import java.util.LinkedList;
import java.io.IOException;
import java.math.BigDecimal;

import weka.core.*;

/**
 <!-- globalinfo-start -->
 * SuLQID3 implements ID3 while conforming to the privacy constraints of differential privacy.
 * The class constructs an unpruned decision tree based on the ID3 algorithm, while adhering to differential privacy guarantees.
 * Can only deal with nominal attributes. No missing values allowed. Empty leaves may result in unclassified instances. For more information see: <br/>
 * <br/>
 * TODO: TBD self reference
 * TODO: SuLQ paper reference
 *
 * <p/>
 <!-- globalinfo-end -->
 * TODO: change the following:
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
 * <pre> -d
 *  determines the maximal allowed depth of the
 * induced decision tree</pre>
 *
 <!-- options-end -->
 *
 * @author Arik Friedman (arikf@cs.technion.ac.il)
 * @version $Revision 2.00$, 17 November 2009
 */
public class SuLQID3
              extends DiffPrivacyClassifier
              implements Sourcable{       

       /** The node's successors. */
       protected SuLQID3[] m_Successors;

       /** Attribute used for splitting. */
       protected C45Attribute m_Attribute;

       /** Class value if node is leaf. */
       protected double m_ClassValue;

       /** Class attribute of dataset. */
       protected Attribute m_ClassAttribute;

       /** The value of epsilon for each differential privacy operation */
       protected BigDecimal m_PrivacyBudgetPerAction;

       /** Maximal allowed depth for the induced decision tree. */
       private int m_MaxDepth =DEFAULT_MAX_DEPTH;

       /** Default allowed depth. */
       final private static int DEFAULT_MAX_DEPTH=10;

       /** Option for setting decision tree depth. */
       final public static String MAX_DEPTH_OPTION="d";

       /**
        * Construct a new decision tree node/leaf
        * @return a new node/leaf
        */
       protected SuLQID3 CreateSuLQID3()
       {
              SuLQID3 newTree = new SuLQID3();
              newTree.m_Debug=m_Debug;
              newTree.m_PrivacyBudgetPerAction = m_PrivacyBudgetPerAction;
              newTree.m_MaxDepth = m_MaxDepth -1;               
              return newTree;
       }

       /**
        * Returns default capabilities of the classifier.
        * (used by Weka framework)
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

       /**
        * Return the maximal number of values that an attribute can take
        * @param attList a list of attributes used to induce the tree
        * @return the maximal number of values than an attribute may take
        */
       private int maxNumAttributeValues(List<C45Attribute> attList)
       {
              int max=0;
              for (C45Attribute att:attList)
                     if (att.numValues()>max)
                            max=att.numValues();
              return max;
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

              PrivacyAgent privacyAgent = new PrivacyAgentBudget(m_Epsilon);
              m_PrivacyBudgetPerAction =m_Epsilon.divide(BigDecimal.valueOf(2* (m_MaxDepth+1)), MATH_CONTEXT); // +1 accounts for leaf's budget
              if (m_Debug)
                     System.out.println("epsilon per action is " + m_PrivacyBudgetPerAction);

              // remove instances with missing class
              data.deleteWithMissingClass();
              if (m_Debug)
                     System.out.println("Total (accurate) number of instances: " + data.numInstances());

              PrivateInstances privateData= new PrivateInstances(privacyAgent,data);
              privateData.setDebugMode(m_Debug);
              privateData.setSeed(getSeed());

              List<C45Attribute> candidateAttributes = new LinkedList<C45Attribute>();
              Enumeration attEnum = data.enumerateAttributes();
              while (attEnum.hasMoreElements())
                     candidateAttributes.add(new C45Attribute(((Attribute)attEnum.nextElement())));

              makeTree(privateData,candidateAttributes);
       }

       /**
        * Method for building an Id3 tree while maintaining differential privacy.
        *
        * @param data the training data
        * @param candidateAttributes the attributes to check for splitting the tree
        * @throws PrivacyBudgetExhaustedException thrown if budget is exhausted, shouldn't happen
        */
       protected void makeTree(PrivateInstances data,List<C45Attribute> candidateAttributes) throws PrivacyBudgetExhaustedException {

              m_ClassAttribute = data.classAttribute();

              // Check if no instances have reached this node.
              double numInstances=data.NoisyNumInstances(m_PrivacyBudgetPerAction);
              int maxNumAttributeValues= maxNumAttributeValues(candidateAttributes);
              int numClassValues=data.classAttribute().numValues();

              if (m_Debug)
              {
                     System.out.println("Got noisy count for node: " + numInstances);
                     System.out.println("maximal number of attribute values is " + maxNumAttributeValues + " , and maximal number of class values is " + numClassValues);
              }

              // Check whether
              //     a) No further splits are allowed (depth limit),                              OR
              //     b) there are no more attributes available for splits,               OR
              //     c) there are not enough instances to split the node.
              // In any of this cases, we pick a class value (the node is turned into a leaf).
              if (m_MaxDepth <=0 || candidateAttributes.size()==0  ||
                            !EnoughInstancesToSplit(numInstances, maxNumAttributeValues, numClassValues))
              {
                     if (m_Debug)
                            System.out.println("Getting noisy distribution for leaf");
                     double[] distribution = data.getNoisyDistribution(m_PrivacyBudgetPerAction);
                     m_ClassValue = Utils.maxIndex(distribution);
                     return;
              }

              // If we got here, then we split the node
              // Choose attribute with maximum information gain.
              if (m_Debug)
                     System.out.println("Choosing an attribute");
              m_Attribute=SuLQChooseAttribute(data, candidateAttributes);

              if (m_Debug)
                     System.out.println("Splitting with attribute " + m_Attribute.WekaAttribute().name());

              PrivateInstances[] splitData = data.PartitionByAttribute(m_Attribute);
              m_Successors = new SuLQID3[m_Attribute.numValues()];
              candidateAttributes.remove(m_Attribute); // attribute will not be available in subtrees
              for (int j = 0; j < m_Successors.length; j++) {
                     m_Successors[j] = CreateSuLQID3();
                     if (m_Debug)
                            System.out.println("Making new node");
                     m_Successors[j].makeTree(splitData[j],candidateAttributes);
              }
              candidateAttributes.add(m_Attribute);// attribute will be available for next successors
       }

       private boolean EnoughInstancesToSplit(double estimatedNumInstances, double maxNumAttributeValues, double numClassValues)
       {
              double instancesPerClass=estimatedNumInstances/(maxNumAttributeValues*numClassValues);
              double noiseStddev=PrivateInstances.Stddev(m_PrivacyBudgetPerAction.doubleValue());

              return instancesPerClass>noiseStddev;
       }

       /**
        * Choose an attribute based on infoGain approximation using
        * only sub-linear queries
        * @param data the private data set used for calculations
        * @param candidateAttributes the set of attributes from which we choose
        * @return the attribute to split with
        * @throws PrivacyBudgetExhaustedException in case privacy budget is exhausted
        */
       private C45Attribute SuLQChooseAttribute(PrivateInstances data, List<C45Attribute> candidateAttributes) throws PrivacyBudgetExhaustedException
       {
              double[] attScores=new double[candidateAttributes.size()];
              BigDecimal epsilonPerAction=m_PrivacyBudgetPerAction.divide(BigDecimal.valueOf(2*candidateAttributes.size()), MATH_CONTEXT);

              if (m_Debug)
                     System.out.println("Checking " + candidateAttributes.size() + " attributes, epsilon per action is " + epsilonPerAction);
              for (int attNum=0;attNum<candidateAttributes.size();attNum++)
              {
                     PrivateInstances[] splitData = data.PartitionByAttribute(candidateAttributes.get(attNum));
                     if (m_Debug)
                                   System.out.println("\tChecking attribute " + attNum);

                     // calculate Sum_att (  Sum_cls(   N_{j,c}^{A} * log2 (N_{j,c}^{A} / N_{j}^{A} ))
                     for (PrivateInstances attSplit : splitData) {
                            // calculate Sum_cls(   N_{j,c}^{A} * log2 (N_{j,c}^{A} / N_{j}^{A} )

                            double partitionSize = attSplit.NoisyNumInstances(epsilonPerAction); //N_{j}^{A}
                            if (partitionSize<=0)
                                   continue;

                            double[] distribution = attSplit.getNoisyDistribution(epsilonPerAction);        // N_{j,c}^{A}

                            double scoreShift=0;
                            for (double classCount : distribution)
                            {
                                   if (classCount<=0)
                                          continue;
                                   if (classCount>partitionSize)
                                          classCount=partitionSize;
                                   scoreShift += classCount * Utils.log2(classCount / partitionSize);
                            }
                            attScores[attNum] +=scoreShift;
                     }

              }
              if (m_Debug)
                     for (double score:attScores)
                            System.out.println("Attribute score: " + score);

              return candidateAttributes.get(Utils.maxIndex(attScores));
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

              if (instance.hasMissingValue())
                     throw new NoSupportForMissingValuesException("SuLQID3: no missing values, please.");

              if (m_Attribute == null) // leaf node
                     return m_ClassValue;

              return m_Successors[(int) instance.value(m_Attribute.WekaAttribute())].classifyInstance(instance);
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

              if (instance.hasMissingValue())
                     throw new NoSupportForMissingValuesException("Id3: no missing values, please.");

              if (m_Attribute == null) // leaf node
              {
                     double[] distribution = new double[instance.numClasses()];
                     distribution[(int) m_ClassValue]=1.0;
                     return distribution;
              }

              return m_Successors[(int) instance.value(m_Attribute.WekaAttribute())].distributionForInstance(instance);
       }

       /**
        * Prints the decision tree using the private toString method from below.
        * Function altered with respect to original in Id3 - epsilon parameter is output as well
        *
        * @return a textual description of the classifier
        */
       public String toString() {
              if ((m_ClassAttribute== null) && (m_Successors == null)) {
                     return m_Epsilon+"-Differential Privacy SuLQID3: No model built yet.\n\n";
              }
              return m_Epsilon+"-Differential Privacy SuLQID3\n\n" + toString(0);
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
                        
                     if ( Double.isNaN(m_ClassValue) ) {
                     //if (Instance.isMissingValue(m_ClassValue)) {
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
        * This method was taken as is from ID3 implementation in Weka.
        * It creates a code which represents the decision tree.
        *
        * Adds this tree recursively to the buffer.
        *
        * @param id          the unique id for the method
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
              return RevisionUtils.extract("$Revision: 2.0 $");
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
        * Returns an enumeration of all the available options..
        *
        * @return an enumeration of all available options.
        */
       public Enumeration listOptions() {
              Vector<Option> newVector = new Vector<Option>(1);
              newVector.addElement(new Option("\tMaximal allowed depth for the induced decision tree (default: " +
                            DEFAULT_MAX_DEPTH + ").", MAX_DEPTH_OPTION, 1,
                            "-" + MAX_DEPTH_OPTION ));

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

              String [] options = new String [2+superOptions.length];
              int current = 0;

              options[current++] = "-" + MAX_DEPTH_OPTION; options[current++] = "" + m_MaxDepth;

              for (String superOption : superOptions) options[current++] = superOption;
              while (current < options.length) {
                     options[current++] = "";
              }

              return options;
       }

       public static void main(String[] args) throws IOException
       {
       }
}
