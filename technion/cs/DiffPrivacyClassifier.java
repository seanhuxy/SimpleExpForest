package technion.cs;

import weka.classifiers.RandomizableClassifier;
import weka.core.Option;
import weka.core.Utils;
import java.util.Enumeration;
import java.util.Vector;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;


/**
 * Abstract utility class for handling settings common to
 * classifiers.that conform to differential privacy.
 * Specifically, the class handles input of the privacy budget (epsilon).
 * Since the class extends RandomizableClassifier , it also inherits
 * methods to use a random seed.
 *
 * @author Arik Friedman (arikf@cs.technion.ac.il)
 * @version $Revision: 1.0 $
 */
public abstract class DiffPrivacyClassifier extends RandomizableClassifier {

       public static final MathContext MATH_CONTEXT = new MathContext(20, RoundingMode.DOWN); // default is round down, to avoid exhausting the privacy budget

       /** A default value for the privacy parameter */
       protected static final String DEFAULT_E_PARAMETER = "0.1";

       /** privacy parameter */
       protected BigDecimal m_Epsilon=new BigDecimal(DEFAULT_E_PARAMETER, MATH_CONTEXT);


       /** string for setting the value of epsilon */
       public static final String E_PARAMETER_OPTION = "e";


       /**
        * Returns the tip text for this property
        * @return tip text for this property suitable for
        * displaying in the explorer/experimenter gui
        */
       public String epsilonTipText() {
              return "The epsilon  parameter for e-differential privacy.";              
       }

       /**
        * Get the value of privacy parameter.
        * @return Value of privacy parameter.
        */
       public String getEpsilon() {
              return m_Epsilon.toString();
       }

       /**
        * Set the value of privacy parameter.
        * @param eStr Value to assign to privacy parameter, in string format
        */
       public void setEpsilon(String eStr) {
              if (eStr!=null && eStr.length()!=0)
                     m_Epsilon = new BigDecimal(eStr,MATH_CONTEXT);
       }

       /**
        * Returns an enumeration of all the available options..
        *
        * @return an enumeration of all available options.
        */
       public Enumeration listOptions() {
              Vector<Option> newVector = new Vector<Option>(1);
              newVector.addElement(new Option("\tThe e-differential privacy parameter (default: " +
                            DEFAULT_E_PARAMETER + ").", E_PARAMETER_OPTION, 1,
                            "-" + E_PARAMETER_OPTION ));
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
              String paramString = Utils.getOption(E_PARAMETER_OPTION, options);
              if (paramString!=null && paramString.length() != 0)
                     setEpsilon(paramString);
              else
                     setEpsilon(DEFAULT_E_PARAMETER);
       }

       /**
        * Gets the current option settings for the OptionHandler.
        *
        * @return the list of current option settings as an array of strings
        */
//@ ensures \result != null;
//@ ensures \nonnullelements(\result);
/*@pure@*/
       @SuppressWarnings({"UnusedAssignment"})
       public String[] getOptions()
       {
              String [] options = new String [2];
              int current = 0;
              options[current++] = "-" + E_PARAMETER_OPTION; options[current++] = "" + m_Epsilon;              
              return options;
       }



}
