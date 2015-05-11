package technion.cs;

import java.io.Serializable;

/**
 * Created by IntelliJ IDEA.
 * Author: Arik Friedman
 * Date: 17/11/2009
 * An exception class for reporting that the privacy budget is exhausted
 * This type of exception is not used by technion.cs.PrivacyAgent, but other classes can use
 * it to report exhaustion of the privacy budget
 */
public class PrivacyBudgetExhaustedException extends IllegalAccessException  implements Serializable {
       public PrivacyBudgetExhaustedException() {super();}
       public PrivacyBudgetExhaustedException(String message) {super(message);}
}