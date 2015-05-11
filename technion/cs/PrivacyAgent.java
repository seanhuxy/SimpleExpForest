package technion.cs;

import java.math.BigDecimal;

/**
 * Author : Arik Friedman
 * Date: 16/11/2009
 *
 * technion.cs.PrivacyAgent is parallel to the PINQAgent class in the PINQ framework.
 * It keeps track of the provided privacy budget. The technion.cs.PrivacyAgent is consulted
 * every time access to the dataset is required. If the available budget can
 * accommodate the request, the request will be authorized and the remaining budget
 * will be updated accordingly. Otherwise, the request will be rejected.
 *  Different logic for budget management can be applied by subtyping the
 * technion.cs.PrivacyAgent class with different implementations for the Request method
 */
public abstract class PrivacyAgent {

        /**
         * The request method is used to obtain a privacy budget for a specific query
         * operation. It gets as a parameter the amount of requested budget. This
         * method is equivalent to the Alert/Apply method presented in the PINQ framework.
         * If the return value is true, than the query operation is approved, and the
         * budget is updated to reflect the budget use. A false return value means that
         * the existing budget is not sufficient to comply for the request, and no change
         * is made to the budget.
         * It is possible to pass a negative budget to rollback an operation (it is assumed
         * that only trusted code accesses this method). Subclasses of technion.cs.PrivacyAgent
         * should respect a negative budget and increase the budget. Invocations of
         * Request with negative value are not expected to check the return value.
         *
         * BigDecimal is used for budget rather than decimal to provide  higher level
         * of accuracy (budget may be consumed in extremely small chunks)
         *
         * @param budget the request budget
         * @return true if budget is approved, false otherwise
         */
       public abstract boolean Request(BigDecimal budget);

       /**
        * Inquire how much unused budget is available
        * @return the amount of unused budget
        */
       public abstract BigDecimal RemainingBudget();
}
