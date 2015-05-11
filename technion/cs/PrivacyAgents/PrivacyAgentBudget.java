package technion.cs.PrivacyAgents;

import technion.cs.PrivacyAgent;

import java.math.BigDecimal;

/**
  * Author: Arik Friedman
 * Date: 16/11/2009
 * To change this template use File | Settings | File Templates.
 */
public class PrivacyAgentBudget extends PrivacyAgent {

       private BigDecimal m_Budget;

       public PrivacyAgentBudget(BigDecimal initialBudget)
       {
              m_Budget =initialBudget;
       }

       /**
        * The request method is used to obtain a privacy budget for a specific query
        * operation. It gets as a parameter the amount of requested budget.
        * If the return value is true, than the query operation is approved, and the
        * budget is updated to reflect the budget use. A false return value means that
        * the existing budget is not sufficient to comply for the request, and no change
        * is made to the budget.
        *
        * @param reqBudget the request budget
        * @return true if budget is approved, false otherwise
        */
       @Override
       public boolean Request(BigDecimal reqBudget) {

            if (reqBudget.compareTo(m_Budget) >0 )     // BigDecimal equivalent of reqBudget>m_Budget
            {
                return false;
            }

            m_Budget=m_Budget.subtract(reqBudget);            
            return true;
        }

       /**
        * Inquire how much unused budget is available
        *
        * @return the amount of unused budget
        */
       @Override
       public BigDecimal RemainingBudget() {
              return m_Budget;
       }


}
