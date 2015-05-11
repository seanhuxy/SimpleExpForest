package technion.cs.PrivacyAgents;

import technion.cs.PrivacyAgent;

import java.math.BigDecimal;

/**
  * Author: Arik Friedman
 * Date: 16/11/2009
 * A privacy agent 
 */
public class PrivacyAgentUnary extends PrivacyAgent {

       private PrivacyAgent m_Agent;
       private double m_Scale;

       public PrivacyAgentUnary(PrivacyAgent agent, double scale)
       {
              if (agent==null)
                     throw new IllegalArgumentException("Null agent was passed as argument to PrivacyAgentUnary constructor");
              if (scale<=0)
                     throw new IllegalArgumentException("Negative or zero scale was passed as argument to PrivacyAgentUnary constructor");
              
              m_Agent =agent;
              m_Scale =scale;
       }

       /**
        * The request method is used to obtain a privacy budget for a specific query
        * operation. It gets as a parameter the amount of requested budget. 
        * If the return value is true, than the query operation is approved, and the
        * budget is updated to reflect the budget use. A false return value means that
        * the existing budget is not sufficient to comply for the request, and no change
        * is made to the budget.
        *
        * @param budget the request budget
        * @return true if budget is approved, false otherwise
        */
       @Override
       public boolean Request(BigDecimal budget) {
              return m_Agent.Request(budget.multiply(BigDecimal.valueOf(m_Scale)));
       }

       /**
        * Inquire how much unused budget is available
        *
        * @return the amount of unused budget
        */
       @Override
       public BigDecimal RemainingBudget() {
              return m_Agent.RemainingBudget().divide(BigDecimal.valueOf(m_Scale));
       }
}
