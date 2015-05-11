package technion.cs.PrivacyAgents;

import technion.cs.PrivacyAgent;

import java.math.BigDecimal;

/**
 * Author: Arik Friedman
 * Date: 16/11/2009
 */
public class PrivacyAgentBinary extends PrivacyAgent {

       private PrivacyAgent m_AgentA;
       private PrivacyAgent m_AgentB;

       public PrivacyAgentBinary(PrivacyAgent agentA, PrivacyAgent agentB)
       {
              if (agentA==null || agentB==null)
                     throw new IllegalArgumentException("Null agent was passed as argument to PrivacyAgentBinary constructor");

              m_AgentA =agentA;
              m_AgentB =agentB;
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
       public boolean Request(BigDecimal budget)
       {
              if (!m_AgentA.Request(budget))
                     return false;

              if (!m_AgentB.Request(budget))
              {
                     m_AgentA.Request(budget.negate()); // rollback request to agent A
                     return false;
              }

              return true;
       }

       /**
        * Inquire how much unused budget is available
        *
        * @return the amount of unused budget
        */
       @Override
       public BigDecimal RemainingBudget() {
              BigDecimal agentAbudget=m_AgentA.RemainingBudget();
              BigDecimal agentBbudget=m_AgentB.RemainingBudget();
              if (agentAbudget.compareTo(agentBbudget)<0)
                     return agentBbudget;
              else return agentAbudget;
       }
}
