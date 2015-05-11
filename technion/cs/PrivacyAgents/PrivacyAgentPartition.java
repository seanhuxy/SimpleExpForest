package technion.cs.PrivacyAgents;

import technion.cs.PrivacyAgent;

import java.util.Collections;
import java.util.Map;
import java.math.BigDecimal;

/**
 * Author: Arik Friedman
 * Date: 16/11/2009
 */
public class PrivacyAgentPartition extends PrivacyAgent {

       /** The original agent at the source of the partition */
       private PrivacyAgent m_SourceAgent;

       /** The maximal budget used so far by any of the partitions */
       private CommonBigDecimal m_Common;

       /** a Dictionary listing the used budget for each of the partitions */
       private Map<Object, BigDecimal> m_Budget;

       /** The key associated with a particular partition */
       private Object m_Key;

       public PrivacyAgentPartition(PrivacyAgent agent, Map<Object,BigDecimal> budget, Object key, CommonBigDecimal sharedMaxBudget)
       {
              if (agent==null || budget==null || key==null)
                     throw new IllegalArgumentException("Null argument was passed to PrivacyAgentPartition constructor");              

              m_SourceAgent =agent;
              m_Budget =budget;
              m_Common=sharedMaxBudget;              
              m_Key =key;              
              // ensure that the map includes an entry for the current key.
              // If not, create it with a zero entry (no budget consumed so far)
              if (!m_Budget.containsKey(m_Key))
                     m_Budget.put(m_Key,BigDecimal.valueOf(0.0));
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

              // The request budget will require budget beyond what was used so far
              BigDecimal effectiveBudget =m_Budget.get(m_Key).add(budget);
              if (effectiveBudget.compareTo(m_Common.maxBudget)>0) // BigDecimal equivalent of (effectiveBudget>m_MaxBudget)
              {
                     // if the source agent cannot support the extra budget needed, deny the request
                     if (!m_SourceAgent.Request(effectiveBudget.subtract(m_Common.maxBudget)))
                            return false;                     

                     // otherwise update the budgets to reflect the approval
                     m_Budget.put(m_Key, effectiveBudget);
                     m_Common.maxBudget=effectiveBudget;
                     return true;
              }

              // if we reduce the budget, and the maximal budget is determined by
              // the current agent, find a new max value
              if (m_Budget.get(m_Key)== m_Common.maxBudget && budget.signum()<0)  // signum checks if sign of budget is negative
                     m_Common.maxBudget=Collections.max(m_Budget.values());
              
              // update the budget and return success
              m_Budget.put(m_Key, effectiveBudget);
              return true;
       }

       /**
        * Inquire how much unused budget is available
        *
        * @return the amount of unused budget
        */
       @Override
       public BigDecimal RemainingBudget() {
              return m_SourceAgent.RemainingBudget().add(m_Common.maxBudget).subtract(m_Budget.get(m_Key));
       }

}
