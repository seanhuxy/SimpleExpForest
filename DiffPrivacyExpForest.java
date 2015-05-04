
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;
import java.util.Random;
import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.io.Serializable;

import com.sun.corba.se.impl.javax.rmi.CORBA.Util;

import weka.classifiers.AbstractClassifier;
import weka.core.*;

public class DiffPrivacyExpForest extends AbstractClassifier{
	
	private static final long serialVersionUID = 1L;

	// default is round down, to avoid exhausting the privacy budget
    public static final MathContext MATH_CONTEXT = 
    			new MathContext(20, RoundingMode.DOWN); 

	private BigDecimal m_Epsilon = new BigDecimal(1.0);
	
	private int m_MaxDepth = 5;
	
	private int m_MaxIteration = 100000;
	
	private double m_EquilibriumThreshold = 0.0005;
	
	private Random m_Random = new Random();
	
	private ScoreFunc m_ScoreFunc = new ScoreFunc();
	
	private Node m_Root;
	private Set<Node> m_InnerNodes = new HashSet<Node>();
	
	public void setEpsilon(double e){
		m_Epsilon = new BigDecimal(e);
	}
	
	public void setMaxIteration(int it){
		m_MaxIteration = it;
	}
	
	public void setMaxDepth(int maxDepth){
		m_MaxDepth = maxDepth;
	}
	
	public void setSeed(int seed){
		m_Random = new Random(seed);
	}
	
	public class ScoreFunc implements Serializable{
		
		private static final long serialVersionUID = 1L;

		public double sensitivity(){
			return 2.;
		}
		
		private double leafScore(Node node){
		
			assert( node.splitAttr == null);
			double m_Score = 1.;
			int numInst = node.data.numInstances();
			
			//System.out.format("numInst:%d\n", numInst);
			
			if(numInst == 0){
				return 0.;
			}
			
			double count[] = node.count;
			for(int i=0; i< count.length; ++i){
				if( count[i] > 0.){
					//System.out.format("		count[%d]=%.1f	", i, count[i]);
					
					double part = Math.pow( count[i]/numInst, 2. );
					
					//System.out.format("%.2f\n", part);
					
					m_Score -= part;
				}
			}
			
			//System.out.format("leafScore: %.2f\n\n", m_Score);
			
			assert( m_Score > 0.0 && m_Score < 1.0);
			return m_Score;
		}
		
		public double score(Node node){
			double m_Score = 0.;
			
			if(node.data.numInstances() <= 0)
				return 0.;
			
			if(node.splitAttr == null)
			{
				m_Score = leafScore(node);
				//System.out.format("%.2f\n",m_Score);
				return m_Score;
			}
			
			for(Node child : node.children)
			{
				
				double part = score(child);
				
				//System.out.format("		child:%.2f\n", part);
				
				m_Score += ( (double)child.data.numInstances()
						 	/(double)node.data.numInstances() )
						 	*part;
				
				//System.out.format("m_score: %.2f\n", m_Score);
			}
			return m_Score;
		}
	}

	public class Node implements Serializable{
		
		private static final long serialVersionUID = 1L;

		//public Set<Attribute> attrs;	// XXX tend to remove
		
		public Instances data;
		
		public int depth;
		
		// inner node
		public Attribute splitAttr;
		
		public Node parent = null;
		
		public int index = 0;
		
		public Node[] children;
		
		// leaf node
		public double[] count;
		public double[] dist;
		
		public void updateDist(){
			dist = count.clone();
			if( Utils.sum(dist) != 0.0)
				Utils.normalize(dist);
		}
		
		public Node(Instances data, Node parent, int index){
			
			this.data = data;
			this.parent = parent;
			this.index = index;
			
			this.depth = 0;
			if(parent!=null)
				this.depth = parent.depth+1;
			
		}
		
		/**
		 * Deep copy Node other to this node
		 * @param node
		 */
		public Node(Node other){
			
			/*
			attrs = new HashSet<Attribute>();
			for(Attribute a: other.attrs){
				attrs.add(a);
			}*/
			
			data = new Instances(other.data); // deep copy a list of instance
			depth = other.depth;
			splitAttr = other.splitAttr;
			
			//parent = other.parent;
			//index = other.index;
			
			/*
			if( other.children != null)
			{
				children = new Node[other.children.length];
				for(int i=0; i<children.length; i++)
				{
					children[i] = new Node(other.children[i]);
				}
			}*/
			/*
			if( other.count != null){
				count = other.dist; // XXX
				dist = new double[other.dist.length];
				
			}*/
		}
		
	}
	/**
	 * Remove the attributes of Node {@code node} as well as those of
	 *  {@code node}'s descendants from {@code attrs}.
	 * @param attrs the set of attributes
	 * @param node the node whose attribute and whose descendants' 
	 * 		  attributes should be removed from {@code attrs}
	 */
	private void rmChildrenAttrs(Set<Attribute> attrs, Node node){

		if( node == null || node.splitAttr == null ) return;
		
		// remove its own attributes
		attrs.remove(node.splitAttr);
		
		if( node.children == null) return;
		
		// remove children's attributes
		for(Node child : node.children)			
			rmChildrenAttrs(attrs, child);
	}
	
	private void rmParentAttrs(Set<Attribute> attrs, Node node){
	
		if( node == null) return;
		
		Node parent = node.parent;
		while(parent != null)
		{
			attrs.remove( parent.splitAttr );
			parent = parent.parent;
		}
	}
	
	/**
	 * random select an attribute to swap with the split
	 * attribute Ai of {@code node}. The attribute should not be the split 
	 * attribute of any ancestors or descendants of {@code node}.
	 * so we should first remove those attributes from the candidate
	 * attribute set. (see method removeNodeAttr)
	 * @param node the node that the chosen attribute is going to
	 * @param attrs the set of attribute to be chosen from
	 * @return
	 */
	/*
	private Attribute randomSelectAttr(Set<Attribute> attrs){
		
		// randomly select an attribute from attrs
		Attribute attr 
			= (Attribute)attrs.toArray()[m_Random.nextInt(attrs.size())];
		return attr;
	}*/

	private Instances[] partitionByAttr(Instances data, Attribute attr){
		
		Instances[] parts = new Instances[attr.numValues()];
		for(int i=0; i<parts.length; i++)
		{
			parts[i] = new Instances( data, data.numInstances() );
		}
		
		//System.out.println("attr: numValues:"+attr.numValues());
		
		Enumeration<Instance> instEnum = data.enumerateInstances();
		while(instEnum.hasMoreElements())
		{
			Instance inst = instEnum.nextElement();
			parts[(int)inst.value(attr)].add(inst);
		}
		
		return parts;
	}
	/**
	 * @param node
	 * set count and dist
	 */
	private void makeLeafNode(Node node){
		node.splitAttr = null;
		node.children  = null;
		
		double[] count = new double[node.data.numClasses()];
		Enumeration<Instance> instEnum = node.data.enumerateInstances();
		while(instEnum.hasMoreElements()){
			Instance inst = (Instance)instEnum.nextElement();
			count[(int)inst.classValue()]++;
		}
		
		node.count = count;
		
		node.dist = count.clone();
		if( Utils.sum(node.dist) != 0.0)
			Utils.normalize(node.dist);
		
		return;
	}
	
	/**
	 * Split {@code node}. Continuously split {@code node}
	 * and its descendants by randomly pick an attribute from 
	 * attribute pool as the split attribute, until the max 
	 * depth is reached.
	 * @param node the node to split
	 */
	private void splitNode(Node node, Set<Attribute> attrs){
		
		Instances data = node.data;
		
		// make a leaf node
		if( node.depth >= m_MaxDepth){
			makeLeafNode(node);
			return;
		}
		
		// randomly choose an attribute as the split attribute
		Attribute splitAttr = (Attribute)attrs.toArray()[m_Random.nextInt(attrs.size())];
		
		node.splitAttr = splitAttr;
		
		// partition instances to N sets, where N is the numValues of splitAttr
		Instances[] parts = partitionByAttr(data, splitAttr);
		Node[] children = new Node[splitAttr.numValues()];
		
		// build child trees
		attrs.remove(splitAttr);
		for(int i=0; i < parts.length; i++){
			children[i] = new Node( parts[i], node, i);
			splitNode(children[i], attrs);
		}
		attrs.add(splitAttr);
		node.children = children;
	}
	
	private void redistribute(Node node, Node orgNode){
	
		Attribute splitAttr = node.splitAttr;
		
		if( node.depth >= m_MaxDepth){
			makeLeafNode(node);
			return;
		}
		
		//Node[] children = node.children;
		Node[] children = new Node[splitAttr.numValues()];
		Instances[] parts = partitionByAttr(node.data, splitAttr);
		
		for(int i=0; i < parts.length; i++){
			children[i] = new Node(orgNode.children[i]);
			children[i].data = parts[i];
			children[i].depth = orgNode.children[i].depth;
			children[i].splitAttr = orgNode.children[i].splitAttr;
			children[i].parent = node;
			children[i].index = i;
			
			redistribute(children[i], orgNode.children[i]);
		}

		node.children = children;
	}
	
	private Node createNewNode(Node orgNode, Set<Attribute> attrs, Attribute newAttr){
		
		Node newNode = new Node(orgNode);
		newNode.splitAttr = newAttr;
		newNode.parent = orgNode.parent;
		newNode.index  = orgNode.index;
		
		Node[] children = new Node[newAttr.numValues()];
		Instances[] parts = partitionByAttr(newNode.data, newAttr);
		
		int minlen = (children.length < orgNode.children.length)
					 ?children.length : orgNode.children.length;
		
		for(int i=0; i < minlen; i++)
		{
			children[i] = new Node(orgNode.children[i]); // deep copy
			children[i].data = parts[i];
			children[i].depth = orgNode.children[i].depth;
			children[i].splitAttr = orgNode.children[i].splitAttr;
			children[i].parent = newNode;
			children[i].index = i;
			
			redistribute(children[i], orgNode.children[i]);
		}
		
		if( orgNode.children.length < children.length )
		{	
			attrs.remove(newAttr);
			for(int i= orgNode.children.length; i < children.length; ++i)
			{
				children[i] = new Node(parts[i], newNode, i);
				
				splitNode(children[i], attrs);
			}
		}

		newNode.children = children;
		
		return newNode;
	}
	/*
	private double getNewScore(double totalScore, Node orgNode, Node newNode){
		
		double totalSize = (double)m_Root.data.numInstances();
		double nodeSize = (double)newNode.data.numInstances();
		
		double orgScore = m_ScoreFunc.score(orgNode);
		double newScore = m_ScoreFunc.score(newNode);
		
		//System.out.printf("orgScore:%.2f, newScore:%.2f\n", orgScore, newScore);
		//System.out.printf("nodesize:%.0f, totalSize: %.0f\n", nodeSize, totalSize);
		
		//System.out.printf("global diff:%.8f\n\n", (nodeSize/totalSize)*(newScore-orgScore));
		
		double newTotalScore = totalScore + (nodeSize/totalSize)*(newScore-orgScore);
		return newTotalScore;
	}*/
	
	private double expScoreFunc(double score, BigDecimal e)
	{
		return Math.exp( 
					(e.doubleValue() * score) 
					/ (2. * m_ScoreFunc.sensitivity()) 
				);
	}
	/**
	 * check if the tree reaches a state of equilibrium.
	 * specifically, if the maximum score and the minimum 
	 * score of last N tree didn't exceed ..., we assume the
	 * tree achieves a state of equilibrium
	 * @param new_score
	 * @return
	 */
	final int s_BufferSize = 1000;
    int s_InitPointer = 0;
	int s_Pointer = 0;
	double[] s_ScoreBuffer = new double[s_BufferSize];
	
	double variance = -1.;
	
	private double deltaVariance(double[] arr, int index, double y){
		
		double n = (double)arr.length;
		double x = arr[index];
		
		//arr[index] = y;
		
		double oldSum = Utils.sum(arr);
		double newSum = oldSum + y -x;
		
		double r = ( y*y - x*x)/n + Math.pow(newSum,2) - Math.pow(oldSum,2) -
					2*(y-x)*(newSum+oldSum)/Math.pow(n,2);
		return r;
	}
	
	private boolean isEquilibrium(double newScore){
		
		if( s_InitPointer < s_BufferSize ){
			s_ScoreBuffer[s_InitPointer] = newScore;
			++ s_InitPointer;
			
			variance = -1.;
			return false;
		}
		assert( s_InitPointer == s_BufferSize);
		
		if( s_Pointer >= s_BufferSize){
			assert( s_Pointer == s_BufferSize );
			s_Pointer = 0;
		}
		/*
		if( variance < 0){
			variance = Utils.variance(s_ScoreBuffer);
		}
		else{
			variance += deltaVariance(s_ScoreBuffer, s_Pointer, newScore);
		}*/
		
		s_ScoreBuffer[s_Pointer] = newScore;
		++ s_Pointer;
		
		variance = Utils.variance(s_ScoreBuffer);
		System.out.printf("  Variance: %f", variance);
		
		if( variance < m_EquilibriumThreshold){
			//variance = -1.;
			s_InitPointer = 0;
			s_Pointer = 0;
			return true;
		}
		return false;
		
		/*
		int maxI = Utils.maxIndex(s_ScoreBuffer);
		int minI = Utils.minIndex(s_ScoreBuffer);

		double diff = s_ScoreBuffer[maxI]-s_ScoreBuffer[minI];
		if( diff < m_EquilibriumThreshold){
			return true;
		}
		return false;*/
	}
	
	private void rmFromInnerNodes(Node node){
		if(node.splitAttr == null) return;
		
		if(node.children != null)
		{
			for(Node child: node.children)
			{
				rmFromInnerNodes(child);
			}
		}

		m_InnerNodes.remove(node);
	}

	private void addToInnerNodes(Node node){
		
		if(node.splitAttr == null) return;
		
		if(node.children != null)
		{
			for(Node child: node.children)
			{
				addToInnerNodes(child);
			}
		}
		
		m_InnerNodes.add(node);
	}
	
    public void buildClassifier(Instances data) throws Exception{
        
    	// allAttributes
        HashSet<Attribute> allAttributes = new HashSet<Attribute>();
        Enumeration<Attribute> attrEnum = data.enumerateAttributes();
        while( attrEnum.hasMoreElements()){
        	allAttributes.add( attrEnum.nextElement() );
        }

        BigDecimal budget1 = m_Epsilon.divide(BigDecimal.valueOf(2));
        BigDecimal budget2 = m_Epsilon.divide(BigDecimal.valueOf(2));

        // build a random tree
        m_Root = new Node(data, null, 0);
        splitNode(m_Root, allAttributes);
        addToInnerNodes(m_Root);
        
        double totalScore = m_ScoreFunc.score(m_Root);
        
        //System.out.println("Init Score:"+totalScore);
       
        boolean b_Equilibrium = false;
        
        // iteratively switch attribute
        int iteration = 0;
        while( iteration++ < m_MaxIteration && !b_Equilibrium ){  
        	
        	//System.out.println("Iteration:"+iteration);
        	
            // randomly select an inner node from the tree
        	Node node = (Node)m_InnerNodes.toArray()
        				[m_Random.nextInt(m_InnerNodes.size())];
        	
    		Set<Attribute> subAttrs = new HashSet<Attribute>(allAttributes);
    		//subAttrs.addAll(allAttributes);
    		rmParentAttrs(subAttrs, node);
    		
    		Set<Attribute> attrsWoParents = new HashSet<Attribute>(subAttrs);
    		
    		rmChildrenAttrs(subAttrs, node);
        	
            // randomly select attribute in the attribute pool
    		Attribute Aj;
    		if( subAttrs.size() == 0){
    			Aj = (Attribute)attrsWoParents.toArray()[m_Random.nextInt(attrsWoParents.size())];
    		}else
    			Aj = (Attribute)subAttrs.toArray()[m_Random.nextInt(subAttrs.size())];
        	
   
    		//subAttrs = new HashSet<Attribute>(allAttributes);
    		//subAttrs.addAll(allAttributes);
    		//rmParentAttrs(subAttrs, node);
    		
        	Node orgNode = node;
        	Node newNode = createNewNode(node, attrsWoParents, Aj); 
        	
        	double orgScore = m_ScoreFunc.score(orgNode);
        	double newScore = m_ScoreFunc.score(newNode);
        			
        	double orgExpScore = expScoreFunc(orgScore, budget1);
    		double newExpScore = expScoreFunc(newScore, budget1);
    		
    		totalScore = m_ScoreFunc.score(m_Root);
    		
    		//System.out.format("Iter[%d]  total:%.5f  org:%.5f new:%.5f  variance:%.5f", 
    		//		iteration, totalScore, orgScore, newScore, variance);
    		
    		//System.out.format("Iter[%d]org:%.5f new:%.5f\n", 
    	    //				iteration, orgScore, newScore);
    		//System.out.format("        exp:%.5f exp:%.5f\n", 
    		//							orgExpScore, newExpScore);
    		
    		//double prop = Math.min(1., newExpScore/orgExpScore);
    		
    		//boolean b_Replace = ( m_Random.nextDouble() <= Math.min(1., newExpScore/orgExpScore) );
    		
    		boolean b_Replace = ( m_Random.nextDouble() <= newExpScore/(newExpScore+orgExpScore) );
    		
    		//boolean b_Replace = ( m_Random.nextDouble() <= newScore/(newScore+orgScore) );
    		//boolean b_Replace = (newExpScore>orgExpScore);
    		
    		if(b_Replace)
    		{
    			//System.out.println(" : replaced");
    			
    			Node parent = orgNode.parent;
    			
    			if( parent == null){
    				m_Root = newNode;
    			}else{
	    			// replace orgNode by newNode
	    			parent.children[orgNode.index] = newNode;
    			}
    			
    			//System.out.printf("prevNode: %f  postNode: %f\n", orgScore, newScore);
    			
    			System.out.printf("prev: %f  ", totalScore);
    			
    			// check if the tree reaches a state of equilibrium.
    			totalScore = m_ScoreFunc.score(m_Root);
    			
    			System.out.printf("post: %f  ", totalScore);
    			
        		b_Equilibrium = isEquilibrium(totalScore);
        		
        		System.out.printf("\n");
        		
        		// remove old nodes from m_InnerNode
        		rmFromInnerNodes(orgNode);
        		addToInnerNodes(newNode);
    		}
    		else{
    			//System.out.println();
    		}
        }

        // adding Laplace noise in the leaf node
        addNoise(m_Root, budget2);
    }
    
    private double laplace(BigDecimal bigBeta){
    	
    	double miu = 0.;
    	
    	double beta=bigBeta.doubleValue();
    	
        double uniform= m_Random.nextDouble()-0.5;
        return miu-beta*((uniform>0) ? -Math.log(1.-2*uniform) : Math.log(1.+2*uniform));
    }
    
    private void addNoiseDistribution(double[] count, BigDecimal budget){
    	
    	int maxIndex = Utils.maxIndex(count);
    	
    	for(int i=0; i<count.length; i++)
    	{
    		count[i] += laplace(BigDecimal.ONE.divide(budget, MATH_CONTEXT));
 
    		if(count[i] < 0.)	
    			count[i] = 0.;
    	}
 
    	double sum = Utils.sum(count);
    	if(sum <= 0.){	
    		count[maxIndex] = 1.0;
    	}
    }
    
    private void addNoise(Node node, BigDecimal budget){
    	
    	// for leaf node
    	if(node.splitAttr == null)
    	{
    		addNoiseDistribution(node.count, budget);
    		node.updateDist();
	    	return;
    	}
		// for inner node
    	for(Node child : node.children)
    	{
    		addNoise(child, budget);
    	}
    }
    
    public double classifyInstance(Instance instance)
            throws NoSupportForMissingValuesException { 	
    	assert( instance.hasMissingValue() == false);
    	
    	Node node = m_Root;
    	while(node.splitAttr != null){
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return Utils.maxIndex(node.dist);
    }
		
	public double[] distributionForInstance(Instance instance)
	            throws NoSupportForMissingValuesException {	
		assert( instance.hasMissingValue() == false);
		
		Node node = m_Root;
    	while(node.splitAttr != null){
    		Attribute attr = node.splitAttr;
    		node = node.children[ (int)instance.value(attr) ];
    	}
    	
    	return node.dist;
	}
	
	
    /**
     * Prints the decision tree using the private toString method from below.
     * Function altered with respect to original in C4.5 - epsilon parameter is output as well
     *
     * @return a textual description of the classifier
     */
    public String toString() {

         return toString(m_Root);
    }

	/**
	 * Outputs a tree at a certain level.
	 *
	 * @param level
	 *            the level at which the tree is to be printed
	 * @return the tree as string at the given level
	 */
	protected String toString(Node node) {
		
		int level = node.depth;

		StringBuffer text = new StringBuffer();

		if (node.splitAttr == null) {
			

			text.append("  [" + node.data.numInstances() + "]");
			text.append(": ").append(
					node.data.classAttribute().value((int) Utils.maxIndex(node.dist)));
			
			text.append("   Counts  " + distributionToString(node.dist));
			
		} else {
			
			text.append("  [" + node.data.numInstances() + "]");
			for (int j = 0; j < node.children.length; j++) {
				
				text.append("\n");
				for (int i = 0; i < level; i++) {
					text.append("|  ");
				}
				
				text.append(node.splitAttr.name())
					.append(" = ")
					.append(node.splitAttr.value(j));
				
				text.append(toString(node.children[j]));
			}
		}
		return text.toString();
	}

    private String distributionToString(double[] distribution)
    {
           StringBuffer text = new StringBuffer();
           text.append("[");
           for (double d:distribution)
                  text.append(String.format("%.2f", d) + "; ");
           text.append("]");
           return text.toString();             
    }
	
}
