package technion.cs.test;

import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import technion.cs.Scorer.GiniScorer;
import technion.cs.DiffPrivacyC45;

public class TestDiffPrivacyC45 {

	public static int seed = 1;
	//public static int maxIteration = 1000000;
	public static int maxDepth = 5;
	public static String epsilon = "1.0";

	public static void main(String[] args) throws Exception {

		maxDepth = Integer.parseInt(args[0]);
		epsilon  = args[1];

		System.out.printf("%d, %s, ",
						maxDepth, epsilon);
		
		Random random = new Random(seed);
		
		String trainDataPath = "dataset/vote_nomissing.arff";
		Instances trainData = null;
		
		trainData = (new DataSource(trainDataPath)).getDataSet();
		if (trainData.classIndex() == -1)
			trainData.setClassIndex(trainData.numAttributes() - 1);	
				
		DiffPrivacyC45 tree = new DiffPrivacyC45();
		
		tree.setMaxDepth(maxDepth);
		tree.setEpsilon(epsilon);
		tree.setSeed(random.nextInt());

        tree.setConfidenceFactor(0.25f);
        tree.setScorer(new GiniScorer());

        tree.setUnpruned(true);
		
		//tree.buildClassifier(trainData);
		
		Evaluation eval = new Evaluation(trainData);
		eval.crossValidateModel(tree, trainData, 10, new Random(random.nextInt()));
		
		System.out.println(eval.pctCorrect());
		
	}

}
