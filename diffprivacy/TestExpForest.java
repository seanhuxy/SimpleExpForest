package diffprivacy;

import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import diffprivacy.DiffPrivacyExpForest;

public class TestExpForest {

	public static int seed = 1;
	public static int maxIteration = 1000000;
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
		
		DiffPrivacyExpForest tree = new DiffPrivacyExpForest();
		
		tree.setMaxIteration(maxIteration);
		tree.setMaxDepth(maxDepth);
		tree.setEpsilon(epsilon);
		
		tree.setSeed(random.nextInt());
		
		//tree.buildClassifier(trainData);
		
		Evaluation eval = new Evaluation(trainData);
		eval.crossValidateModel(tree, trainData, 10, new Random(random.nextInt()));
		
		System.out.println(eval.pctCorrect());
		
	}

}
