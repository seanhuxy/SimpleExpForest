package diffprivacy;

import java.util.Random;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;

import diffprivacy.DiffPrivacyExpForest;

public class TestExpForest {

	public static void main(String[] args) throws Exception {
		
		//String[] scorers = ["InfoGain","Max","Gini"];

		int seed = 1;
		Random random = new Random(seed);
		
		String trainDataPath = "dataset/vote_nomissing.arff";
		//String attrFile = "E:\\Lectures\\TCloud\\dataset\\adult_nomissing_attr.txt";
		Instances trainData = null;
		
		//String testDataPath = "E:\\Lectures\\TCloud\\dataset\\bank-new.arff";
		//Instances testData  = null;
		
		int maxDepth = 5;
		
		trainData = (new DataSource(trainDataPath)).getDataSet();
		if (trainData.classIndex() == -1)
			trainData.setClassIndex(trainData.numAttributes() - 1);	
		
//		testData = (new DataSource(testDataPath)).getDataSet();
//		if (testData.classIndex() == -1)
//			testData.setClassIndex(testData.numAttributes() - 1);	
		
		DiffPrivacyExpForest tree = new DiffPrivacyExpForest();
		
		tree.setMaxIteration(100000);
		tree.setMaxDepth(maxDepth);
		tree.setEpsilon(1.0);
		
		tree.setSeed(random.nextInt());
		
		tree.buildClassifier(trainData);
		
		System.out.println("done");
		
		//System.out.println(tree.toString());
		
		Evaluation eval = new Evaluation(trainData);
		eval.crossValidateModel(tree, trainData, 10, new Random(random.nextInt()));
		
		System.out.println("Accuracy is "+eval.pctCorrect());
		
	}

}
