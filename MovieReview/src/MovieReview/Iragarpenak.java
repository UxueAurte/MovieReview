package MovieReview;

import java.io.FileWriter;
import java.io.PrintWriter;

import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

public class Iragarpenak {
	
	public static void main (String[] args) throws Exception {
		// TEST_BLIND.ARFF KARGATU
		DataSource source = new DataSource("datuak/test_blind.arff");
		Instances test = source.getDataSet();
		test.setClassIndex(test.numAttributes()-1);
		
		// FILTROAK KARGATU
		Filter vectorizer = (Filter) SerializationHelper.read("vectorizer.model");
		Filter attrsel = (Filter) SerializationHelper.read("attrsel.model");
		
		// FILTROAK APLIKATU
		Instances testBek = Filter.useFilter(test, vectorizer);
		
		Instances testSel = Filter.useFilter(testBek, attrsel);
		
		// MODELOA KARGATU
		SMO svm = (SMO) SerializationHelper.read("svm.model");
		
		//IRAGARPENAK EGIN
		PrintWriter writer = new PrintWriter(new FileWriter("iragarpenak.txt"));

        for (int i = 0; i < testSel.numInstances(); i++) {
            Instance inst = testSel.instance(i);

            double pred = svm.classifyInstance(inst);
            String label = testSel.classAttribute().value((int) pred);

            writer.println(label);
        }

        writer.close();

        System.out.println("Iragarpenak sortuta: iragarpenak.txt");
		
	}

}
