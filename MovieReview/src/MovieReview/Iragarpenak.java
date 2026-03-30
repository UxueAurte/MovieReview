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
	
	public static void main () throws Exception {
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
		testSel.setClassIndex(testSel.numAttributes()-1);
		
		// MODELOA KARGATU
		SMO svm = (SMO) SerializationHelper.read("svm.model");
		
		//IRAGARPENAK EGIN
		PrintWriter writer = new PrintWriter(new FileWriter("iragarpenak.txt"));
		
		int posCount = 0;
		int negCount = 0;

        for (int i = 0; i < testSel.numInstances(); i++) {
            Instance inst = testSel.instance(i);

            double pred = svm.classifyInstance(inst);
            String label = testSel.classAttribute().value((int) pred);
            
            if (label.equals("pos")) posCount++;
            else if (label.equals("neg")) negCount++;

            writer.println(label);
        }
        
        writer.println("\n=== LABURPENA ===");
        writer.println("Positiboak: " + posCount);
        writer.println("Negatiboak: " + negCount);
        writer.println("Guztira: " + (posCount + negCount));
        writer.close();
        System.out.println("--------------------------------------");
        System.out.println("Iragarpenak sortuta: iragarpenak.txt");
        System.out.println("\nIRAGARPENAK");
        System.out.println("Positiboak: " + posCount);
        System.out.println("Negatiboak: " + negCount);
        System.out.println("Guztira: " + (posCount + negCount));
        
		
	}

}
