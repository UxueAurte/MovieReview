package MovieReview;

import java.io.FileWriter;
import java.io.PrintWriter;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;

public class KalitateEstimatua {
	public static void main (String[] args) throws Exception {
		
		// ====================================================
        // 1. Datuak kargatu
        // ====================================================
		DataSource trainSource = new DataSource("datuak/train.arff");
        Instances train = trainSource.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        DataSource devSource = new DataSource("datuak/dev.arff");
        Instances dev = devSource.getDataSet();
        dev.setClassIndex(dev.numAttributes() - 1);
        
        // ====================================================
        // 2. Kargatu aurretik sortutako filtroak
        // (vectorizer + attribute selection)
        // ====================================================
        
        Filter vectorizer = (Filter) SerializationHelper.read("vectorizer.model");
        Filter attrsel = (Filter) SerializationHelper.read("attrsel.model");
        
        // ====================================================
        // 3. Filtroak aplikatu
        // ====================================================
        vectorizer.setInputFormat(train);
        Instances trainBek = Filter.useFilter(train, vectorizer);
        Instances devBek = Filter.useFilter(dev, vectorizer);
      
        attrsel.setInputFormat(trainBek);
        Instances trainSel = Filter.useFilter(trainBek, attrsel);
        Instances devSel = Filter.useFilter(devBek, attrsel);
        
        // ====================================================
        // 4. Azken modeloa sortu
        // ====================================================
        
        SMO svm = new SMO();
        svm.setC(1.0); // Hemen parametro ekorketan lortutako parametro optimoa jarri
        
        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(2.0);
        svm.setKernel(kernel);
        
        // ====================================================
        // 5. Entrenatu
        // ====================================================
        
        svm.buildClassifier(trainSel);
        
        // ====================================================
        // 6. Ebaluatu
        // ====================================================
        
        Evaluation eval = new Evaluation(trainSel);
        eval.evaluateModel(svm, devSel);
        
        // ====================================================
        // 7. Emaitzak inprimatu eta txt batean gorde
        // ====================================================
        
        String txt = "kalitate_estimatua.txt";
        PrintWriter writer = new PrintWriter(new FileWriter(txt));
        
        writer.println("Kalitate estimatua");
        writer.println("Accuracy: " + eval.pctCorrect());
        writer.println("F1 (pos): " + eval.fMeasure(1));
        writer.println("\nNahasmen matrizea:");
        double[][] cm = eval.confusionMatrix();
        for (int i = 0; i < cm.length; i++) {
            for (int j = 0; j < cm[i].length; j++) {
                writer.print(cm[i][j] + " ");
            }
            writer.println();
        }
        
        writer.close();
        System.out.println("Emaitzak gordeta: " + txt);
        
        System.out.println("Kalitate estimatua");
        System.out.println("Accuracy: " + eval.pctCorrect());
        
        System.out.println("F1 (pos): " + eval.fMeasure(1));
        
        System.out.println("\nNahasmen matrizea:");
        double[][] nm = eval.confusionMatrix();

        for (int i = 0; i < nm.length; i++) {
            for (int j = 0; j < nm[i].length; j++) {
                System.out.print(nm[i][j] + " ");
            }
            System.out.println();
        }
      
      
        
	}

}
