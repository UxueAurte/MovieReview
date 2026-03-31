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
		
        // DATUAK KARGATU
		DataSource trainSource = new DataSource("datuak/train.arff");
        Instances train = trainSource.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        DataSource devSource = new DataSource("datuak/dev.arff");
        Instances dev = devSource.getDataSet();
        dev.setClassIndex(dev.numAttributes() - 1);
        
        // FILTROAK KARGATU (vectorizer + attribute selection)
        Filter vectorizer = (Filter) SerializationHelper.read("vectorizer.model");
        Filter attrsel = (Filter) SerializationHelper.read("attrsel.model");
        
        // FILTROAK APLIKATU
        vectorizer.setInputFormat(train);
        Instances trainBek = Filter.useFilter(train, vectorizer);
        Instances devBek = Filter.useFilter(dev, vectorizer);
      
        attrsel.setInputFormat(trainBek);
        Instances trainSel = Filter.useFilter(trainBek, attrsel);
        Instances devSel = Filter.useFilter(devBek, attrsel);
        
        // MODELOA SORTU (PARAMETRO EKORKETAN LORTUTAKO BALIOEKIN)
        SMO svm = new SMO();
        svm.setC(0.01); 
        
        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(1.0);
        svm.setKernel(kernel);
        
        // ENTRENATU
        svm.buildClassifier(trainSel);
        SerializationHelper.write("svm.model", svm);
        
        // EBALUATU 
        Evaluation eval = new Evaluation(trainSel);
        eval.evaluateModel(svm, devSel);
        
        // EBALUATU TRAIN-EN 
        Evaluation evalTrain = new Evaluation(trainSel);
        evalTrain.evaluateModel(svm, trainSel);
        
        // METRIKAK KALKULATU
        double accDev = eval.pctCorrect();
        double fPos = eval.fMeasure(0);
        double fNeg = eval.fMeasure(1);
        double F1 = (fPos+fNeg)/ 2.0 ;
        
        double precision = eval.precision(1);
        double recall = eval.recall(1);
        double errorRate = eval.errorRate();
        
        // TXT FITXATEGIA SORTU
        String txt = "kalitate_estimatua.txt";
        PrintWriter writer = new PrintWriter(new FileWriter(txt));
        
        writer.println("=== KALITATE ESTIMATUA ===\n");
        writer.println("Accuracy: " + accDev);
        writer.println("Precision (pos): " + precision);
        writer.println("Recall (pos): " + recall);
        writer.println("F-Score (pos): " + fPos);
        writer.println("F-Score (neg): " + fNeg);
        writer.println("Macro F-Score: " + F1);
        writer.println("Error rate: "+ errorRate);
        
        
        
        double[][] cm = eval.confusionMatrix();

	    // Klaseen izenak hartu (automatikoki!)
	    String pos = devSel.classAttribute().value(0);
	    String neg = devSel.classAttribute().value(1);
	    writer.println("\n=== NAHASMEN MATRIZEA (DEV) ===\n");
	
	    writer.println("            Aurreikuspena");
	    writer.println("            " + pos + "\t" + neg);
	
	    writer.println("Erreala " + pos + "\t" + (int)cm[0][0] + "\t" + (int)cm[0][1]);
	    writer.println("        " + neg + "\t" + (int)cm[1][0] + "\t" + (int)cm[1][1]);
        
        writer.println("\n=== INTERPRETAZIOA ===");
        writer.println("Ereduak errendimendu ona erakusten du dev multzoan.");
        writer.println("Precision eta recall balioek erakusten dute klase positiboak modu nahiko egokian detektatzen direla.");
        writer.println("Dev multzoa test multzoaren estimazio gisa erabili da.");
        
        writer.close();
        System.out.println("Emaitzak gordeta: " + txt);
        
        // PANTAILAN ERAKUTSI
        System.out.println("-----------------------------------------");
        System.out.println("KALITATE ESTIMATUA ");
        System.out.println("Accuracy: " + accDev);
        System.out.println("F-Score (pos): " + fPos);
        System.out.println("Error rate: "+ errorRate);

   
      
        
	}

}
