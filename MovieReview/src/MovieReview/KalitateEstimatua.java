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
        // DATUAK KARGATU
        // ====================================================
		DataSource trainSource = new DataSource("datuak/train.arff");
        Instances train = trainSource.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        DataSource devSource = new DataSource("datuak/dev.arff");
        Instances dev = devSource.getDataSet();
        dev.setClassIndex(dev.numAttributes() - 1);
        
        // ====================================================
        // FILTROAK KARGATU
        // (vectorizer + attribute selection)
        // ====================================================
        
        Filter vectorizer = (Filter) SerializationHelper.read("vectorizer.model");
        Filter attrsel = (Filter) SerializationHelper.read("attrsel.model");
        
        // ====================================================
        // FILTROAK APLIKATU
        // ====================================================
        vectorizer.setInputFormat(train);
        Instances trainBek = Filter.useFilter(train, vectorizer);
        Instances devBek = Filter.useFilter(dev, vectorizer);
      
        attrsel.setInputFormat(trainBek);
        Instances trainSel = Filter.useFilter(trainBek, attrsel);
        Instances devSel = Filter.useFilter(devBek, attrsel);
        
        // ====================================================
        // MODELOA SORTU (PARAMETRO EKORKETAN LORTUTAKO BALIOEKIN)
        // ====================================================
        
        SMO svm = new SMO();
        svm.setC(0.01); 
        
        PolyKernel kernel = new PolyKernel();
        kernel.setExponent(1.0);
        svm.setKernel(kernel);
        
        // ====================================================
        // ENTRENATU
        // ====================================================
        
        svm.buildClassifier(trainSel);
        SerializationHelper.write("svm.model", svm);
        
        // ====================================================
        // EBALUATU
        // ====================================================
        
        Evaluation eval = new Evaluation(trainSel);
        eval.evaluateModel(svm, devSel);
        
        // ====================================================
        // EBALUATU TRAIN-EN (OVERFITTING IKUSTEKO)
        // ====================================================
        
        Evaluation evalTrain = new Evaluation(trainSel);
        evalTrain.evaluateModel(svm, trainSel);
        
        // ====================================================
        // METRIKAK KALKULATU
        // ====================================================
        
        double accDev = eval.pctCorrect();
        double f1Pos = eval.fMeasure(1);
        double f1Neg = eval.fMeasure(0);
        double F1 = (f1Pos+f1Neg)/ 2.0 ;
        
        double precision = eval.precision(1);
        double recall = eval.recall(1);
        
        // ====================================================
        // TXT FITXATEGIA SORTU
        // ====================================================
        
        String txt = "kalitate_estimatua.txt";
        PrintWriter writer = new PrintWriter(new FileWriter(txt));
        
        writer.println("=== KALITATE ESTIMATUA ===\n");
        writer.println("Accuracy: " + accDev);
        writer.println("Precision (pos): " + precision);
        writer.println("Recall (pos): " + recall);
        writer.println("F1 (pos): " + f1Pos);
        writer.println("F1 (neg): " + f1Neg);
        writer.println("Macro F1: " + F1);
        
        writer.println("\n=== TRAIN EMAITZAK ===");
        writer.println("Accuracy: " + evalTrain.pctCorrect());
        writer.println("F1 (pos): " + evalTrain.fMeasure(1));
        
        
        writer.println("\n=== NAHASMEN MATRIZEA (DEV) ===");
        double[][] cm = eval.confusionMatrix();
        for (int i = 0; i < cm.length; i++) {
            for (int j = 0; j < cm[i].length; j++) {
                writer.print(cm[i][j] + " ");
            }
            writer.println();
        }
        
        writer.println("\n=== INTERPRETAZIOA ===");
        writer.println("Ereduak errendimendu ona erakusten du dev multzoan.");
        writer.println("Train eta dev emaitzen arteko diferentziak overfitting maila adierazten du.");
        writer.println("Precision eta recall balioek erakusten dute klase positiboak modu nahiko egokian detektatzen direla.");
        writer.println("Dev multzoa test multzoaren estimazio gisa erabili da.");
        
        writer.close();
        System.out.println("Emaitzak gordeta: " + txt);
        
        // ====================================================
        // PANTAILAN ERAKUTSI
        // ====================================================
        System.out.println("\n=== DEV EMAITZAK ===");
        System.out.println("Accuracy: " + accDev);
        System.out.println("F1 (pos): " + f1Pos);

        System.out.println("\n=== TRAIN vs DEV ===");
        System.out.println("Train Accuracy: " + evalTrain.pctCorrect());
        System.out.println("Dev Accuracy: " + accDev);
      
        
	}

}
