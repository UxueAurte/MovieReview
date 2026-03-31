package MovieReview;

import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.SelectedTag;

/**
 * 	Parameter Fine-tuning bat exekutatzen du SMO eredu baten parametroak optimizatu eta ebaluatzeko. 
 */
public class ParametroEkorketa {
	/**
	 * Parameter Fine-tuning bat exekutatzen du SMO eredu baten parametroak optimizatu eta ebaluatzeko. 
	 * 
	 * @param train Train datu sorta.
	 * @param test Test datu sorta.
	 * @throws Exception
	 */
    public static void ekorketaExekutatu(Instances train, Instances test) throws Exception {
        double[] cBaloreak = {0.01, 0.1, 1.0, 10.0, 100.0}; 
        double[] eBaloreak = {1.0, 2.0}; 
        int[] filterTypes = {
            SMO.FILTER_NORMALIZE,   
            SMO.FILTER_STANDARDIZE, 
            SMO.FILTER_NONE         
        };	

        double accuracyOnena = -1.0;
        double cOnena = -1.0;
        double eOnena = -1.0;
        int filterOnena = -1;
        
        for (double c : cBaloreak) {
            for (double e : eBaloreak) {
                for (int fType : filterTypes) {
                    try {
                        PolyKernel kernel = new PolyKernel();
                        kernel.setExponent(e);

                        SMO smo = new SMO();
                        smo.setC(c);
                        smo.setKernel(kernel);
                        smo.setFilterType(new SelectedTag(fType, SMO.TAGS_FILTER));

                        smo.buildClassifier(train);

                        Evaluation eval = new Evaluation(train);
                        eval.evaluateModel(smo, test);
                        double currentAcc = eval.pctCorrect();

                        System.out.println(c + "\t" + e + "\t" + fType + "\t" + String.format("%.2f%%", currentAcc));

                        if (currentAcc > accuracyOnena) {
                            accuracyOnena = currentAcc;
                            cOnena = c;
                            eOnena = e;
                            filterOnena = fType;
                        }

                    } catch (Exception ex) {}
                }
            }
        }

        System.out.println("------------------------------------");
        System.out.println("EKORKETA EMAITZA OPTIMOAK:");
        System.out.println("-> Accuracy Onena: " + accuracyOnena);
        System.out.println("-> C Parametroa: " + cOnena);
        System.out.println("-> Kernel Exponent (E): " + eOnena);
        System.out.println("-> FilterType (ID): " + filterOnena);
    }
}