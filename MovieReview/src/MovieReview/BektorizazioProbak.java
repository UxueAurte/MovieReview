package MovieReview;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;

import weka.classifiers.Classifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;

public class BektorizazioProbak {

    public static void main(String[] args) throws Exception {

        // 📥 Cargar datasets
        Instances train = new DataSource("datuak/train.arff").getDataSet();
        Instances dev = new DataSource("datuak/dev.arff").getDataSet();

        train.setClassIndex(train.numAttributes() - 1);
        dev.setClassIndex(dev.numAttributes() - 1);

        // 🔁 Probar TODAS las combinaciones
        String[] vectorTypes = {"BoW", "TF", "TF-IDF"};
        String[] attrSelTypes = {"InfoGain", "GainRatio"};

        for (String v : vectorTypes) {
            for (String a : attrSelTypes) {
                testConfiguration(train, dev, v, a);
            }
        }
    }

    public static void testConfiguration(Instances train, Instances dev,
                                         String vectorType, String attrSelType) throws Exception {

        System.out.println("====================================");
        System.out.println("Vectorization: " + vectorType + " | AttributeSelection: " + attrSelType);

        // 🔹 1. StringToWordVector
        StringToWordVector filter = new StringToWordVector();
        filter.setWordsToKeep(5000);
        filter.setLowerCaseTokens(true);

        if (vectorType.equals("BoW")) {
            filter.setOutputWordCounts(false);
            filter.setTFTransform(false);
            filter.setIDFTransform(false);
        } else if (vectorType.equals("TF")) {
            filter.setOutputWordCounts(true);
            filter.setTFTransform(true);
            filter.setIDFTransform(false);
        } else if (vectorType.equals("TF-IDF")) {
            filter.setOutputWordCounts(true);
            filter.setTFTransform(true);
            filter.setIDFTransform(true);
        }

        filter.setInputFormat(train);

        Instances trainFiltered = Filter.useFilter(train, filter);
        Instances devFiltered = Filter.useFilter(dev, filter);

        // 🔹 2. Attribute Selection
        AttributeSelection attrSel = new AttributeSelection();
        Ranker ranker = new Ranker();
        ranker.setNumToSelect(1000);

        if (attrSelType.equals("InfoGain")) {
            attrSel.setEvaluator(new InfoGainAttributeEval());
        } else if (attrSelType.equals("GainRatio")) {
            attrSel.setEvaluator(new GainRatioAttributeEval());
        }

        attrSel.setSearch(ranker);
        attrSel.setInputFormat(trainFiltered);

        trainFiltered = Filter.useFilter(trainFiltered, attrSel);
        devFiltered = Filter.useFilter(devFiltered, attrSel);

        // 🔹 3. Clasificador: SMO (SVM)
        SMO cls = new SMO();
        cls.buildClassifier(trainFiltered);

        // 🔹 4. Evaluación en dev
        Evaluation eval = new Evaluation(trainFiltered);
        eval.evaluateModel(cls, devFiltered);

        // 🔹 5. Resultados en %
        double accuracy = (1 - eval.errorRate()) * 100;
        double f1 = eval.fMeasure(1) * 100;

        System.out.printf("Accuracy: %.2f%%\n", accuracy);
        System.out.printf("F1 (pos): %.2f%%\n", f1);
        System.out.println("Num attributes: " + trainFiltered.numAttributes());
    }
}