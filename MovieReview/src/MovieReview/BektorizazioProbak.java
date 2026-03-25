package MovieReview;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.*;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.Evaluation;

public class BektorizazioProbak {
    public static void main(String[] args) throws Exception {

        // =====================
        // 1. Datuak kargatu
        // =====================
        DataSource trainSource = new DataSource("datuak/train.arff");
        Instances train = trainSource.getDataSet();
        train.setClassIndex(train.numAttributes() - 1);

        DataSource devSource = new DataSource("datuak/dev.arff");
        Instances dev = devSource.getDataSet();
        dev.setClassIndex(dev.numAttributes() - 1);

        // =====================
        // 2. Aukerak
        // =====================
        boolean[] tfOptions = {false, true};
        boolean[] idfOptions = {false, true};

        ASEvaluation[] evaluators = {
            new InfoGainAttributeEval(),
            new GainRatioAttributeEval()
        };

        String[] evalNames = {
            "InfoGain",
            "GainRatio"
        };

        // =====================
        // 3. Konbinazio guztiak
        // =====================
        for (boolean tf : tfOptions) {
            for (boolean idf : idfOptions) {

                if (!tf && idf) continue;

                System.out.println("\n==============================");
                System.out.println("Vectorization: TF=" + tf + " IDF=" + idf);

                // =====================
                // 4. Bektorizazioa
                // =====================
                StringToWordVector filter = new StringToWordVector();
                filter.setTokenizer(new AlphabeticTokenizer());
                filter.setStemmer(new LovinsStemmer());
                filter.setStopwordsHandler(new Rainbow());

                filter.setTFTransform(tf);
                filter.setIDFTransform(idf);
                filter.setLowerCaseTokens(true);
                filter.setWordsToKeep(5000);

                filter.setInputFormat(train);

                Instances trainBek = Filter.useFilter(train, filter);
                Instances devBek = Filter.useFilter(dev, filter);

                // =====================
                // 5. Atributuen selekzioa
                // =====================
                for (int i = 0; i < evaluators.length; i++) {

                    AttributeSelection attr = new AttributeSelection();
                    Ranker ranker = new Ranker();
                    ranker.setNumToSelect(1000);

                    attr.setEvaluator(evaluators[i]);
                    attr.setSearch(ranker);
                    attr.setInputFormat(trainBek);

                    Instances trainSel = Filter.useFilter(trainBek, attr);
                    Instances devSel = Filter.useFilter(devBek, attr);

                    // =====================
                    // 6. Modeloa + ebaluazioa
                    // =====================
                    NaiveBayes nb = new NaiveBayes();
                    nb.buildClassifier(trainSel);

                    Evaluation eval = new Evaluation(trainSel);
                    eval.evaluateModel(nb, devSel);

                    // =====================
                    // 7. Emaitzak
                    // =====================
                    System.out.println("\n--- " + evalNames[i] + " ---");
                    System.out.println("Attributes: " + trainSel.numAttributes());
                    System.out.println("Accuracy: " + eval.pctCorrect());
                    System.out.println(eval.toSummaryString());
                }
                
            }
        }
    }
}