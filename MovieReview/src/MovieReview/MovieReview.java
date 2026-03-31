package MovieReview;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.stemmers.LovinsStemmer;
import weka.core.stopwords.Rainbow;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.SerializationHelper;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.*;

public class MovieReview {
	
	public static void main(String[] args) throws Exception {
		
		// ====================================================
		// 1. datuak kargatu eta beren klasea definitu, defektuz azken instantzia izango dena
		// ====================================================
		// Train
		DataSource trainSource = new DataSource("datuak/train.arff");
		Instances train = trainSource.getDataSet();
		train.setClassIndex(train.numAttributes() - 1);
		
		// Dev
		DataSource devSource = new DataSource("datuak/dev.arff");
		Instances dev = devSource.getDataSet();
		dev.setClassIndex(dev.numAttributes() - 1);
		
		System.out.println("Hasierako atributu kopurua: " + train.numAttributes());
				
		// ====================================================
		// 2. Bektorizazioa egin (BoW)
		// ====================================================
		StringToWordVector filter = new StringToWordVector();
		
		// Clean prozesua
		filter.setTokenizer(new AlphabeticTokenizer());	// Hitzak bakarrik hartzen ditu (puntuazioak eta zenbakiak kanpora)
		filter.setStemmer(new LovinsStemmer());		// Erroak ateratzeko (movies -> movi)
		filter.setStopwordsHandler(new Rainbow()); 	// "the", "a", "of" bezalako hitzak kentzeko
		filter.setLowerCaseTokens(true);	// Letra xehean ipini
		
		// BoW
		filter.setWordsToKeep(5000);	// gure hiztegi maximoa 5000 hitz izango ditu balio bakoitzeko
		filter.setTFTransform(false);	// TF ez aktibatu
		filter.setIDFTransform(false);	// IDF ez aktibatu
		filter.setInputFormat(train);	// Filtroa entrenatu
		
		// Datu sortak bektorizatu (String -> int[])
		Instances trainBek = Filter.useFilter(train, filter);
		Instances devBek = Filter.useFilter(dev, filter);
		
		// Bektorizatzailea gorde, gero test-ean erabiltzeko
		SerializationHelper.write("vectorizer.model", filter);
		
		// Konprobatu ondo bektorizatu den
		System.out.println("Bektorizazioa bukatuta");
		System.out.println("Bektorizazioa eta gero geratu diren atributuak:" + trainBek.numAttributes());
		
		// ====================================================
		// 3. AttibuteSelection metodoa egin (InfoGain)
		// ====================================================
		AttributeSelection attr = new AttributeSelection();
		InfoGainAttributeEval eval = new InfoGainAttributeEval();
		
		Ranker bilatu = new Ranker();
		bilatu.setNumToSelect(1000);
		
		attr.setEvaluator(eval);
		attr.setSearch(bilatu);
		
		attr.setInputFormat(trainBek); // selektorea entrenatu
		
		// Selekzioa aplikatu bektorizatutako fitxategiei
		Instances trainSel = Filter.useFilter(trainBek, attr);
		Instances devSel = Filter.useFilter(devBek, attr);
		
		// Gorde eta inprimatu konprobatzeko ondo dagoen
		SerializationHelper.write("attrsel.model", attr);
		System.out.println("Atributu kopurua selekzioa eta gero:" + trainSel.numAttributes());
		
		ParametroEkorketa.ekorketaExekutatu(trainSel,devSel);
		
		KalitateEstimatua.main(args);
		Iragarpenak.main(args);
	}

}
