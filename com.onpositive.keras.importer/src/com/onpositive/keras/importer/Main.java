package com.onpositive.keras.importer;
/**
 * Created by Alex on 09.06.2016.
 */

import org.apache.commons.io.FileUtils;
import org.jblas.DoubleMatrix;

import com.google.gson.Gson;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {

    public static void main(String[] args) throws IOException {

    	loadAndTestNetwork2("learnSet2981");
//    	compare();
    	
    }

	private static void loadAndTestNetwork() throws IOException, FileNotFoundException {
		Gson gson = new Gson();
//    	Model mdl = gson.fromJson(new BufferedReader(new FileReader("model.json")), Model.class);
//    	List<AbstractLayer> layers = ImportingFactory.loadLayers(mdl,new File("weights"));
    	List<double[]> testData = Utils.loadTestDataFromCSVStream(new FileInputStream("pima-indians-diabetes.csv"));
    	BasicPropagator basicPropagator = new BasicPropagator(new File("model.json"),new File("weights"));
    	double missed = 0;
    	double[] results = new double[testData.size()];
    	for (int i = 0; i < testData.size(); i++) {
    		double[] input = testData.get(i);
    		DoubleMatrix result = basicPropagator.forwardPropagate(new DoubleMatrix(Arrays.copyOfRange(input,0,input.length - 1)));
    		double desired = input[input.length - 1];
    		double actual = Math.round(result.get(0));
    		missed += Math.abs(actual - desired);
    		results[i] = result.get(0);
    	}
    	saveJavaRes(results);
//    	for (double[] input : testData) {
//    		DoubleMatrix result = basicPropagator.forwardPropagate(new DoubleMatrix(Arrays.copyOfRange(input,0,input.length - 1)));
//    		double desired = input[input.length - 1];
//    		double actual = Math.round(result.get(0));
//    		missed += Math.abs(actual - desired);
//		}
    	System.out.println(missed / testData.size());
//        double[][] x = new double[][]{{0.0,3.0,0.0,0.0,0.4849019607843267,0.14588235294117696,0.0147,0.0,0.5,0.52,2.0},
//                {1.0,3.0,0.0,1.0,0.34590000000000004,0.0855,0.0147,0.0,0.0,0.467,0.0},
//                {1.0,3.0,0.0,1.0,0.34590000000000004,0.0855,0.0147,0.0,0.0,0.467,0.0}};
//
//        DoubleMatrix X = new DoubleMatrix(x);
//
//        SimpleLSTMPropagator propagator = new SimpleLSTMPropagator("C:\\Users\\Alex\\IdeaProjects\\LSTM\\src\\situation0\\", 2);
//        DoubleMatrix prediction = propagator.forward_propagate_full(X);
//        System.out.println(prediction);
	}
	
	private static void loadAndTestNetwork2(String sampleName) throws IOException, FileNotFoundException {
		BasicPropagator basicPropagator = new BasicPropagator(new File("model2.json"),new File("weights"));
		try (Stream<String> stream = Files.lines(Paths.get(sampleName))) {
			List<Sample> samples = stream.map(str -> Sample.fromString(str)).collect(Collectors.toList());
			int okCount = 0;
			for (Sample sample : samples) {
				if (testOK(basicPropagator, sample)) {
					okCount++;
				}
			}
			System.out.println("Was OK:" + okCount + " from " + samples.size() );
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static boolean testOK(BasicPropagator basicPropagator, Sample sample) {
		DoubleMatrix result = basicPropagator.forwardPropagate(new DoubleMatrix(sample.inputs));
		int maxIdx = result.argmax();
		return maxIdx == sample.answer;
	}

	private static void saveJavaRes(double[] results) {
		try (PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("java_result.txt")))) {
	    	for (double res : results) {
				pw.println(String.format("%.18e", res));
			}
    	} catch (Exception e) {
    		e.printStackTrace();
		}
	}
    
    static void compare() {
    	try {
			List<String> javaResults = FileUtils.readLines(new File("java_result.txt"));
			List<String> pythonResults = FileUtils.readLines(new File("python_result.txt"));
			int n = Math.min(javaResults.size(), pythonResults.size());
			int misses = 0;
			double totalDiff = 0;
			for (int i = 0; i < n; i++) {
				double x1 = Double.parseDouble(javaResults.get(i));
				double x2 = Double.parseDouble(pythonResults.get(i));
				totalDiff += Math.abs(x1 - x2);
				misses += Math.abs(Math.round(x1) - Math.round(x2));
			}
			System.out.println(misses);
			System.out.println("Middle diff " + String.format("%.5f", totalDiff / n));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
}
