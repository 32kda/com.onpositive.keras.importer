package com.onpositive.keras.importer;
import org.jblas.DoubleMatrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alex on 23.06.2016.
 */
public class Utils {

    public static DoubleMatrix loadMatrixFromFile(String filePath) throws IOException {
        FileInputStream fstream = new FileInputStream(filePath);
        return loadMatrixFromStream(fstream);
    }
    
    public static DoubleMatrix loadMatrixFromFile(File file) throws IOException {
        FileInputStream fstream = new FileInputStream(file);
        return loadMatrixFromStream(fstream);
    }

	private static DoubleMatrix loadMatrixFromStream(InputStream fstream) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
        ArrayList<ArrayList<Double>> loadedArrays = new ArrayList<ArrayList<Double>>();

        String strLine;
        while ((strLine = br.readLine()) != null) {
            ArrayList<Double> row = new ArrayList<Double>();
            String[] a = strLine.split(" ");
            for (String s : a) {
                row.add(Double.parseDouble(s));
            }
            loadedArrays.add(row);
        }
        br.close();

        int columns = loadedArrays.get(0).size();
        int rows = loadedArrays.size();
        double[][] target = new double[rows][columns];

        for (int i = 0; i < loadedArrays.size(); i++) {
            for (int j = 0; j < target[i].length; j++) {
                target[i][j] = (Double) loadedArrays.get(i).get(j);
            }
        }
        return new DoubleMatrix(target);
	}
	
	public static List<double[]> loadTestDataFromCSVStream(InputStream fstream) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(fstream));
        ArrayList<double[]> loadedArrays = new ArrayList<>();

        String strLine;
        while ((strLine = br.readLine()) != null) {
            String[] a = strLine.split(",");
            double[] row = new double[a.length];
            for (int i = 0; i < a.length; i++) {
                row[i] = Double.parseDouble(a[i].trim());
            }
            loadedArrays.add(row);
        }
        br.close();
        
        return loadedArrays;

//        int columns = loadedArrays.get(0).size();
//        double[] target = new double[columns];
//        List<DoubleMatrix> result = new ArrayList<>();
//        
//        for (int i = 0; i < loadedArrays.size(); i++) {
//            for (int j = 0; j < target.length; j++) {
//                target[j] = (Double) loadedArrays.get(i).get(j);
//                result.add(new DoubleMatrix(target));
//            }
//        }
//        return result;
	}


}
