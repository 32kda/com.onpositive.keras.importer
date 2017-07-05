package com.onpositive.keras.importer.layers;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import com.google.gson.Gson;
import com.google.gson.JsonIOException;
import com.google.gson.JsonSyntaxException;
import com.onpositive.keras.importer.Utils;
import com.onpositive.keras.importer.function.HardSigmoidActivationFunction;
import com.onpositive.keras.importer.function.IAbstractActivationFunction;
import com.onpositive.keras.importer.function.ReLUActivationFunction;
import com.onpositive.keras.importer.function.SoftMaxActivationFunction;
import com.onpositive.keras.importer.function.TanHActivationFunction;
import com.onpositive.keras.importer.model.LayerModel;
import com.onpositive.keras.importer.model.Model;

public class ImportingFactory {
	
	public static List<AbstractLayer> loadLayers(File modelFile, File weightsFolder) {
		Gson gson = new Gson();
		Model model;
		try {
			model = gson.fromJson(new BufferedReader(new FileReader(modelFile)), Model.class);
		} catch (JsonSyntaxException | JsonIOException | FileNotFoundException e) {
			e.printStackTrace();
			return null;
		}
		
		List<AbstractLayer> result = new ArrayList<AbstractLayer>();
		try {
			for (int i = 0; i < model.getLayers().size(); i++) {
				LayerModel curLayer = model.getLayers().get(i);
				String name = curLayer.getName();
				String className = curLayer.getClassName();
				if ("dense".equalsIgnoreCase(className)) {
					DoubleMatrix biasMatrix = null;
					if (curLayer.isUseBias()) { 
						biasMatrix = loadDenseBiasMatrix(weightsFolder,name);
					}
					DoubleMatrix kernelMatrix = loadDenseKernelMatrix(weightsFolder,name);
					String activation = curLayer.getActivation();
					IAbstractActivationFunction activationFunction = getActivationFunction(activation);
					result.add(new DenseLayer(kernelMatrix, biasMatrix, activationFunction));
				} else if ("lstm".equalsIgnoreCase(className)) {
					IAbstractActivationFunction activationFunc = getActivationFunction(curLayer.getActivation());
					IAbstractActivationFunction recurrentActivationFunc = getActivationFunction(curLayer.getRecurrentActivation());
					int x = 0;
					int y = 0;
					Integer[] inputShape = curLayer.getBatchInputShape();
					if (inputShape != null) {
						if (inputShape.length > 1 && inputShape[1] != null) {
							y = inputShape[1];
						}
						if (inputShape.length > 2 && inputShape[2] != null) {
							x = inputShape[2];
						}
					}
					if (hasKernel(weightsFolder, name)) {
						String fileName = String.format("%s_kernel.txt", name);
						DoubleMatrix matrix = Utils.loadMatrixFromFile(new File(weightsFolder, fileName));
						int size = matrix.columns / 4;
						DoubleMatrix W_i = matrix.getColumns(new IntervalRange(0,size));
						DoubleMatrix W_f = matrix.getColumns(new IntervalRange(size,size * 2));
						DoubleMatrix W_c = matrix.getColumns(new IntervalRange(size * 2,size * 3));
						DoubleMatrix W_o = matrix.getColumns(new IntervalRange(size * 3,size * 4));
						
						matrix = Utils.loadMatrixFromFile(new File(weightsFolder, String.format("%s_recurrent_kernel.txt", name)));
						size = matrix.columns / 4;
						DoubleMatrix U_i = matrix.getColumns(new IntervalRange(0,size));
						DoubleMatrix U_f = matrix.getColumns(new IntervalRange(size,size * 2));
						DoubleMatrix U_c = matrix.getColumns(new IntervalRange(size * 2,size * 3));
						DoubleMatrix U_o = matrix.getColumns(new IntervalRange(size * 3,size * 4));
						
						matrix = Utils.loadMatrixFromFile(new File(weightsFolder, String.format("%s_bias.txt", name)));
						size = matrix.rows / 4;
						DoubleMatrix b_i = matrix.getRows(new IntervalRange(0,size));
						DoubleMatrix b_f = matrix.getRows(new IntervalRange(size,size * 2));
						DoubleMatrix b_c = matrix.getRows(new IntervalRange(size * 2,size * 3));
						DoubleMatrix b_o = matrix.getRows(new IntervalRange(size * 3,size * 4));
						result.add(new LSTMLayer(i, activationFunc, recurrentActivationFunc, x, y, W_i, U_i, b_i, W_c, U_c, b_c, W_f, U_f, b_f, W_o, U_o, b_o, false));
					} else {
						DoubleMatrix b_c = loadMatrix(weightsFolder,name,"b_c");
						DoubleMatrix b_f = loadMatrix(weightsFolder,name,"b_f");
						DoubleMatrix b_i = loadMatrix(weightsFolder,name,"b_i");
						DoubleMatrix b_o = loadMatrix(weightsFolder,name,"b_o");
	
						DoubleMatrix U_c = loadMatrix(weightsFolder,name,"U_c");
						DoubleMatrix U_f = loadMatrix(weightsFolder,name,"U_f");
						DoubleMatrix U_i = loadMatrix(weightsFolder,name,"U_i");
						DoubleMatrix U_o = loadMatrix(weightsFolder,name,"U_o");
						
						DoubleMatrix W_c = loadMatrix(weightsFolder,name,"W_c");
						DoubleMatrix W_f = loadMatrix(weightsFolder,name,"W_f");
						DoubleMatrix W_i = loadMatrix(weightsFolder,name,"W_i");
						DoubleMatrix W_o = loadMatrix(weightsFolder,name,"W_o");
						result.add(new LSTMLayer(i, activationFunc, recurrentActivationFunc, x, y, W_i, U_i, b_i, W_c, U_c, b_c, W_f, U_f, b_f, W_o, U_o, b_o, false));
					}
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return result;
	}

	private static DoubleMatrix loadDenseKernelMatrix(File weightsFolder, String name) throws IOException {
		File file = new File(weightsFolder, String.format("%s_%s.txt", name, "W"));
		if (file.exists()) {
			return Utils.loadMatrixFromFile(file);
		}
		file = new File(weightsFolder, String.format("%s_%s.txt", name, "kernel"));
		return Utils.loadMatrixFromFile(file);
	}

	private static DoubleMatrix loadDenseBiasMatrix(File weightsFolder, String name) throws IOException {
		File file = new File(weightsFolder, String.format("%s_%s.txt", name, "b"));
		if (file.exists()) {
			return Utils.loadMatrixFromFile(file);
		}
		file = new File(weightsFolder, String.format("%s_%s.txt", name, "bias"));
		return Utils.loadMatrixFromFile(file);
	}

	private static boolean hasKernel(File weightsFolder, String name) {
		String fileName = String.format("%s_kernel.txt", name);
		return new File(weightsFolder, fileName).isFile();
	}

	private static DoubleMatrix loadMatrix(File weightsFolder, String name, String matrixName) throws IOException {
		String fileName = String.format("%s_%s.txt", name, matrixName);
		return Utils.loadMatrixFromFile(new File(weightsFolder, fileName));
	}

//	private static String getDenseKernelFileName(int i, String name) {
//		return i + "_" + name + "_kernel.txt";
//	}
//
//	private static String getDenseBiasFileName(int i, String name) {
//		return i + "_" + name + "_bias.txt";
//	}

	private static IAbstractActivationFunction getActivationFunction(String activation) {
		if ("hard_sigmod".equals(activation) || "sigmoid".equals(activation))
			return new HardSigmoidActivationFunction();
		if ("tanh".equals(activation))
			return new TanHActivationFunction();
		if ("relu".equals(activation)) {
			return new ReLUActivationFunction();
		}
		return new SoftMaxActivationFunction(); //TODO implement more options here
	}

}
