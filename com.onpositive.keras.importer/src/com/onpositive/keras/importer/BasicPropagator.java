package com.onpositive.keras.importer;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.jblas.DoubleMatrix;

import com.onpositive.keras.importer.layers.AbstractLayer;
import com.onpositive.keras.importer.layers.ImportingFactory;

public class BasicPropagator {
	
	private List<AbstractLayer> layers = new ArrayList<AbstractLayer>();
	
	public BasicPropagator(File modelFile, File weightsFolder) {
		layers = ImportingFactory.loadLayers(modelFile, weightsFolder);
	}
	
	public DoubleMatrix forwardPropagate(DoubleMatrix X) {
		 	DoubleMatrix intermediateResult = X;
	        for (AbstractLayer layer: layers) {
	        	layer.setRealSize(X.rows);
	            intermediateResult = layer.forwardStep(intermediateResult);
	        }
	        return intermediateResult;
	}

}
