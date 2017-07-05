package com.onpositive.keras.importer.function;

import org.jblas.DoubleMatrix;

public class HardSigmoidActivationFunction implements IAbstractActivationFunction {

	@Override
	public DoubleMatrix calculate(DoubleMatrix X) {
		 double slope = 0.2;
	        double shift = 0.5;
	        X = X.mul(slope).add(shift);
	        DoubleMatrix clippedX = new DoubleMatrix(X.rows, X.columns);
	        for (int i = 0; i < X.length; i++) {
	            if (X.get(i) > 1) {
	                clippedX.put(i, 1);
	            } else if (X.get(i) < 0) {
	                clippedX.put(i, 0);
	            } else {
	                clippedX.put(i, X.get(i));
	            }
	        }
	        return clippedX;
	}

}
