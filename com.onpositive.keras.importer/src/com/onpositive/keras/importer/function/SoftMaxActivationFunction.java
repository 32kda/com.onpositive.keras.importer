package com.onpositive.keras.importer.function;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class SoftMaxActivationFunction implements IAbstractActivationFunction {

	@Override
	public DoubleMatrix calculate(DoubleMatrix X) {
//		X = X.transpose(); //XXX unclear, why
		DoubleMatrix expM = MatrixFunctions.exp(X);
		for (int i = 0; i < X.rows; i++) {
			DoubleMatrix expMi = expM.getRow(i);
			expM.putRow(i, expMi.div(expMi.sum()));
		}
		return expM;
	}

}
