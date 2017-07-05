package com.onpositive.keras.importer.function;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class TanHActivationFunction implements IAbstractActivationFunction {

	@Override
	public DoubleMatrix calculate(DoubleMatrix X) {
		return MatrixFunctions.tanh(X);
	}

}
