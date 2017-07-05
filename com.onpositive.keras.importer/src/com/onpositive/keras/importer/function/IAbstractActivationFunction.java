package com.onpositive.keras.importer.function;

import org.jblas.DoubleMatrix;

public interface IAbstractActivationFunction {

	DoubleMatrix calculate(DoubleMatrix X);
	
}
