package com.onpositive.keras.importer.function;

import org.jblas.DoubleMatrix;

public class ReLUActivationFunction implements IAbstractActivationFunction {

	@Override
	public DoubleMatrix calculate(DoubleMatrix x) {
		 for (int i = 0; i < x.length; i++) {
			x.put(i, Math.max(x.get(i), 0));
		 }
		 return x;
	}

}
