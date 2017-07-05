package com.onpositive.keras.importer.layers;

import org.jblas.DoubleMatrix;

import com.onpositive.keras.importer.function.IAbstractActivationFunction;

public class DenseLayer implements AbstractLayer {
	
	private DoubleMatrix kernel;
	private DoubleMatrix bias;
	private IAbstractActivationFunction activationFunction;
	private int realSize;

	public DenseLayer(DoubleMatrix kernel, DoubleMatrix bias, IAbstractActivationFunction activationFunction) {
		super();
		this.kernel = kernel;
		this.bias = bias;
		this.activationFunction = activationFunction;
	}

	@Override
	public DoubleMatrix forwardStep(DoubleMatrix X) {
		return activationFunction.calculate(this.kernel.transpose().mmul(X).addColumnVector(this.bias));
	}

	@Override
	public void setRealSize(int realSize) {
		this.realSize = realSize;
	}

}
