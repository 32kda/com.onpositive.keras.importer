package com.onpositive.keras.importer.model;

import com.google.gson.annotations.SerializedName;

public class Config {
	
	private String name;
	private boolean trainable;
	private int units;
	private String activation;
	@SerializedName(value="recurrent_activation")
	private String recurrentActivation;
	@SerializedName(value="use_bias")
	private boolean useBias;
	@SerializedName(value="batch_input_shape")
	private Integer[] batchInputShape;
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public boolean isTrainable() {
		return trainable;
	}
	public void setTrainable(boolean trainable) {
		this.trainable = trainable;
	}
	public int getUnits() {
		return units;
	}
	public void setUnits(int units) {
		this.units = units;
	}
	public String getActivation() {
		return activation;
	}
	public void setActivation(String activation) {
		this.activation = activation;
	}
	public boolean isUseBias() {
		return useBias;
	}
	public void setUseBias(boolean useBias) {
		this.useBias = useBias;
	}
	@Override
	public String toString() {
		return "Config [name=" + name + ", units=" + units + "]";
	}
	public String getRecurrentActivation() {
		return recurrentActivation;
	}
	public void setRecurrentActivation(String recurrentActivation) {
		this.recurrentActivation = recurrentActivation;
	}
	public Integer[] getBatchInputShape() {
		return batchInputShape;
	}
	public void setBatchInputShape(Integer[] batchInputShape) {
		this.batchInputShape = batchInputShape;
	}

}
