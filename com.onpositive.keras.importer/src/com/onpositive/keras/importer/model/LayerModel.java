package com.onpositive.keras.importer.model;

import com.google.gson.annotations.SerializedName;

public class LayerModel {
	
	private Config config;
	@SerializedName(value = "class_name")
	private String className;

	public String getClassName() {
		return className;
	}

	public void setClassName(String className) {
		this.className = className;
	}

	public Config getConfig() {
		return config;
	}

	public void setConfig(Config config) {
		this.config = config;
	}

	public String getName() {
		return config.getName();
	}

	public boolean isTrainable() {
		return config.isTrainable();
	}

	public int getUnits() {
		return config.getUnits();
	}

	public String getActivation() {
		return config.getActivation();
	}

	public boolean isUseBias() {
		return config.isUseBias();
	}

	public String getRecurrentActivation() {
		return config.getRecurrentActivation();
	}

	@Override
	public String toString() {
		return "Layer [" + className + "]";
	}

	public Integer[] getBatchInputShape() {
		return config.getBatchInputShape();
	}

	public void setBatchInputShape(Integer[] batchInputShape) {
		config.setBatchInputShape(batchInputShape);
	}
}
