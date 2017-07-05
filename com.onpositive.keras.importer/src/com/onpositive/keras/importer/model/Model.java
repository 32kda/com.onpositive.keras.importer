package com.onpositive.keras.importer.model;

import java.util.ArrayList;
import java.util.List;

import com.google.gson.annotations.SerializedName;

public class Model {
	
	@SerializedName(value = "class_name")
	private String className;
	@SerializedName(value = "config")
	private List<LayerModel> layers = new ArrayList<>();

	public String getClassName() {
		return className;
	}

	public void setClassName(String className) {
		this.className = className;
	}

	public List<LayerModel> getLayers() {
		return layers;
	}

	public void setLayers(List<LayerModel> layers) {
		this.layers = layers;
	}

	@Override
	public String toString() {
		return className + ", layers=[" + layers + "]";
	}
	
}
