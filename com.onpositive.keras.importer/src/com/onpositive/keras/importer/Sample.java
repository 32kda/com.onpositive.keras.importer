package com.onpositive.keras.importer;

import java.util.Arrays;

public class Sample {
	
	public final double[] inputs;
	public final int answer;
	
	public Sample(double[] inputs, int answer) {
		super();
		this.inputs = inputs;
		this.answer = answer;
	}
	
	public static Sample fromString(String str) {
		int idx = str.indexOf(';');
		if (idx > 0) {
			int answer =  Integer.valueOf(str.substring(0, idx).trim());
			str = str.substring(idx + 1, str.length());
			String[] splitted = str.split(",");
			double[] inputs = Arrays.asList(splitted).stream().mapToDouble(item -> Double.parseDouble(item.trim())).toArray();
			return new Sample(inputs, answer);
		}
		return null;
	}

}
