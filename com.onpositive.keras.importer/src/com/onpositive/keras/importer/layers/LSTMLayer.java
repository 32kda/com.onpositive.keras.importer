package com.onpositive.keras.importer.layers;
import org.jblas.DoubleMatrix;

import com.onpositive.keras.importer.function.IAbstractActivationFunction;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by Alex on 23.06.2016.
 * Basic LSTM layer with forward propagation routine
 */
public class LSTMLayer implements AbstractLayer {

    private DoubleMatrix W_i;
    private DoubleMatrix U_i;
    private DoubleMatrix b_i;
    private DoubleMatrix W_c;
    private DoubleMatrix U_c;
    private DoubleMatrix b_c;
    private DoubleMatrix W_f;
    private DoubleMatrix U_f;
    private DoubleMatrix b_f;
    private DoubleMatrix W_o;
    private DoubleMatrix U_o;
    private DoubleMatrix b_o;

    private int realSize;
    private int layerNum;
    private boolean returnSequence;
	private IAbstractActivationFunction activationFunction;
	private IAbstractActivationFunction recurrentActivationFunction;
	private int sizeX;
	private int sizeY;

 
    public LSTMLayer(int layerNum, IAbstractActivationFunction activationFunction, IAbstractActivationFunction recurrentActivationFunction, int sizeX, int sizeY, DoubleMatrix w_i, DoubleMatrix u_i, DoubleMatrix b_i, DoubleMatrix w_c, DoubleMatrix u_c,
			DoubleMatrix b_c, DoubleMatrix w_f, DoubleMatrix u_f, DoubleMatrix b_f, DoubleMatrix w_o, DoubleMatrix u_o,
			DoubleMatrix b_o, boolean returnSequence) {
		super();
		this.activationFunction = activationFunction;
		this.recurrentActivationFunction = recurrentActivationFunction;
		this.sizeX = sizeX;
		this.sizeY = sizeY;
		W_i = w_i;
		U_i = u_i;
		this.b_i = b_i;
		W_c = w_c;
		U_c = u_c;
		this.b_c = b_c;
		W_f = w_f;
		U_f = u_f;
		this.b_f = b_f;
		W_o = w_o;
		U_o = u_o;
		this.b_o = b_o;
		this.layerNum = layerNum;
		this.returnSequence = returnSequence;
	}

	public void setRealSize(int realSize) {
        this.realSize = realSize;
    }

    public DoubleMatrix forwardStep(DoubleMatrix X) {

    	if (sizeX > 1 && sizeY > 1) {
    		realSize = sizeX;
    		X = new DoubleMatrix(sizeX,sizeY,X.data);
    	}
        if (this.layerNum == 0) {
            X = inputFix(X);
        }
        ArrayList<DoubleMatrix> outputs = new ArrayList<DoubleMatrix>();

        // Let's define previous cell output and hidden state
        DoubleMatrix h_t_1 = DoubleMatrix.zeros(this.W_i.columns, 1);
        DoubleMatrix C_t_1 = DoubleMatrix.zeros(this.W_i.columns, 1);

        for (int i = 0; i < X.columns; i++) {

            // Weights update for every cell step-by-step.
            // For more details check out: http://deeplearning.net/tutorial/lstm.html
            DoubleMatrix x_t = X.getColumn(i);
            DoubleMatrix W_i_mul_x = this.W_i.transpose().mmul(x_t);
            DoubleMatrix U_i_mul_h_1 = this.U_i.transpose().mmul(h_t_1);
            DoubleMatrix i_t = recurrentActivationFunction.calculate(W_i_mul_x.addColumnVector(U_i_mul_h_1).addColumnVector(this.b_i));

            DoubleMatrix W_c_mul_x = this.W_c.transpose().mmul(x_t);
            DoubleMatrix U_c_mul_h_1 = this.U_c.transpose().mmul(h_t_1);
            DoubleMatrix C_tilda = activationFunction.calculate(W_c_mul_x.addColumnVector(U_c_mul_h_1).addColumnVector(this.b_c));

            DoubleMatrix W_f_mul_x = this.W_f.transpose().mmul(x_t);
            DoubleMatrix U_f_mul_h_1 = this.U_f.transpose().mmul(h_t_1);
            DoubleMatrix f_i = recurrentActivationFunction.calculate(W_f_mul_x.addColumnVector(U_f_mul_h_1).addColumnVector(this.b_f));

            DoubleMatrix C_t = (i_t.mul(C_tilda)).add(f_i.mul(C_t_1));

            DoubleMatrix W_o_mul_x = this.W_o.transpose().mmul(x_t);
            DoubleMatrix U_o_mul_h_1 = this.U_o.transpose().mmul(h_t_1);

            DoubleMatrix o_t = recurrentActivationFunction.calculate(W_o_mul_x.addColumnVector(U_o_mul_h_1).addColumnVector(this.b_o));
            DoubleMatrix h_t = o_t.mul(activationFunction.calculate(C_t));

            outputs.add(h_t);
            h_t_1 = h_t;
            C_t_1 = C_t;

        }

        if (this.returnSequence) {

            // We return out sequence corresponding to our input,
            // which has length of this.realSize.
            // We will restore it in next layer again using fixInput()
            int rows = outputs.get(0).rows;
            DoubleMatrix result = DoubleMatrix.zeros(rows, this.realSize);
            for (int i = 0; i < outputs.size(); i++) {
                for (int j = 0; j < this.realSize; j++) {
                    result.put(i, j, outputs.get(j).get(i));
                }
            }
            return result;

        } else {
            // If we don't want to return sequence of outputs from every cell,
            // but only for the last one (for the last LSTM layer), use this.
            return outputs.get(outputs.size() - 1);
        }

    }

	private DoubleMatrix fixSize(DoubleMatrix x) {
		return new DoubleMatrix(W_i.rows, W_i.columns, x.data);
	}

	public DoubleMatrix inputFix(DoubleMatrix X) {
        DoubleMatrix res = DoubleMatrix.zeros(W_i.rows, W_i.columns);
        for (int i = 0; i < X.rows; i++) {
            res.putColumn(i, X.getRow(i));
        }
        return res;
    }


}
