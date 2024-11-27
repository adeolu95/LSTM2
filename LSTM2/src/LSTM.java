import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class LSTM {
    private int inputSize;
    private int hiddenSize; 
    private int outputSize;

    // LSTM weights and biases
    private double[][] Wf, Wi, Wc, Wo, Wy; // Weight matrices for forget, input, cell, and output gates
    private double[] bf, bi, bc, bo;   // Biases for each gate
    private double by;

    // Hidden and cell states
    private double[] h, c;

    private double learningRate = 0.001;

    public LSTM(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;
        

        //Initialization of Weights and biases
        double scale = 0.1;

        Wf = randomMatrix(hiddenSize, inputSize+hiddenSize,scale);
        Wi = randomMatrix(hiddenSize, inputSize+hiddenSize,scale);
        Wc = randomMatrix(hiddenSize, inputSize+hiddenSize,scale);
        Wo = randomMatrix(hiddenSize, inputSize+hiddenSize,scale);
        Wy = randomMatrix(hiddenSize, inputSize+hiddenSize,scale);

        bf = randomVector(hiddenSize, scale);
        bi = randomVector(hiddenSize, scale);
        bc = randomVector(hiddenSize, scale);
        bo = randomVector(hiddenSize, scale);
        

    }

    
    // Sigmoid activation function
    private double sigmoid(double x) {
        double denominator = (1 + Math.exp(-x));
        if(denominator==0)denominator=0.0001;
        return 1 /denominator ;
    }

    private double diffSig(double x){
        return(sigmoid(x)*(1-sigmoid(x)));
    }

    // Tanh activation function
    private double tanh(double x) {
        return Math.tanh(x);
    }

    private double difftanh(double x){
        return (1-(Math.pow(tanh(x),2)));
    }

    // Utility methods for matrix-vector multiplication and vector operations
    private double[] matVecMul(double[][] matrix, double[] vector, double[] bias) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
            result[i] += bias[i];
        }
        return result;
    }
    private double[] vectorAdd(double[] v1, double[] v2) {
        double[] result = new double[v1.length];
        for (int i = 0; i < v1.length; i++) {
            result[i] = v1[i] + v2[i];
        }
        return result;
    }

    private double[] vectorMultiply(double[] v1, double[] v2) {
        double[] result = new double[v1.length];
        for (int i = 0; i < v1.length; i++) {
            result[i] = v1[i] * v2[i];
        }
        return result;
    }

    private double[] sigmoidVector(double[] vector) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = sigmoid(vector[i]);
        }
        return result;
    }

    private double[] tanhVector(double[] vector) {
        double[] result = new double[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = tanh(vector[i]);
        }
        return result;
    }

    private double[] concatArrays(double[] a, double[] b) {
        double[] result = new double[a.length + b.length];
        System.arraycopy(a, 0, result, 0, a.length);
        System.arraycopy(b, 0, result, a.length, b.length);
        return result;
    }


    private double[][] randomMatrix(int rows, int cols, double scale) {
        double[][] matrix = new double[rows][cols];
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // Initialize values randomly within the range [-scale, scale]
                matrix[i][j] = (random.nextDouble() * 2 - 1) * scale;
            }
        }
        return matrix;
    }


    private double[] randomVector(int size, double scale) {
            double[] vector = new double[size];
            Random random = new Random();
            for (int i = 0; i < size; i++) {
                // Initialize values randomly within the range [-scale, scale]
                vector[i] = (random.nextDouble() * 2 - 1) * scale;
            }
            return vector;
        }


    private double clip(double value, double min, double max) {
                return Math.max(min, Math.min(max, value));  // Clip gradients between min and max
            }

    




    
}
