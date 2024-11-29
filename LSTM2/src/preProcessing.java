import java.util.ArrayList;

public class preProcessing {

    private int numberFeatures; 


    preProcessing(ArrayList<double[]> input, int timeSteps){

        if(input.size()>10){

            this.numberFeatures = input.get(0).length;

            for (int i =0; i<input.size(); i++){
                


                
            }



        }

        else{
            System.out.println("Check input ArrayList to the preProcessing Class, not up to Spec");
        }

    }

    private double[][] normalizeData(ArrayList<double[]> input){
        int rows = input.size();
        int cols = input.get(0).length;

        double[] min = new double[cols];
        double[] max = new double[cols];

        // initialize min and max arrays
        for(int j= 0; j<cols; j++){
            min[j] = Double.MAX_VALUE;
            max[j] = Double.MIN_VALUE;
        }

        //compute min and max for each feature
        for(int i=0; i< rows; i++){
            for(int j =0; j<cols; j++){
                if(input.get(i)[j]< min[j]) min[j] = input.get(i)[j];
                if(input.get(i)[j]> max[j]) max[j] = input.get(i)[j];

            }
        }

        double[][] normalizedData = new double[rows][cols];
        for(int i =0; i< rows; i++){
            for(int j=0; j<cols; j++){
                normalizedData[i][j]= (input.get(i)[j]-min[j])/(max[j]-min[j]);
            }
        }

        return normalizedData;
    }

    private double[] denormalize(double[] normalized, double min, double max){
        double [] original = new double[normalized.length];
        for (int i =0; i< normalized.length; i++){
            original[i] = normalized[i]*(max-min)+min;
        }
        return original;
    }



    
}
