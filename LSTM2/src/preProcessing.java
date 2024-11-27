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
    
}
