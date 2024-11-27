import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class input{

    private ArrayList<double[]> data;


    input(String file){

        data = new ArrayList<double[]>();

        readfile(file,data);

    }

    private static void readfile(String loc, ArrayList<double[]> storage){
        String line ="";
        try{
            BufferedReader br = new BufferedReader(new FileReader(loc));
            while((line =br.readLine())!=null){
                if(line.isEmpty()!=true){
                    String[] data = line.split(",");

                    //Data Points
                    double open = Double.parseDouble(data[0]);
                    double high = Double.parseDouble(data[1]);
                    double low = Double.parseDouble(data[2]);
                    double close = Double.parseDouble(data[3]);

                    double[] info = {open, high, low, close};

                    storage.add(info);

                }
            }

            br.close();

        }

        catch(IOException e){

        }
    }
}
