package org.deeplearning4j.examples.feedforward.mnist.Logger;

import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.ArrayList;

/**
 * Created by aydar on 21.03.17.
 */
public class ILogger implements IterationListener {
    private boolean invoked = false;
    private long startTime, endTime;
    private ArrayList<Integer> arrayList = new ArrayList<>();
    private ArrayList<Long> epochTime = new ArrayList<>();
    private int iter = 0, printIterations = 1;

    public ArrayList<Integer> getArrayList(){
        return arrayList;
    }

    @Override
    public boolean invoked() {
        return invoked;
    }

    @Override
    public void invoke() {
        this.invoked = true;
    }

    @Override
    public void iterationDone(Model model, int iteration) {
        invoke();
        if(iter % printIterations == 0) {
            double result = model.score();
            int round = (int)Math.round(result*100);
            arrayList.add(round);
            //System.out.println("Результат " + iter + " итерации: " + result);
        }
        iter++;
    }

    public void info(String s){
        System.out.println(s);
    }
    public void startAt(){
        startTime = System.currentTimeMillis();
    }
    public void endAt(){
        endTime = System.currentTimeMillis();
        epochTime.add(endTime-startTime);
        System.out.println("Time for one epoch: " + (endTime-startTime));
    }
}
