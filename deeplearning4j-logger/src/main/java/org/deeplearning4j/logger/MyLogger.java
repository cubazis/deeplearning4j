package org.deeplearning4j.logger;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.stage.Stage;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.optimize.api.IterationListener;

import java.util.ArrayList;

/**
 * Created by aydar on 21.03.17.
 */
public class MyLogger extends Application implements IterationListener {
    private boolean invoked = false;
    ArrayList<Double> arrayList = new ArrayList<Double>();
    private int iter = 0, row = 250, column;
    AnimationTimer loop;

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
        double result = model.score();
        arrayList.add(result);
        //System.out.println("Результат " + iter + " итерации: " + result);
        iter++;
    }

    public void info(String s){
        System.out.println(s);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        primaryStage.setTitle( "Canvas Example" );
        Group root = new Group();
        Scene theScene = new Scene(root);
        primaryStage.setScene(theScene);
        column = arrayList.size();
        Canvas canvas = new Canvas(column, row);
        root.getChildren().add(canvas);
        final GraphicsContext gc = canvas.getGraphicsContext2D();

        loop = new AnimationTimer() {
            @Override
            public void handle(long now) {
                for (int j = 0; j < column; j++) {
                    gc.strokeOval(j, arrayList.get(j), 1, 0);                 }
                }
        };
        loop.start();
        primaryStage.show();
    }

    public void ready(){
        launch();
    }
}
