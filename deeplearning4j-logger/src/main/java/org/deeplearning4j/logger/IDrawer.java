package org.deeplearning4j.logger;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Color;
import javafx.stage.Stage;

import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by aydar on 21.03.17.
 */
public class IDrawer extends Application {
    AnimationTimer loop;
    static int row = 250, column;
    static ArrayList<Integer> myArray;
    public static void setMyArray(ArrayList<Integer> arr){
        myArray = arr;
        column = myArray.size();
    }
    public static void ready(){
        launch();
    }
    @Override
    public void start(Stage primaryStage) throws Exception {
        primaryStage.setTitle( "Canvas Example" );
        Group root = new Group();
        Scene theScene = new Scene(root);
        primaryStage.setScene(theScene);
        Canvas canvas = new Canvas(column, row);
        root.getChildren().add(canvas);
        final GraphicsContext gc = canvas.getGraphicsContext2D();

        loop = new AnimationTimer() {
            @Override
            public void handle(long now) {
                for (int j = 0; j < myArray.size(); j++) {
                    gc.strokeOval(j, myArray.get(j), 1, 0);
                }
            }
        };
        loop.start();
        primaryStage.show();
    }
}
