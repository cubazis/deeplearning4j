package org.deeplearning4j.mnist;

import org.deeplearning4j.logger.IDrawer;
import org.deeplearning4j.logger.InLogger;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.util.ArrayList;

public class MLPMnistSingleLayerExample {

    public static void main(String[] args) throws Exception {
        //number of rows and columns in the input pictures
        final int numRows = 28;
        final int numColumns = 28;
        int outputNum = 10; // number of output classes
        int batchSize = 128; // batch size for each epoch
        int rngSeed = 123; // random number seed for reproducibility
        int numEpochs = 15; // number of epochs to perform
        ArrayList<Integer> myArray;
        InLogger myLogger = new InLogger();

        //Get the DataSetIterators:
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);


        myLogger.info("Сборка модели....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed) //include a random seed for reproducibility
                // use stochastic gradient descent as an optimization algorithm
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .iterations(1)
                .learningRate(0.006) //specify the learning rate
                .updater(Updater.NESTEROVS).momentum(0.9) //specify the rate of change of the learning rate.
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder() //create the first, input layer with xavier initialization
                        .nIn(numRows * numColumns)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                        .nIn(1000)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //use backpropagation to adjust weights
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(myLogger);

        myLogger.info("Тренировка модели....");

        for( int i=0; i<1; i++ ){
            myLogger.startAt();
            model.fit(mnistTrain);
            myLogger.endAt();
        }
        myArray = myLogger.getArrayList();
        IDrawer.setMyArray(myArray);
        IDrawer.ready();


        myLogger.info("Оценка модели....");
        Evaluation eval = new Evaluation(outputNum); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        myLogger.info(eval.stats());
    }
}
