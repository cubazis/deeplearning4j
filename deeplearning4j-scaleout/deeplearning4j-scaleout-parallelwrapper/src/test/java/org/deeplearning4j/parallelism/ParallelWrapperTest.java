package org.deeplearning4j.parallelism;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * Created by agibsonccc on 11/12/16.
 */
public class ParallelWrapperTest {
    private static final Logger log = LoggerFactory.getLogger(ParallelWrapperTest.class);

    @Test
    public void testParallelWrapperRun() throws Exception {

        int nChannels = 1;
        int outputNum = 10;

        // for GPU you usually want to have higher batchSize
        int batchSize = 128;
        int nEpochs = 10;
        int iterations = 1;
        int seed = 123;

        log.info("Load data....");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, 12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, 12345);

        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().seed(seed).iterations(iterations)
                        .regularization(true).l2(0.0005).learningRate(0.01)//.biasLearningRate(0.02)
                        //.learningRateDecayPolicy(LearningRatePolicy.Inverse).lrPolicyDecayRate(0.001).lrPolicyPower(0.75)
                        .weightInit(WeightInit.XAVIER)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)
                        .momentum(0.9).list()
                        .layer(0, new ConvolutionLayer.Builder(5, 5)
                                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                                        .nIn(nChannels).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                        .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build())
                        .layer(2, new ConvolutionLayer.Builder(5, 5)
                                        //Note that nIn needed be specified in later layers
                                        .stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                        .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2)
                                        .stride(2, 2).build())
                        .layer(4, new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
                        .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                        .nOut(outputNum).activation(Activation.SOFTMAX).build())
                        .backprop(true).pretrain(false);
        // The builder needs the dimensions of the image along with the number of channels. these are 28x28 images in one channel
        new ConvolutionLayerSetup(builder, 28, 28, 1);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // ParallelWrapper will take care of load balancing between GPUs.
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                        // DataSets prefetching options. Set this value with respect to number of actual devices
                        .prefetchBuffer(24)

                        // set number of workers equal or higher then number of available devices. x1-x2 are good values to start with
                        .workers(2)

                        // rare averaging improves performance, but might reduce model accuracy
                        .averagingFrequency(3)

                        // if set to TRUE, on every averaging model score will be reported
                        .reportScoreAfterAveraging(true)

                        // optinal parameter, set to false ONLY if your system has support P2P memory access across PCIe (hint: AWS do not support P2P)
                        .useLegacyAveraging(true)

                        .build();

        log.info("Train model....");
        model.setListeners(new ScoreIterationListener(100));
        long timeX = System.currentTimeMillis();

        // optionally you might want to use MultipleEpochsIterator instead of manually iterating/resetting over your iterator
        //MultipleEpochsIterator mnistMultiEpochIterator = new MultipleEpochsIterator(nEpochs, mnistTrain);

        for (int i = 0; i < nEpochs; i++) {
            long time1 = System.currentTimeMillis();

            // Please note: we're feeding ParallelWrapper with iterator, not model directly
            //            wrapper.fit(mnistMultiEpochIterator);
            wrapper.fit(mnistTrain);
            long time2 = System.currentTimeMillis();
            log.info("*** Completed epoch {}, time: {} ***", i, (time2 - time1));
        }
        long timeY = System.currentTimeMillis();

        log.info("*** Training complete, time: {} ***", (timeY - timeX));

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while (mnistTest.hasNext()) {
            DataSet ds = mnistTest.next();
            INDArray output = model.output(ds.getFeatureMatrix(), false);
            eval.eval(ds.getLabels(), output);
        }
        log.info(eval.stats());
        mnistTest.reset();

        log.info("****************Example finished********************");
        wrapper.shutdown();
    }


}
