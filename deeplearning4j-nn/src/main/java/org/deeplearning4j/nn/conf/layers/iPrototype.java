package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.layers.convolution.LeftAndRight;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.As;
import org.nd4j.shade.jackson.annotation.JsonTypeInfo.Id;

import java.io.Serializable;
import java.util.Collection;
import java.util.Map;

/**
 * A neural network layer.
 */
@JsonTypeInfo(use = Id.NAME, include = As.WRAPPER_OBJECT)
@JsonSubTypes(value = {
        @JsonSubTypes.Type(value = AutoEncoder.class, name = "autoEncoder"),
        @JsonSubTypes.Type(value = ConvolutionLayer.class, name = "convolution"),
        @JsonSubTypes.Type(value = Convolution1DLayer.class, name = "convolution1d"),
        @JsonSubTypes.Type(value = GravesLSTM.class, name = "gravesLSTM"),
        @JsonSubTypes.Type(value = GravesBidirectionalLSTM.class, name = "gravesBidirectionalLSTM"),
        @JsonSubTypes.Type(value = OutputLayer.class, name = "output"),
        @JsonSubTypes.Type(value = RnnOutputLayer.class, name = "rnnoutput"),
        @JsonSubTypes.Type(value = LossLayer.class, name = "loss"),
        @JsonSubTypes.Type(value = RBM.class, name = "RBM"),
        @JsonSubTypes.Type(value = DenseLayer.class, name = "dense"),
        @JsonSubTypes.Type(value = SubsamplingLayer.class, name = "subsampling"),
        @JsonSubTypes.Type(value = Subsampling1DLayer.class, name = "subsampling1d"),
        @JsonSubTypes.Type(value = BatchNormalization.class, name = "batchNormalization"),
        @JsonSubTypes.Type(value = LocalResponseNormalization.class, name = "localResponseNormalization"),
        @JsonSubTypes.Type(value = EmbeddingLayer.class, name = "embedding"),
        @JsonSubTypes.Type(value = ActivationLayer.class, name = "activation"),
        @JsonSubTypes.Type(value = VariationalAutoencoder.class, name = "VariationalAutoencoder"),
        @JsonSubTypes.Type(value = DropoutLayer.class, name = "dropout"),
        @JsonSubTypes.Type(value = GlobalPoolingLayer.class, name = "GlobalPooling"),
        @JsonSubTypes.Type(value = ZeroPaddingLayer.class, name = "zeroPadding")
})
@Data
@NoArgsConstructor
public interface iPrototype extends Serializable, Cloneable {
    String layerName = null;
    IActivation activationFn = null;
    WeightInit weightInit = null;
    double biasInit = 0;
    Distribution dist = null;
    double learningRate = 0;
    double biasLearningRate = 0;
    //learning rate after n iterations
    Map<Integer, Double> learningRateSchedule = null;
    double momentum = 0;
    Map<Integer, Double> momentumSchedule = null;
    double l1 = 0;
    double l2 = 0;
    double l1Bias = 0;
    double l2Bias = 0;
    double dropOut = 0;
    Updater updater = null;
    double rho = 0;
    double epsilon = 0;
    double rmsDecay = 0;
    double adamMeanDecay = 0;
    double adamVarDecay = 0;
    GradientNormalization gradientNormalization = GradientNormalization.None; //Clipping, rescale based on l2 norm, etc
    double gradientNormalizationThreshold = 1.0; //Threshold for l2 and element-wise gradient clipping

    /**
     * Reset the learning related configs of the layer to default. When instantiated with a global neural network configuration
     * the parameters specified in the neural network configuration will be used.
     * For internal use with the transfer learning API. Users should not have to call this method directly.
     */
    void resetLayerDefaultConfig();

    iPrototype clone();

    LeftAndRight instantiate(NeuralNetConfiguration conf,
                             Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView,
                             boolean initializeParams);
    ParamInitializer initializer();
    InputType getOutputType(int layerIndex, InputType inputType);
    void setNIn(InputType inputType, boolean override);
    InputPreProcessor getPreProcessorForInputType(InputType inputType);
    double getL1ByParam(String paramName);
    double getL2ByParam(String paramName);
    double getLearningRateByParam(String paramName);
    Updater getUpdaterByParam(String paramName);

    abstract class Builder<T extends Builder<T>> {
        protected String layerName = null;
        protected IActivation activationFn = null;
        protected WeightInit weightInit = null;
        protected double biasInit = Double.NaN;
        protected Distribution dist = null;
        protected double learningRate = Double.NaN;
        protected double biasLearningRate = Double.NaN;
        protected Map<Integer, Double> learningRateSchedule = null;
        protected double momentum = Double.NaN;
        protected Map<Integer, Double> momentumAfter = null;
        protected double l1 = Double.NaN;
        protected double l2 = Double.NaN;
        protected double l1Bias = Double.NaN;
        protected double l2Bias = Double.NaN;
        protected double dropOut = Double.NaN;
        protected Updater updater = null;
        protected double rho = Double.NaN;
        protected double epsilon = Double.NaN;
        protected double rmsDecay = Double.NaN;
        protected double adamMeanDecay = Double.NaN;
        protected double adamVarDecay = Double.NaN;
        protected GradientNormalization gradientNormalization = null;
        protected double gradientNormalizationThreshold = Double.NaN;
        protected LearningRatePolicy learningRatePolicy = null;

        public T name(String layerName) {
            this.layerName = layerName;
            return (T) this;
        }

        @Deprecated
        public T activation(String activationFunction) {
            return activation(Activation.fromString(activationFunction));
        }

        public T activation(IActivation activationFunction) {
            this.activationFn = activationFunction;
            return (T) this;
        }

        public T activation(Activation activation) {
            return activation(activation.getActivationFunction());
        }

        public T weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return (T) this;
        }

        public T biasInit(double biasInit) {
            this.biasInit = biasInit;
            return (T) this;
        }
        public T dist(Distribution dist) {
            this.dist = dist;
            return (T) this;
        }

        public T learningRate(double learningRate) {
            this.learningRate = learningRate;
            return (T) this;
        }

        public T biasLearningRate(double biasLearningRate) {
            this.biasLearningRate = biasLearningRate;
            return (T) this;
        }
        public T learningRateSchedule(Map<Integer, Double> learningRateSchedule) {
            this.learningRateSchedule = learningRateSchedule;
            return (T) this;
        }

        public T l1(double l1) {
            this.l1 = l1;
            return (T) this;
        }
        public T l2(double l2) {
            this.l2 = l2;
            return (T) this;
        }

        public T l1Bias(double l1Bias) {
            this.l1Bias = l1Bias;
            return (T) this;
        }

        public T l2Bias(double l2Bias) {
            this.l2Bias = l2Bias;
            return (T) this;
        }

        public T dropOut(double dropOut) {
            this.dropOut = dropOut;
            return (T) this;
        }

        public T momentum(double momentum) {
            this.momentum = momentum;
            return (T) this;
        }

        public T momentumAfter(Map<Integer, Double> momentumAfter) {
            this.momentumAfter = momentumAfter;
            return (T) this;
        }

        public T updater(Updater updater) {
            this.updater = updater;
            return (T) this;
        }

        public T rho(double rho) {
            this.rho = rho;
            return (T) this;
        }

        public T rmsDecay(double rmsDecay) {
            this.rmsDecay = rmsDecay;
            return (T) this;
        }

        public T epsilon(double epsilon) {
            this.epsilon = epsilon;
            return (T) this;
        }


        public T adamMeanDecay(double adamMeanDecay) {
            this.adamMeanDecay = adamMeanDecay;
            return (T) this;
        }


        public T adamVarDecay(double adamVarDecay) {
            this.adamVarDecay = adamVarDecay;
            return (T) this;
        }

        public T gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return (T) this;
        }

        public T gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return (T) this;
        }

        public T learningRateDecayPolicy(LearningRatePolicy policy) {
            this.learningRatePolicy = policy;
            return (T) this;
        }

        public abstract <E extends Layer> E build();
    }
}
