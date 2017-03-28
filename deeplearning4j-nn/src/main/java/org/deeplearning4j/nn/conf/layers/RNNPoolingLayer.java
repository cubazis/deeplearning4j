package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.LeftAndRight;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

@Data
@NoArgsConstructor
public class RNNPoolingLayer extends SpecifiedLayer {

    protected PoolingType poolingType;
    private int[] poolingDimensions;
    private boolean collapseDimensions;

    private RNNPoolingLayer(Builder builder) {
        super(tab);
        this.poolingType = builder.poolingType;
        this.poolingDimensions = builder.poolingDimensions;
        this.collapseDimensions = builder.collapseDimensions;
        int pnorm = builder.pnorm;
        this.layerName = builder.layerName;
    }


    @Override
    public LeftAndRight instantiate(NeuralNetConfiguration conf,
                                    Collection<IterationListener> iterationListeners, int layerIndex, INDArray layerParamsView,
                                    boolean initializeParams) {
        org.deeplearning4j.nn.layers.pooling.RNNPoolingLayer ret =
                        new org.deeplearning4j.nn.layers.pooling.RNNPoolingLayer(conf);
        ret.setListeners(iterationListeners);
        ret.setIndex(layerIndex);
        ret.setParamsViewArray(layerParamsView);
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        ret.setParamTable(paramTable);
        ret.setConf(conf);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType;
        if (collapseDimensions) {
            return InputType.feedForward(recurrent.getSize());
        } else {
            return recurrent;
        }
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //Not applicable
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        return null;
    }

    public int[] getPoolingDimensions() {
        return poolingDimensions;
    }

    public void setPoolingDimensions(int[] poolingDimensions) {
        this.poolingDimensions = poolingDimensions;
    }


    public static class Builder extends SpecifiedLayer.Builder<Builder> {

        private PoolingType poolingType = PoolingType.MAX;
        private int[] poolingDimensions;
        private int pnorm = 2;
        private boolean collapseDimensions = true;

        public Builder() {

        }

        public Builder(PoolingType poolingType) {
            this.poolingType = poolingType;
        }

        public Builder poolingDimensions(int... poolingDimensions) {
            this.poolingDimensions = poolingDimensions;
            return this;
        }

        public Builder poolingType(PoolingType poolingType) {
            this.poolingType = poolingType;
            return this;
        }

        public Builder collapseDimensions(boolean collapseDimensions) {
            this.collapseDimensions = collapseDimensions;
            return this;
        }

        public Builder pnorm(int pnorm) {
            if (pnorm <= 0)
                throw new IllegalArgumentException("Invalid input: p-norm value must be greater than 0. Got: " + pnorm);
            this.pnorm = pnorm;
            return this;
        }


        public RNNPoolingLayer build() {
            return new RNNPoolingLayer(this);
        }
    }
}
