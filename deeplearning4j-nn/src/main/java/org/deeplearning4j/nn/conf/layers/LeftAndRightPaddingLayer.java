package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.layers.convolution.LeftAndRight;
import org.deeplearning4j.nn.params.EmptyParamInitializer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.Collection;
import java.util.Map;

@Data
public class LeftAndRightPaddingLayer extends SpecifiedLayer {

    private int[] lar_padding;

    private LeftAndRightPaddingLayer(Builder lar) {
        super(lar);
        this.lar_padding = lar.padding;
    }

    @Override
    public LeftAndRight instantiate(NeuralNetConfiguration lar_config,
                                    Collection<IterationListener> lar_listeners, int lar_index, INDArray lar_array,
                                    boolean lar_init) {
        org.deeplearning4j.nn.layers.convolution.LeftAndRight ret =
                new org.deeplearning4j.nn.layers.convolution.LeftAndRight(lar_config);
        ret.setListeners(lar_listeners);
        ret.setIndex(lar_index);
        Map<String, INDArray> paramTable = initializer().init(lar_config, lar_array, lar_init);
        ret.setParamTable(paramTable);
        ret.setConf(lar_config);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int index, InputType type) {
        int inLeft;
        int inRight;
        if (type instanceof InputType.InputTypeConvolutional) {
            InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) type;
            inLeft = conv.getHeight();
            inRight = conv.getWidth();
        } else if (type instanceof InputType.InputTypeConvolutionalFlat) {
            InputType.InputTypeConvolutionalFlat conv = (InputType.InputTypeConvolutionalFlat) type;
            inLeft = conv.getHeight();
            inRight = conv.getWidth();
        } else {
            throw new IllegalStateException(
                    "Invalid left and right type: " + type);
        }

        int outLeft = inLeft + lar_padding[0] + lar_padding[1];
        int outRight = inRight + lar_padding[2] + lar_padding[3];

        return InputType.convolutional(outLeft, outRight, 0);
    }

    @Override
    public void setNIn(InputType inputType, boolean override) {
        //No op
    }

    @Override
    public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
        if (inputType == null) {
            throw new IllegalStateException("Invalid input for TopAndBottom layer (layer name=\"" + getLayerName()
                    + "\"): input is null");
        }

        return InputTypeUtil.getPreProcessorForInputTypeCnnLayers(inputType, getLayerName());
    }

    @Override
    public double getL1ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getL2ByParam(String paramName) {
        return 0;
    }

    @Override
    public double getLearningRateByParam(String paramName) {
        return learningRate;
    }

    public String getLayerName() {
        return layerName;
    }

    public static class Builder extends Layer.Builder<Builder> {

        private int[] padding = new int[] {0, 0, 0, 0}; //Padding: top, bottom, left, right

        public Builder(int left, int right) {
            this(left, left, right, right);
        }

        public Builder(int padTop, int padBottom, int padLeft, int padRight) {
            this(new int[] {padTop, padBottom, padLeft, padRight});
        }

        public Builder(int[] padding) {
            this.padding = padding;
        }

        @Override
        @SuppressWarnings("unchecked")
        public LeftAndRightPaddingLayer build() {
            for (int p : padding) {
                if (p < 0) {
                    throw new IllegalStateException(
                            "Invalid LeftAndRight padding layer config: padding [top, bottom, left, right]"
                                    + " must be > 0 for all elements. Got: "
                                    + Arrays.toString(padding));
                }
            }

            return new LeftAndRightPaddingLayer(this);
        }
    }
}
