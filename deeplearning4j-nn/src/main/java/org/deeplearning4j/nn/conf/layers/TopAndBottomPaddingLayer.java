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
public class TopAndBottomPaddingLayer extends SpecifiedLayer {

    private int[] tab_padding;

    private TopAndBottomPaddingLayer(Builder tab) {
        super(tab);
        this.tab_padding = tab.padding;
    }

    @Override
    public LeftAndRight instantiate(NeuralNetConfiguration tab_config,
                                    Collection<IterationListener> tab_listeners, int tab_index, INDArray tab_array,
                                    boolean tab_init) {
        org.deeplearning4j.nn.layers.convolution.TopAndBottom ret =
                new org.deeplearning4j.nn.layers.convolution.TopAndBottom(tab_config);
        ret.setListeners(tab_listeners);
        ret.setIndex(tab_index);
        Map<String, INDArray> paramTable = initializer().init(tab_config, tab_array, tab_init);
        ret.setParamTable(paramTable);
        ret.setConf(tab_config);
        return ret;
    }

    @Override
    public ParamInitializer initializer() {
        return EmptyParamInitializer.getInstance();
    }

    @Override
    public InputType getOutputType(int index, InputType type) {
        int inTop;
        int inBot;
        if (type instanceof InputType.InputTypeConvolutional) {
            InputType.InputTypeConvolutional conv = (InputType.InputTypeConvolutional) type;
            inTop = conv.getHeight();
            inBot = conv.getWidth();
        } else if (type instanceof InputType.InputTypeConvolutionalFlat) {
            InputType.InputTypeConvolutionalFlat conv = (InputType.InputTypeConvolutionalFlat) type;
            inTop = conv.getHeight();
            inBot = conv.getWidth();
        } else {
            throw new IllegalStateException(
                            "Invalid top and bottom type: " + type);
        }

        int outTop = inTop + tab_padding[0] + tab_padding[1];
        int outBot = inBot + tab_padding[2] + tab_padding[3];

        return InputType.convolutional(outTop, outBot, 0);
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

        public Builder(int top, int bot) {
            this(top, top, bot, bot);
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
                                    "Invalid TopAndBottom padding layer config: padding [top, bottom, left, right]"
                                                    + " must be > 0 for all elements. Got: "
                                                    + Arrays.toString(padding));
                }
            }

            return new TopAndBottomPaddingLayer(this);
        }
    }
}
