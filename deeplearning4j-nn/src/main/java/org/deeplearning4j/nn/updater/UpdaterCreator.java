package org.deeplearning4j.nn.updater;

import org.deeplearning4j.nn.api.Model;
import org.nd4j.linalg.learning.GradientUpdater;

/**
 *
 *
 * @author Adam Gibson
 */
public class UpdaterCreator {

    private UpdaterCreator() {}

    public static org.deeplearning4j.nn.api.Updater getUpdater(Model layer) {
        if (layer instanceof MultiLayerNetwork) {
            return new MultiLayerUpdater((MultiLayerNetwork) layer);
        } else {
            return new LayerUpdater();
        }
    }

}
