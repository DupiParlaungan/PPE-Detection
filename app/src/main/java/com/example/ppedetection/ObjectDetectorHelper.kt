package com.example.ppedetection

import android.content.Context
import org.tensorflow.lite.task.vision.detector.ObjectDetector
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector.ObjectDetectorOptions

class ObjectDetectorHelper(context: Context) {
    private val detector: ObjectDetector

    init {
        val options = ObjectDetectorOptions.builder()
            .setMaxResults(1)
            .setScoreThreshold(0.5f)
            .build()
        detector = ObjectDetector.createFromFileAndOptions(
            context,
            "PPE-Detection.tflite",
            options
        )
    }

    fun detect(bitmap: android.graphics.Bitmap): List<Detection> {
        return detector.detect(bitmap)
        // The ObjectDetector API expects either a TensorImage or MlImage. Passing a
        // raw Bitmap directly results in a compilation error because there is no
        // overload of `detect` that accepts `Bitmap`. Convert the Bitmap into a
        // TensorImage before invoking the detector.
        val tensorImage = org.tensorflow.lite.support.image.TensorImage.fromBitmap(bitmap)
        return detector.detect(tensorImage)
    }
}