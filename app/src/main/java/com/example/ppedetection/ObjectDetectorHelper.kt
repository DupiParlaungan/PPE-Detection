package com.example.ppedetection

import android.content.Context
import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector
import org.tensorflow.lite.task.gms.vision.detector.Detection
import org.tensorflow.lite.task.gms.vision.detector.ObjectDetector.ObjectDetectorOptions

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
        // Konversi Bitmap ke TensorImage seperti komentar pada kode.
        val tensorImage =
            org.tensorflow.lite.support.image.TensorImage.fromBitmap(bitmap)
        return detector.detect(tensorImage)
    }
}