package com.example.ppedetection

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.TensorImage
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

    fun detect(bitmap: Bitmap): List<Detection> {
        // Konversi Bitmap ke TensorImage seperti komentar pada kode.
        val tensorImage =
            TensorImage.fromBitmap(bitmap)
        return detector.detect(tensorImage)
    }
}