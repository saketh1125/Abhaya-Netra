package com.example.deepfakeai

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions

class FaceDetectorHelper {
    private val detector: FaceDetector
    
    companion object {
        private const val TAG = "FACE_DETECT"
    }

    init {
        // Optimized for REALTIME performance
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setMinFaceSize(0.15f) // Detect smaller faces
            .build()

        detector = FaceDetection.getClient(options)
        Log.d(TAG, "FaceDetector initialized with FAST mode, minFaceSize=0.15")
    }

    fun detectFace(
        bitmap: Bitmap,
        onSuccess: (Rect?) -> Unit,
        onError: (Exception) -> Unit
    ) {
        val startTime = System.currentTimeMillis()
        Log.d(TAG, "detectFace() called. Bitmap: ${bitmap.width}x${bitmap.height}, config=${bitmap.config}")
        
        val image = InputImage.fromBitmap(bitmap, 0)
        Log.d(TAG, "InputImage created, running ML Kit detector...")
        
        detector.process(image)
            .addOnSuccessListener { faces ->
                val elapsed = System.currentTimeMillis() - startTime
                Log.d(TAG, "Detection SUCCESS in ${elapsed}ms. Faces found: ${faces.size}")
                
                if (faces.isNotEmpty()) {
                    val face = faces[0]
                    Log.d(TAG, "Face[0] boundingBox=${face.boundingBox}")
                    onSuccess(face.boundingBox)
                } else {
                    Log.d(TAG, "No faces in result list")
                    onSuccess(null)
                }
            }
            .addOnFailureListener { e ->
                val elapsed = System.currentTimeMillis() - startTime
                Log.e(TAG, "Detection FAILED in ${elapsed}ms", e)
                onError(e)
            }
    }
}

