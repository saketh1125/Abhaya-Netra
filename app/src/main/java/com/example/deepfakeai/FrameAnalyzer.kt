package com.example.deepfakeai

import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

/**
 * FrameAnalyzer implements ImageAnalysis.Analyzer for real-time deepfake detection.
 * Uses ML Kit for face detection and DeepfakeDetector for TFLite inference.
 * 
 * Key features:
 * - No per-frame Bitmap allocation (uses YUV directly)
 * - DIAG_* logging for debugging
 * - Throttling to prevent overload
 */
class FrameAnalyzer(
    private val detector: DeepfakeDetector,
    private val yuvConverter: YuvToRgbConverter,
    private val onResult: (Float, RiskLevel, Int) -> Unit,
    private val onFaceStatus: (Boolean, Rect?) -> Unit
) : ImageAnalysis.Analyzer {
    
    companion object {
        private const val TAG = "FrameAnalyzer"
        private const val MIN_FRAME_INTERVAL_MS = 200L // 5 FPS max
    }
    
    // Risk Level Classification
    enum class RiskLevel(val displayName: String, val colorHex: Int) {
        LOW("Low Risk", 0xFF4CAF50.toInt()),
        SUSPICIOUS("Suspicious", 0xFFFFC107.toInt()),
        HIGH("High Risk", 0xFFF44336.toInt())
    }
    
    // ML Kit face detector
    private val faceDetector: FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.15f)
            .build()
    )
    
    // Frame processing state
    private val isProcessing = AtomicBoolean(false)
    private val lastFrameTime = AtomicLong(0)
    private var frameCount = 0
    
    // Thresholds for risk evaluation
    private var lowRiskMax = 0.35f
    private var highRiskMin = 0.65f
    
    @androidx.camera.core.ExperimentalGetImage
    override fun analyze(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()
        
        // Throttle: Skip if processing or too soon
        if (isProcessing.get() || currentTime - lastFrameTime.get() < MIN_FRAME_INTERVAL_MS) {
            imageProxy.close()
            return
        }
        
        val mediaImage = imageProxy.image
        if (mediaImage == null) {
            imageProxy.close()
            return
        }
        
        isProcessing.set(true)
        lastFrameTime.set(currentTime)
        frameCount++
        
        val rotation = imageProxy.imageInfo.rotationDegrees
        val width = imageProxy.width
        val height = imageProxy.height
        
        // Create InputImage for ML Kit (no Bitmap allocation)
        val inputImage = InputImage.fromMediaImage(mediaImage, rotation)
        
        // Run face detection
        faceDetector.process(inputImage)
            .addOnSuccessListener { faces ->
                processFaces(imageProxy, faces, width, height)
            }
            .addOnFailureListener { e ->
                Log.e(TAG, "Face detection failed: ${e.message}")
                Log.d("DIAG_BBOX", "Frame $frameCount: face_detection_error")
                onFaceStatus(false, null)
                isProcessing.set(false)
                imageProxy.close()
            }
    }
    
    private fun processFaces(imageProxy: ImageProxy, faces: List<Face>, width: Int, height: Int) {
        if (faces.isEmpty()) {
            Log.d("DIAG_BBOX", "Frame $frameCount: no_face")
            onFaceStatus(false, null)
            isProcessing.set(false)
            imageProxy.close()
            return
        }
        
        val face = faces[0]
        val bbox = face.boundingBox
        
        // DIAG: Log bounding box
        Log.d("DIAG_BBOX", "Frame $frameCount: bbox=[L=${bbox.left},T=${bbox.top},W=${bbox.width()},H=${bbox.height()}]")
        
        onFaceStatus(true, bbox)
        
        try {
            // Convert YUV to RGB (reuses internal buffers)
            val rgbBytes = yuvConverter.imageProxyToRgbBytes(imageProxy)
            
            // DIAG: Log input hash
            val inputHash = yuvConverter.computeRegionHash(rgbBytes, width, height, bbox)
            Log.d("DIAG_INPUT", "Frame $frameCount: pixelHash=$inputHash")
            
            // Run inference
            detector.enqueueInference(
                yuvConverter, rgbBytes, width, height, bbox,
                object : DeepfakeDetector.InferenceCallback {
                    override fun onResult(fakeProb: Float, realProb: Float, frameId: Int) {
                        val riskLevel = evaluateRiskLevel(fakeProb)
                        
                        Log.i("SCORE_DEBUG", "Frame $frameId: fakeProb=${"%.4f".format(fakeProb)}")
                        
                        onResult(fakeProb, riskLevel, frameId)
                        isProcessing.set(false)
                    }
                    
                    override fun onError(error: Exception, frameId: Int) {
                        Log.e(TAG, "Inference error: ${error.message}")
                        isProcessing.set(false)
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "Processing error: ${e.message}", e)
            isProcessing.set(false)
        } finally {
            imageProxy.close()
        }
    }
    
    private fun evaluateRiskLevel(fakeProb: Float): RiskLevel {
        return when {
            fakeProb >= highRiskMin -> RiskLevel.HIGH
            fakeProb > lowRiskMax -> RiskLevel.SUSPICIOUS
            else -> RiskLevel.LOW
        }
    }
    
    /**
     * Update risk thresholds.
     */
    fun setThresholds(lowMax: Float, highMin: Float) {
        lowRiskMax = lowMax
        highRiskMin = highMin
    }
    
    /**
     * Release resources.
     */
    fun release() {
        faceDetector.close()
        Log.d(TAG, "FrameAnalyzer released")
    }
}
