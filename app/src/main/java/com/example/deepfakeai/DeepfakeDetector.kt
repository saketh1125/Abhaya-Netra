package com.example.deepfakeai

import android.content.Context
import android.graphics.Rect
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * DeepfakeDetector encapsulates TFLite inference with preallocated buffers.
 * Thread-safe, runs inference on background executor.
 */
class DeepfakeDetector(
    private val context: Context,
    private val modelFileName: String = "model.tflite",
    private val inputWidth: Int = 224,
    private val inputHeight: Int = 224
) {
    companion object {
        private const val TAG = "DeepfakeDetector"
        private const val NUM_THREADS = 2
    }
    
    // TFLite interpreter
    private var interpreter: Interpreter? = null
    
    // Preallocated buffers (created once, reused per frame)
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * 1 * inputWidth * inputHeight * 3)
        .order(ByteOrder.nativeOrder())
    private val outputBuffer: Array<FloatArray> = Array(1) { FloatArray(2) }
    
    // Background executor for inference
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    
    // Frame counter for logging
    private val frameCounter = AtomicInteger(0)
    
    // Callback interface for inference results
    interface InferenceCallback {
        fun onResult(fakeProb: Float, realProb: Float, frameId: Int)
        fun onError(error: Exception, frameId: Int)
    }
    
    init {
        loadModel()
        logModelInfo()
    }
    
    private fun loadModel() {
        try {
            val modelBuffer = loadModelFile(modelFileName)
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)
            }
            interpreter = Interpreter(modelBuffer, options)
            Log.i(TAG, "Model loaded successfully: $modelFileName")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model: ${e.message}", e)
            throw e
        }
    }
    
    private fun loadModelFile(fileName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(fileName)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    private fun logModelInfo() {
        interpreter?.let { interp ->
            val inputTensor = interp.getInputTensor(0)
            val outputTensor = interp.getOutputTensor(0)
            Log.i(TAG, "=== MODEL TENSOR INFO ===")
            Log.i(TAG, "Input: shape=${inputTensor.shape().contentToString()}, dtype=${inputTensor.dataType()}")
            Log.i(TAG, "Output: shape=${outputTensor.shape().contentToString()}, dtype=${outputTensor.dataType()}")
            Log.i(TAG, "=========================")
        }
    }
    
    /**
     * Enqueue inference with pre-converted RGB bytes.
     * Runs on background executor and calls callback on completion.
     */
    fun enqueueInference(
        yuvConverter: YuvToRgbConverter,
        rgbBytes: ByteArray,
        srcWidth: Int,
        srcHeight: Int,
        faceRect: Rect,
        callback: InferenceCallback
    ) {
        val frameId = frameCounter.incrementAndGet()
        
        inferenceExecutor.submit {
            try {
                // Step 1: Crop, resize, and normalize into input buffer
                yuvConverter.cropResizeNormalizeToBuffer(
                    rgbBytes, srcWidth, srcHeight,
                    faceRect, inputWidth, inputHeight,
                    inputBuffer
                )
                
                // Step 2: Run inference
                synchronized(this) {
                    inputBuffer.rewind()
                    interpreter?.run(inputBuffer, outputBuffer)
                }
                
                // Step 3: Extract results
                val out0 = outputBuffer[0][0]
                val out1 = outputBuffer[0][1]
                
                // DIAGNOSTIC: Log raw output
                Log.d("DIAG_OUTPUT", "Frame $frameId: output[0]=${"%.6f".format(out0)}, output[1]=${"%.6f".format(out1)}, sum=${"%.4f".format(out0 + out1)}")
                
                // Interpretation: output[0] = fake probability (based on diagnostic results)
                val fakeProb = out0
                val realProb = out1
                
                Log.d("DIAG_INTERPRET", "Frame $frameId: fakeProb=${"%.4f".format(fakeProb)}, realProb=${"%.4f".format(realProb)}")
                
                callback.onResult(fakeProb, realProb, frameId)
                
            } catch (e: Exception) {
                Log.e(TAG, "Inference error at frame $frameId: ${e.message}", e)
                callback.onError(e, frameId)
            }
        }
    }
    
    /**
     * Direct synchronous inference (for testing or single-frame use).
     * Returns pair of (fakeProb, realProb).
     */
    @Synchronized
    fun runInferenceSync(
        yuvConverter: YuvToRgbConverter,
        rgbBytes: ByteArray,
        srcWidth: Int,
        srcHeight: Int,
        faceRect: Rect
    ): Pair<Float, Float> {
        val frameId = frameCounter.incrementAndGet()
        
        // Crop, resize, normalize
        yuvConverter.cropResizeNormalizeToBuffer(
            rgbBytes, srcWidth, srcHeight,
            faceRect, inputWidth, inputHeight,
            inputBuffer
        )
        
        // Run inference
        inputBuffer.rewind()
        interpreter?.run(inputBuffer, outputBuffer)
        
        val out0 = outputBuffer[0][0]
        val out1 = outputBuffer[0][1]
        
        Log.d("DIAG_OUTPUT", "Frame $frameId: output[0]=${"%.6f".format(out0)}, output[1]=${"%.6f".format(out1)}")
        
        return Pair(out0, out1)
    }
    
    /**
     * Reset interpreter state (useful when switching modes).
     */
    fun reset() {
        interpreter?.resetVariableTensors()
        frameCounter.set(0)
        Log.d(TAG, "Detector reset")
    }
    
    /**
     * Release resources.
     */
    fun release() {
        inferenceExecutor.shutdown()
        interpreter?.close()
        interpreter = null
        Log.d(TAG, "Detector released")
    }
}
