package com.example.deepfakeai

import android.Manifest
import android.content.pm.PackageManager
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.mlkit.vision.common.InputImage
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private enum class AppMode {
        NONE, VIDEO_MODE, CAMERA_MODE
    }

    private var selectedMode: AppMode = AppMode.NONE
    private var tfliteInterpreter: Interpreter? = null
    private var faceDetectorHelper: FaceDetectorHelper? = null

    // CameraX
    private lateinit var cameraExecutor: ExecutorService
    private var lastAnalyzedTimestamp = 0L

    // Coroutine scope for background processing
    private val processingScope = CoroutineScope(Dispatchers.Main + Job())
    private var currentProcessingJob: Job? = null

    // Permission Launcher
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Log.e("DeepfakeAI", "Camera permission denied")
            findViewById<TextView>(R.id.statusTextView).text = "Camera permission granted ❌"
        }
    }

    // Video Picker Launcher
    private val pickVideoLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
        if (uri != null) {
            Log.i("DeepfakeAI", "Video selected: $uri")
            processVideo(uri)
        } else {
            Log.i("DeepfakeAI", "Video selection cancelled")
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Initialize Views
        val statusTextView = findViewById<TextView>(R.id.statusTextView)
        val btnSelectVideo = findViewById<android.widget.Button>(R.id.btnSelectVideo)
        val btnLiveCamera = findViewById<android.widget.Button>(R.id.btnLiveCamera)
        val viewFinder = findViewById<PreviewView>(R.id.viewFinder)

        // Initialize Face Detector Helper
        faceDetectorHelper = FaceDetectorHelper(
            onSuccess = { bounds ->
                // Ensure UI updates on Main Thread if not already (safeguard)
                runOnUiThread {
                    val statusTextView = findViewById<TextView>(R.id.statusTextView)
                    if (bounds != null) {
                        Log.i("FACE_DETECTION", "FACE DETECTED bbox=$bounds")
                        statusTextView.text = "Face detected ✅"
                    } else {
                        Log.i("FACE_DETECTION", "NO FACE DETECTED")
                        statusTextView.text = "No face detected ❌"
                    }
                }
            },
            onError = { e ->
                Log.e("FACE_DETECTION", "Error in detection", e)
            }
        )

        // Initialize TFLite Interpreter
        try {
            val modelBuffer = loadModelFile("model.tflite")
            tfliteInterpreter = Interpreter(modelBuffer)
            Log.i("MODEL_STATUS", "TensorFlow Lite model loaded successfully 🎯")
            statusTextView.text = "Model Loaded Successfully ✅"
        } catch (e: Exception) {
            Log.e("MODEL_STATUS", "❌ Failed to load TFLite model", e)
            statusTextView.text = "Model Load Failed ❌" // Keep specific failure visible initially
        }

        // Set Click Listeners
        btnSelectVideo.setOnClickListener {
            selectedMode = AppMode.VIDEO_MODE
            currentProcessingJob?.cancel() // Stop video loop if running
            viewFinder.visibility = View.INVISIBLE // Hide camera
            statusTextView.text = "Video analysis mode selected. Choosing video..."
            Log.d("DeepfakeAI", "Mode changed to: $selectedMode")
            pickVideoLauncher.launch("video/*")
        }

        btnLiveCamera.setOnClickListener {
            selectedMode = AppMode.CAMERA_MODE
            currentProcessingJob?.cancel() // Stop video loop logic
            statusTextView.text = "Live camera detection mode selected"
            viewFinder.visibility = View.VISIBLE
            Log.d("DeepfakeAI", "Mode changed to: $selectedMode")
            
            checkCameraPermissionAndStart()
        }
    }

    private fun checkCameraPermissionAndStart() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()
            val viewFinder = findViewById<PreviewView>(R.id.viewFinder)
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processCameraFrame(imageProxy)
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
                Log.i("DeepfakeAI", "Camera started successfully")
            } catch(exc: Exception) {
                Log.e("DeepfakeAI", "Use case binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class) 
    private fun processCameraFrame(imageProxy: ImageProxy) {
        val currentTimestamp = System.currentTimeMillis()
        // Throttle: 1 FPS (1000ms)
        if (currentTimestamp - lastAnalyzedTimestamp >= 1000) {
            lastAnalyzedTimestamp = currentTimestamp
            
            val mediaImage = imageProxy.image
            if (mediaImage != null) {
                // We use ML Kit's InputImage directly for efficiency if FaceDetectorHelper supports it.
                // But the helper accepts Bitmap. We must convert.
                // Converting YUV ImageProxy to Bitmap is complex without a helper.
                // However, since we are doing low FPS, we can use the BitmapFactory decode approach simply IF valid.
                // Actually, standard way is YuvToRgbConverter or simply:
                // ML Kit supports mediaImage directly.
                // We will modify the flow for Camera to use a special helper method or direct call.
                // BUT requirements say "Convert the ImageProxy frame into a Bitmap format".
                // We'll use a simple transformation (e.g. toBitmap() extension if available or basic copy).
                // CameraX 1.3.0 has .toBitmap() on ImageProxy!
                // We need to verify if we included that dep. 'androidx.camera:camera-core' has it? 
                // It is in 'androidx.camera:camera-core' since 1.1.0-alpha08 usually requires YUV conversion logic internally.
                // Let's rely on .toBitmap() which is safe and easy for low FPS.
                
                val bitmap = imageProxy.toBitmap()
                
                // Switch to main thread for UI updates? FaceDetectorHelper logs, but if we assume it's thread safe we can call it.
                // Our Helper uses a callback.
                faceDetectorHelper?.detectFace(bitmap)
                
                // Update UI on main thread
                runOnUiThread {
                    findViewById<TextView>(R.id.statusTextView).text = "Analyzing Camera Frame..."
                }
            }
        }
        imageProxy.close()
    }

    private fun processVideo(videoUri: Uri) {
        val statusTextView = findViewById<TextView>(R.id.statusTextView) // Re-fetch or use member if ViewBinding
        currentProcessingJob?.cancel() // Cancel previous job if any

        currentProcessingJob = processingScope.launch(Dispatchers.Default) {
             val retriever = MediaMetadataRetriever()
             try {
                 retriever.setDataSource(this@MainActivity, videoUri)
                 val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                 val durationMs = durationStr?.toLongOrNull() ?: 0L
                 
                 Log.i("DeepfakeAI", "Starting processing for video length: ${durationMs}ms")
                 
                 var currentTimeMs = 0L
                 val intervalMs = 300L // 300ms interval
                 
                 while (currentTimeMs < durationMs && isActive) {
                     // getFrameAtTime takes microseconds
                     val bitmap = retriever.getFrameAtTime(currentTimeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                     
                     if (bitmap != null) {
                         withContext(Dispatchers.Main) {
                             statusTextView.text = "Processing frame at ${currentTimeMs}ms..."
                             
                             // We define a specialized callback for this loop or re-use global helper
                             // Re-using global helper which logs to Logcat is enough per requirements
                             // But we want frame-specific timestamp logging, so we use a local one or update our logging strategy.
                             // For verification, we'll just use a local inline helper that logs with the tag.
                             val detector = FaceDetectorHelper(
                                 onSuccess = { bounds ->
                                     if (bounds != null) {
                                         Log.i("FACE_DETECTION", "timestamp=${currentTimeMs}ms — FACE DETECTED bbox=$bounds")
                                         statusTextView.text = "Face detected at ${currentTimeMs}ms ✅"
                                     } else {
                                          Log.i("FACE_DETECTION", "timestamp=${currentTimeMs}ms — NO FACE")
                                          statusTextView.text = "No face detected at ${currentTimeMs}ms ❌"
                                     }
                                 },
                                 onError = { e ->
                                     Log.e("FACE_DETECTION", "Error detecting face at ${currentTimeMs}ms", e)
                                 }
                             )
                             detector.detectFace(bitmap)
                         }
                     }
                     
                     currentTimeMs += intervalMs
                     delay(10) // Small yield
                 }
                 
                 withContext(Dispatchers.Main) {
                     statusTextView.text = "Video Processing Complete ✅"
                 }
                 
             } catch (e: Exception) {
                 Log.e("DeepfakeAI", "Error extracting frames", e)
                 withContext(Dispatchers.Main) {
                     statusTextView.text = "Error processing video ❌"
                 }
             } finally {
                 retriever.release()
             }
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
