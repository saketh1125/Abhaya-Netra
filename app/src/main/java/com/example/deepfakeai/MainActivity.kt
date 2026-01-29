package com.example.deepfakeai

import android.Manifest
import android.graphics.Bitmap
import android.content.pm.PackageManager
import android.media.MediaMetadataRetriever
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.TextView
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
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


    // Risk Level Classification with UX Wording
    private enum class RiskLevel(
        val displayName: String, 
        val explanation: String,
        val colorHex: Int, // Android Color Int
        val iconResId: Int // Android Drawable Res ID
    ) {
        LOW(
            "Low Risk",
            "No strong signs of manipulation detected. Continue normally, but stay aware.",
            0xFF4CAF50.toInt(), // Green
            android.R.drawable.checkbox_on_background
        ),
        SUSPICIOUS(
            "Potential Manipulation Detected",
            "Some signals suggest this content may be altered. Verify the source before acting on it.",
            0xFFFFC107.toInt(), // Amber/Yellow
            android.R.drawable.stat_notify_error
        ),
        HIGH(
            "High Risk of Deepfake",
            "Strong indicators of synthetic or manipulated media. Treat this content with caution and verify through trusted channels.",
            0xFFF44336.toInt(), // Red
            android.R.drawable.ic_delete
        )
    }
    
    // Hysteresis Thresholds (to prevent rapid state flipping)
    // Hysteresis Constant
    companion object {
        const val MAX_SESSION_EVENTS = 500
    }
    
    // Configurable Settings (Default Values)
    private var highRiskEntry = 0.70f
    private var highRiskExit = 0.60f
    private var lowRiskEntry = 0.30f
    private var lowRiskExit = 0.40f
    private var isHysteresisEnabled = true
    private var smoothingWindowSize = 15
    
    // Settings State Class
    private data class SettingsState(
        var smoothingWindow: Int = 15,
        var lowRiskMax: Float = 0.35f,
        var highRiskMin: Float = 0.65f,
        var hysteresisEnabled: Boolean = true
    )
    
    private val currentSettings = SettingsState()
    
    // Session Stats Tracking
    private data class SessionStats(
        var sessionStartTime: Long = 0L,
        var framesProcessed: Int = 0,
        var facesDetected: Int = 0,
        var inferenceCalls: Int = 0,
        var lowRiskCount: Int = 0,
        var suspiciousRiskCount: Int = 0,
        var highRiskCount: Int = 0,
        var scoreSum: Float = 0f,
        var peakScore: Float = 0f
    ) {
        val averageScore: Float
            get() = if (inferenceCalls > 0) scoreSum / inferenceCalls else 0f
        
        val sessionDurationMs: Long
            get() = if (sessionStartTime > 0) System.currentTimeMillis() - sessionStartTime else 0L
        
        fun reset() {
            sessionStartTime = System.currentTimeMillis()
            framesProcessed = 0
            facesDetected = 0
            inferenceCalls = 0
            lowRiskCount = 0
            suspiciousRiskCount = 0
            highRiskCount = 0
            scoreSum = 0f
            peakScore = 0f
        }
    }
    
    private val sessionStats = SessionStats()
    
    private var currentRiskLevel: RiskLevel = RiskLevel.LOW
    
    // We need to access scoreBuffer to resize it, so we'll move it here or handle resizing dynamically
    // Currently scoreBuffer is defined further down. We'll leave it but dynamic resizing needs care.

    private var selectedMode: AppMode = AppMode.NONE
    private var tfliteInterpreter: Interpreter? = null
    private var faceDetectorHelper: FaceDetectorHelper? = null

    // NEW: YUV Pipeline Components
    private lateinit var deepfakeDetector: DeepfakeDetector
    private lateinit var yuvConverter: YuvToRgbConverter
    private var frameAnalyzer: FrameAnalyzer? = null

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

    // Connectivity Members
    private lateinit var connectivityManager: ConnectivityManager
    private lateinit var networkCallback: ConnectivityManager.NetworkCallback

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Connectivity Logic
        connectivityManager = getSystemService(ConnectivityManager::class.java)
        val badgeOffline = findViewById<TextView>(R.id.badgeOffline)
        
        networkCallback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                runOnUiThread {
                    badgeOffline.visibility = View.GONE
                    Log.i("TRUST_BADGE", "Online mode detected")
                }
            }

            override fun onLost(network: Network) {
                runOnUiThread {
                    badgeOffline.visibility = View.VISIBLE
                    Log.i("TRUST_BADGE", "Offline mode active")
                }
            }
        }
        
        val networkRequest = NetworkRequest.Builder()
            .addCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .build()
        connectivityManager.registerNetworkCallback(networkRequest, networkCallback)
        
        // Initial check (simplified)
        val activeNetwork = connectivityManager.activeNetwork
        val capabilities = connectivityManager.getNetworkCapabilities(activeNetwork)
        val isConnected = capabilities?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
        badgeOffline.visibility = if (isConnected) View.GONE else View.VISIBLE
        Log.i("TRUST_BADGE", "Initial State: ${if (isConnected) "Online" else "Offline"}")

        // Initialize Views
        val statusTextView = findViewById<TextView>(R.id.statusTextView)
        val btnSelectVideo = findViewById<android.widget.Button>(R.id.btnSelectVideo)
        val btnLiveCamera = findViewById<android.widget.Button>(R.id.btnLiveCamera)
        val viewFinder = findViewById<PreviewView>(R.id.viewFinder)
        
        // Settings Views
        val btnSettings = findViewById<android.widget.ImageButton>(R.id.btnSettings)
        val settingsPanel = findViewById<android.widget.ScrollView>(R.id.settingsPanel)
        val btnCloseSettings = findViewById<android.widget.Button>(R.id.btnCloseSettings)
        val btnResetSettings = findViewById<android.widget.Button>(R.id.btnResetSettings)
        
        val sliderSmoothing = findViewById<android.widget.SeekBar>(R.id.sliderSmoothing)
        val sliderLowRisk = findViewById<android.widget.SeekBar>(R.id.sliderLowRisk)
        val sliderHighRisk = findViewById<android.widget.SeekBar>(R.id.sliderHighRisk)
        val switchHysteresis = findViewById<android.widget.Switch>(R.id.switchHysteresis)
        
        val lblSmoothing = findViewById<TextView>(R.id.lblSmoothing)
        val lblLowRisk = findViewById<TextView>(R.id.lblLowRisk)
        val lblHighRisk = findViewById<TextView>(R.id.lblHighRisk)
        
        // Settings Toggle Logic
        btnSettings.setOnClickListener {
            if (settingsPanel.visibility == View.VISIBLE) {
                settingsPanel.visibility = View.GONE
            } else {
                settingsPanel.visibility = View.VISIBLE
            }
        }
        
        btnCloseSettings.setOnClickListener {
            settingsPanel.visibility = View.GONE
        }
        
        // Settings Logic Functions
        fun updateThresholds() {
            lowRiskEntry = currentSettings.lowRiskMax - 0.05f
            lowRiskExit = currentSettings.lowRiskMax + 0.05f
            
            highRiskEntry = currentSettings.highRiskMin + 0.05f
            highRiskExit = currentSettings.highRiskMin - 0.05f
            
            isHysteresisEnabled = currentSettings.hysteresisEnabled
            smoothingWindowSize = currentSettings.smoothingWindow
            
            Log.i("SETTINGS_UPDATE", "Thresholds Updated: LowMax=${currentSettings.lowRiskMax} HighMin=${currentSettings.highRiskMin} Smooth=$smoothingWindowSize Hyst=$isHysteresisEnabled")
        }
        
        // Listeners
        sliderSmoothing.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                currentSettings.smoothingWindow = progress
                lblSmoothing.text = "Smoothing Window: $progress"
                updateThresholds()
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })
        
        sliderLowRisk.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                val value = progress / 100f
                // Validate visual constraints
                if (value >= currentSettings.highRiskMin) {
                   // Prevent overlap (simple clamp or just ignore)
                   // For now, allow but logic will handle.
                }
                currentSettings.lowRiskMax = value
                lblLowRisk.text = "Low Risk Max: $value"
                updateThresholds()
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })
        
        sliderHighRisk.setOnSeekBarChangeListener(object : android.widget.SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: android.widget.SeekBar?, progress: Int, fromUser: Boolean) {
                val value = progress / 100f
                currentSettings.highRiskMin = value
                lblHighRisk.text = "High Risk Min: $value"
                updateThresholds()
            }
            override fun onStartTrackingTouch(seekBar: android.widget.SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: android.widget.SeekBar?) {}
        })
        
        switchHysteresis.setOnCheckedChangeListener { _, isChecked ->
            currentSettings.hysteresisEnabled = isChecked
            updateThresholds()
            Log.i("SETTINGS_UPDATE", "Hysteresis Toggled: $isChecked")
        }
        
        btnResetSettings.setOnClickListener {
            // Restore Defaults
            currentSettings.smoothingWindow = 15
            currentSettings.lowRiskMax = 0.35f
            currentSettings.highRiskMin = 0.65f
            currentSettings.hysteresisEnabled = true
            
            // Update UI
            sliderSmoothing.progress = 15
            sliderLowRisk.progress = 35
            sliderHighRisk.progress = 65
            switchHysteresis.isChecked = true
            
            lblSmoothing.text = "Smoothing Window: 15"
            lblLowRisk.text = "Low Risk Max: 0.35"
            lblHighRisk.text = "High Risk Min: 0.65"
            
            updateThresholds()
            Log.i("SETTINGS_UPDATE", "Settings Reset to Defaults")
        }
        
        // Initial init
        updateThresholds()

        // Initialize Face Detector Helper
        faceDetectorHelper = FaceDetectorHelper()

        // Initialize NEW YUV Pipeline Components
        yuvConverter = YuvToRgbConverter()
        deepfakeDetector = DeepfakeDetector(this, "model.tflite")
        Log.i("DeepfakeAI", "YUV Pipeline components initialized")

        // Initialize TFLite Interpreter
        try {
            val modelBuffer = loadModelFile("model.tflite")
            tfliteInterpreter = Interpreter(modelBuffer)
            Log.i("MODEL_STATUS", "TensorFlow Lite model loaded successfully 🎯")
            statusTextView.text = "Model Loaded Successfully ✅"
            
            // DIAGNOSTIC: Log model input/output tensor shapes
            val inputTensor = tfliteInterpreter?.getInputTensor(0)
            val outputTensor = tfliteInterpreter?.getOutputTensor(0)
            Log.i("MODEL_SHAPE", "=== MODEL TENSOR INFO ===")
            Log.i("MODEL_SHAPE", "Input: shape=${inputTensor?.shape()?.contentToString()}, dtype=${inputTensor?.dataType()}")
            Log.i("MODEL_SHAPE", "Output: shape=${outputTensor?.shape()?.contentToString()}, dtype=${outputTensor?.dataType()}")
            Log.i("MODEL_SHAPE", "=========================")
        } catch (e: Exception) {
            Log.e("MODEL_STATUS", "❌ Failed to load TFLite model", e)
            statusTextView.text = "Model Load Failed ❌" // Keep specific failure visible initially
        }

        // Set Click Listeners
        btnSelectVideo.setOnClickListener {
            selectedMode = AppMode.VIDEO_MODE
            currentProcessingJob?.cancel() // Stop video loop if running
            synchronized(scoreBuffer) { scoreBuffer.clear() } // Reset score buffer
            currentRiskLevel = RiskLevel.LOW // Reset risk state
            sessionStats.reset() // Reset session stats
            tfliteInterpreter?.resetVariableTensors() // Optional: cleaner state
            Log.i("SESSION_LOG", "Session reset for VIDEO mode")
            
            // hide banner
            findViewById<View>(R.id.riskBannerLayout).visibility = View.INVISIBLE
            
            viewFinder.visibility = View.INVISIBLE // Hide camera
            statusTextView.text = "Video analysis mode selected. Choosing video..."
            Log.d("DeepfakeAI", "Mode changed to: $selectedMode")
            pickVideoLauncher.launch("video/*")
        }

        btnLiveCamera.setOnClickListener {
            selectedMode = AppMode.CAMERA_MODE
            currentProcessingJob?.cancel() // Stop video loop logic
            synchronized(scoreBuffer) { scoreBuffer.clear() } // Reset score buffer
            currentRiskLevel = RiskLevel.LOW // Reset risk state
            sessionStats.reset() // Reset session stats
            Log.i("SESSION_LOG", "Session reset for CAMERA mode")
            
            // hide banner
            findViewById<View>(R.id.riskBannerLayout).visibility = View.INVISIBLE
            
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

            // Create FrameAnalyzer with new YUV pipeline
            frameAnalyzer = FrameAnalyzer(
                detector = deepfakeDetector,
                yuvConverter = yuvConverter,
                onResult = { fakeProb, riskLevel, frameId ->
                    // Update UI on main thread
                    runOnUiThread {
                        if (isFinishing || isDestroyed) return@runOnUiThread
                        
                        val percentage = (fakeProb * 100).toInt()
                        val statusTextView = findViewById<TextView>(R.id.statusTextView)
                        
                        // Map FrameAnalyzer.RiskLevel to MainActivity.RiskLevel
                        val mainRiskLevel = when (riskLevel) {
                            FrameAnalyzer.RiskLevel.LOW -> RiskLevel.LOW
                            FrameAnalyzer.RiskLevel.SUSPICIOUS -> RiskLevel.SUSPICIOUS
                            FrameAnalyzer.RiskLevel.HIGH -> RiskLevel.HIGH
                        }
                        
                        statusTextView.setTextColor(mainRiskLevel.colorHex)
                        statusTextView.text = "${mainRiskLevel.displayName}\n${mainRiskLevel.explanation}\n\nConfidence: $percentage%"
                        
                        updateRiskVisuals(mainRiskLevel, fakeProb)
                        
                        // Update session stats
                        sessionStats.inferenceCalls++
                        sessionStats.scoreSum += fakeProb
                        if (fakeProb > sessionStats.peakScore) {
                            sessionStats.peakScore = fakeProb
                        }
                        updateSessionSummaryUI()
                    }
                },
                onFaceStatus = { faceDetected, bbox ->
                    runOnUiThread {
                        val statusTextView = findViewById<TextView>(R.id.statusTextView)
                        if (!faceDetected) {
                            statusTextView.text = "No face detected ❌"
                        }
                    }
                }
            )

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor, frameAnalyzer!!)
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
                Log.i("DeepfakeAI", "Camera started with YUV pipeline ✅")
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
            sessionStats.framesProcessed++ // Track frame
            
            val mediaImage = imageProxy.image
            if (mediaImage != null) {
                 val bitmap = imageProxy.toBitmap()
                
                 faceDetectorHelper?.detectFace(
                     bitmap = bitmap,
                     onSuccess = { bounds ->
                        runOnUiThread {
                            val statusTextView = findViewById<TextView>(R.id.statusTextView)
                            if (bounds != null) {
                                sessionStats.facesDetected++ // Track face
                                Log.i("FACE_DETECTION", "FACE DETECTED bbox=$bounds")
                                statusTextView.text = "Face detected ✅"
                                
                                // Preprocess off main thread
                                processingScope.launch(Dispatchers.Default) {
                                    preprocessFaceForModel(bitmap, bounds)
                                }
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
                 
                 // Initial UI update
                 runOnUiThread {
                    findViewById<TextView>(R.id.statusTextView).text = "Analyzing Camera Frame..."
                 }
            }
        }
        imageProxy.close()
    }

    private fun processVideo(videoUri: Uri) {
        val statusTextView = findViewById<TextView>(R.id.statusTextView)
        currentProcessingJob?.cancel()

        currentProcessingJob = processingScope.launch(Dispatchers.Default) {
            val retriever = MediaMetadataRetriever()
            try {
                retriever.setDataSource(this@MainActivity, videoUri)
                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                val durationMs = durationStr?.toLongOrNull() ?: 0L
                
                Log.i("DeepfakeAI", "Starting sequential processing for video length: ${durationMs}ms")
                
                var currentTimeMs = 0L
                val intervalMs = 500L // Increased interval to allow inference breathing room
                
                while (currentTimeMs < durationMs && isActive) {
                    val bitmap = retriever.getFrameAtTime(currentTimeMs * 1000, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                    
                    if (bitmap != null) {
                        try {
                            sessionStats.framesProcessed++
                            
                            withContext(Dispatchers.Main) {
                                statusTextView.text = "Analyzing frame at ${currentTimeMs/1000}s..."
                            }

                            // Sequential Wraper: Wait for detection & inference
                            val faceFound = detectAndProcessFaceSafe(bitmap) {
                                Log.d("DeepfakeAI", "Frame processing cancelled safely")
                            }
                            
                            if (faceFound) {
                                withContext(Dispatchers.Main) {
                                    statusTextView.text = "Face analyzed at ${currentTimeMs}ms ✅"
                                }
                            } else {
                                Log.i("FACE_DETECTION", "No face at ${currentTimeMs}ms")
                                withContext(Dispatchers.Main) {
                                    statusTextView.text = "No face at ${currentTimeMs}ms ❌"
                                }
                            }
                        } finally {
                            // Vital: Recycle bitmap to prevent OOM
                            bitmap.recycle()
                        }
                    }
                    
                    currentTimeMs += intervalMs
                    // No need for delay(10), we waited for processing
                }
                
                withContext(Dispatchers.Main) {
                    statusTextView.text = "Video Processing Complete ✅"
                }
                
            } catch (e: CancellationException) {
                Log.i("DeepfakeAI", "Video processing cancelled")
                withContext(Dispatchers.Main) {
                    statusTextView.text = "Processing stopped"
                }
            } catch (e: Exception) {
                Log.e("DeepfakeAI", "Error processing video", e)
                withContext(Dispatchers.Main) {
                    statusTextView.text = "Error processing video: ${e.message}"
                }
            } finally {
                retriever.release()
            }
        }
    }

    // New suspend wrapper for face detection + inference
    private suspend fun detectAndProcessFaceSafe(
        bitmap: Bitmap, 
        onCancellation: (() -> Unit)? = null
    ): Boolean = suspendCancellableCoroutine { continuation ->
        continuation.invokeOnCancellation { 
            onCancellation?.invoke()
        }
        faceDetectorHelper?.detectFace(
            bitmap = bitmap,
            onSuccess = { bounds ->
                if (bounds != null) {
                    sessionStats.facesDetected++
                    try {
                        processingScope.launch(Dispatchers.Default) {
                            preprocessFaceForModel(bitmap, bounds)
                            if (continuation.isActive) continuation.resume(true) {}
                        }
                    } catch (e: Exception) {
                        if (continuation.isActive) continuation.resume(false) {}
                    }
                } else {
                    if (continuation.isActive) continuation.resume(false) {}
                }
            },
            onError = { e ->
                Log.e("FACE_DETECTION", "Error", e)
                if (continuation.isActive) continuation.resume(false) {}
            }
        )
    }

    // Score Smoothing
    private val scoreBuffer = ArrayDeque<Float>()
    private val WINDOW_SIZE = 15

    private fun preprocessFaceForModel(originalBitmap: Bitmap, boundingBox: android.graphics.Rect) {
        try {
            // 1. Clamp bounding box
            val left = boundingBox.left.coerceAtLeast(0)
            val top = boundingBox.top.coerceAtLeast(0)
            val width = boundingBox.width().coerceAtMost(originalBitmap.width - left)
            val height = boundingBox.height().coerceAtMost(originalBitmap.height - top)
            
            // DIAGNOSTIC: Log bbox to detect if face detection is stuck
            Log.d("DIAG_BBOX", "Frame ${sessionStats.framesProcessed}: bbox=[L=$left,T=$top,W=$width,H=$height]")
            
            if (width <= 0 || height <= 0) {
                 Log.w("FACE_PREPROCESS", "Invalid face crop dimensions: $width x $height")
                 return
            }

            // 2. Crop
            val croppedBitmap = android.graphics.Bitmap.createBitmap(originalBitmap, left, top, width, height)
            
            // 3. Resize to 224x224
            val scaledBitmap = android.graphics.Bitmap.createScaledBitmap(croppedBitmap, 224, 224, true)
            
            // UI Verification: Show cropped face
            runOnUiThread {
                val facePreview = findViewById<android.widget.ImageView>(R.id.facePreview)
                facePreview.setImageBitmap(scaledBitmap)
                facePreview.visibility = View.VISIBLE
            }
            
            // 4. Normalize (0-1 float) & Convert to ByteBuffer
            val inputBuffer = java.nio.ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4)
            inputBuffer.order(java.nio.ByteOrder.nativeOrder())
            
            val intValues = IntArray(224 * 224)
            scaledBitmap.getPixels(intValues, 0, 224, 0, 0, 224, 224)
            
            // DIAGNOSTIC: Hash input pixels to detect identical inputs
            val inputHash = intValues.contentHashCode()
            Log.d("DIAG_INPUT", "Frame ${sessionStats.framesProcessed}: pixelHash=$inputHash")
            
            var pixel = 0
            for (i in 0 until 224) {
                for (j in 0 until 224) {
                    val input = intValues[pixel++]
                    inputBuffer.putFloat(((input shr 16 and 0xFF) / 255.0f))
                    inputBuffer.putFloat(((input shr 8 and 0xFF) / 255.0f))
                    inputBuffer.putFloat(((input and 0xFF) / 255.0f))
                }
            }
            
            // CRITICAL: Rewind buffer before inference
            inputBuffer.rewind()
            
            // 5. Run Inference
            val outputBuffer = Array(1) { FloatArray(2) }
            
            try {
                if (tfliteInterpreter == null) {
                    Log.e("MODEL_INFERENCE", "Interpreter is null, skipping frame")
                    return
                }
                tfliteInterpreter?.run(inputBuffer, outputBuffer)
            } catch (e: Exception) {
                Log.e("MODEL_INFERENCE", "Inference failed: ${e.message}")
                return
            }
            
            // DIAGNOSTIC: Log ALL raw output values
            val out0 = outputBuffer[0][0]
            val out1 = outputBuffer[0][1]
            Log.d("DIAG_OUTPUT", "Frame ${sessionStats.framesProcessed}: output[0]=${"%.6f".format(out0)}, output[1]=${"%.6f".format(out1)}, sum=${"%.4f".format(out0 + out1)}")
            
            // Model outputs: try BOTH interpretations
            // Interpretation A: [real, fake] -> fakeProb = out1
            // Interpretation B: [fake, real] -> fakeProb = out0
            val fakeProbA = out1  // Assuming [real, fake]
            val fakeProbB = out0  // Assuming [fake, real]
            
            Log.d("DIAG_INTERPRET", "If [real,fake]: fakeProb=${"%.4f".format(fakeProbA)} | If [fake,real]: fakeProb=${"%.4f".format(fakeProbB)}")
            
            // Use interpretation B (out0 = fake) based on diagnostic results
            // Diagnostic showed ~72% which is likely 'real' prob for real videos
            val fakeProb = fakeProbB  // CHANGED: was fakeProbA
            val rawScore = fakeProb.coerceIn(0f, 1f)
            
            // 6. Use raw score directly (smoothing disabled for debugging)
            val smoothedScore = rawScore
            
            Log.i("SCORE_DEBUG", "Frame ${sessionStats.framesProcessed}: rawScore=${"%.4f".format(rawScore)}")
            
            // 7. Evaluate Risk Level with Hysteresis
            val newRiskLevel = evaluateRiskLevel(smoothedScore)
            
            // 8. Update Session Stats
            sessionStats.inferenceCalls++
            sessionStats.scoreSum += smoothedScore
            if (smoothedScore > sessionStats.peakScore) {
                sessionStats.peakScore = smoothedScore
            }
            when (newRiskLevel) {
                RiskLevel.LOW -> sessionStats.lowRiskCount++
                RiskLevel.SUSPICIOUS -> sessionStats.suspiciousRiskCount++
                RiskLevel.HIGH -> sessionStats.highRiskCount++
            }
            
            Log.i("PREPROCESS_FACE", "Inference #${sessionStats.inferenceCalls} complete")
            
            // 9. Update UI with Risk Level (Color-coded)
            runOnUiThread {
                if (isFinishing || isDestroyed) return@runOnUiThread
                
                // SAFE UI UPDATE
                try {
                    val statusTextView = findViewById<TextView>(R.id.statusTextView)
                    val percentage = (smoothedScore * 100).toInt()
                    
                    // Apply color
                    statusTextView.setTextColor(newRiskLevel.colorHex)
                    
                    // Build message with disclaimer
                    val message = StringBuilder()
                    message.append("${newRiskLevel.displayName}\n")
                    message.append("${newRiskLevel.explanation}\n\n")
                    message.append("Confidence: $percentage%\n")
                    message.append("(This is an automated estimate, not proof.)")
                    
                    statusTextView.text = message.toString()
                    
                    // Update session summary
                    updateSessionSummaryUI()
                    
                    // Update Visual Risk Indicator (ensures internal UI calls are safe)
                    updateRiskVisuals(newRiskLevel, smoothedScore)
                    
                    Log.i("RISK_UI", "Displayed: ${newRiskLevel.displayName} ($percentage%)")
                } catch (e: Exception) {
                    Log.e("RISK_UI", "Error updating UI: ${e.message}")
                }
            }
            
        } catch (e: Exception) {
            Log.e("PREPROCESS_FACE", "Error preprocessing face", e)
        }
    }
    
    private fun updateRiskVisuals(riskLevel: RiskLevel, confidence: Float) {
        val bannerLayout = findViewById<androidx.constraintlayout.widget.ConstraintLayout>(R.id.riskBannerLayout)
        val bannerIcon = findViewById<android.widget.ImageView>(R.id.riskIcon)
        val bannerText = findViewById<TextView>(R.id.riskBannerText)
        
        // Ensure visible
        if (bannerLayout.visibility != View.VISIBLE) {
            bannerLayout.visibility = View.VISIBLE
            bannerLayout.animate().alpha(1f).setDuration(300).start()
        }
        
        // Update Content
        val percentage = (confidence * 100).toInt()
        bannerIcon.setImageResource(riskLevel.iconResId)
        bannerText.text = "${riskLevel.displayName}\nDeepfake likelihood: $percentage%"
        
        // Animate Color Change (Simple substitution for now, could use ValueAnimator for smooth color)
        val color = riskLevel.colorHex
        
        // Semi-transparent background for banner
        // We'll apply the color with some transparency
        val backgroundDrawable = android.graphics.drawable.GradientDrawable()
        backgroundDrawable.setColor(color)
        backgroundDrawable.cornerRadius = 16f // 8dp -> 16px approx
        backgroundDrawable.alpha = 230 // 0-255
        
        bannerLayout.background = backgroundDrawable
        
        // Log only on state change or significant update (handled by caller mostly, but useful here)
    }
    
    private fun updateSessionSummaryUI() {
        val summaryView = findViewById<TextView>(R.id.sessionSummaryTextView)
        val durationSec = sessionStats.sessionDurationMs / 1000
        val avgPct = (sessionStats.averageScore * 100).toInt()
        val peakPct = (sessionStats.peakScore * 100).toInt()
        
        val summary = """
            |📊 SESSION SUMMARY
            |────────────────────
            |⏱ Duration: ${durationSec}s
            |🖼 Frames: ${sessionStats.framesProcessed}
            |👤 Faces: ${sessionStats.facesDetected}
            |🔍 Inferences: ${sessionStats.inferenceCalls}
            |────────────────────
            |🟢 Low: ${sessionStats.lowRiskCount}
            |🟡 Suspicious: ${sessionStats.suspiciousRiskCount}
            |🔴 High: ${sessionStats.highRiskCount}
            |────────────────────
            |Avg: $avgPct% | Peak: $peakPct%
        """.trimMargin()
        
        summaryView.text = summary
        Log.i("SESSION_LOG", "Stats: frames=${sessionStats.framesProcessed} faces=${sessionStats.facesDetected} inferences=${sessionStats.inferenceCalls}")
    }
    
    private fun evaluateRiskLevel(smoothedScore: Float): RiskLevel {
        val previousLevel = currentRiskLevel
        
        // If hysteresis disabled, use simpler logic
        if (!isHysteresisEnabled) {
             val newLevel = when {
                 smoothedScore >= currentSettings.highRiskMin -> RiskLevel.HIGH
                 smoothedScore > currentSettings.lowRiskMax -> RiskLevel.SUSPICIOUS
                 else -> RiskLevel.LOW
             }
             if (newLevel != previousLevel) {
                 Log.i("RISK_AGENT", "STATE CHANGE (NO HYST): $previousLevel -> $newLevel (score=$smoothedScore)")
                 currentRiskLevel = newLevel
             }
             return newLevel
        }

        val newLevel = when (currentRiskLevel) {
            RiskLevel.LOW -> {
                when {
                    smoothedScore >= highRiskEntry -> RiskLevel.HIGH
                    smoothedScore >= lowRiskExit -> RiskLevel.SUSPICIOUS
                    else -> RiskLevel.LOW
                }
            }
            RiskLevel.SUSPICIOUS -> {
                when {
                    smoothedScore >= highRiskEntry -> RiskLevel.HIGH
                    smoothedScore <= lowRiskEntry -> RiskLevel.LOW
                    else -> RiskLevel.SUSPICIOUS
                }
            }
            RiskLevel.HIGH -> {
                when {
                    smoothedScore <= lowRiskEntry -> RiskLevel.LOW
                    smoothedScore <= highRiskExit -> RiskLevel.SUSPICIOUS
                    else -> RiskLevel.HIGH
                }
            }
        }
        
        if (newLevel != previousLevel) {
            Log.i("RISK_AGENT", "STATE CHANGE: $previousLevel -> $newLevel (score=$smoothedScore)")
        }
        
        currentRiskLevel = newLevel
        return newLevel
    }
    
    // Previous fixed WINDOW_SIZE removed, now using smoothingWindowSize var
    private fun calculateSmoothedScore(rawScore: Float): Float {
        synchronized(scoreBuffer) {
            scoreBuffer.addLast(rawScore)
            // Trim to current settings
            while (scoreBuffer.size > smoothingWindowSize) {
                scoreBuffer.removeFirst()
            }
            return scoreBuffer.average().toFloat()
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
