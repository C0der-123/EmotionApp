package com.example.emotionapp

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.ViewGroup
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.geometry.Size as USize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.min
import kotlin.math.max
sealed class EmotionResult {
    data class Success(val faceRect: Rect, val probs: FloatArray) : EmotionResult()
    data class Error(val message: String) : EmotionResult()
}
class MainActivity : ComponentActivity() {

    // Interpreter for your TFLite model
    private lateinit var interpreter: Interpreter

    // Emotion labels must match the output order of your TFLite model
    private val emotionLabels = listOf(
        "Happy", "Sad", "Surprised", "Fearful", "Angry", "Disgusted", "Neutral"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 1) Load your TFLite model from assets
        try {
            interpreter = Interpreter(loadModelFile())
        } catch (e: Exception) {
            Log.e("EmotionApp", "Failed to load model.tflite: ${e.message}", e)
            finish()
            return
        }

        // 2) Create a single‐threaded executor for camera analysis
        val cameraExecutor = Executors.newSingleThreadExecutor()

        // 3) Create two Compose states: one for face bounding box, one for emotion label


        val emotionTextState = mutableStateOf("Detecting...")

        setContent {
            AppNavigation()

        }
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
    }


    /**
     * Loads the TFLite model file ("model.tflite") from assets into a ByteBuffer.
     */
    @Throws(IOException::class)
    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("model.tflite")
        FileInputStream(fileDescriptor.fileDescriptor).use { inputStream ->
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }
    @Composable
    fun AppNavigation() {
        val navController = rememberNavController()
        NavHost(navController, startDestination = "home") {
            composable("home") { HomeScreen { navController.navigate("camera") } }
            composable("camera") { CameraScreen() }
        }
    }
    @Composable
    fun HomeScreen(onStartClick: () -> Unit) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(
                    Brush.verticalGradient(
                        colors = listOf(Color(0xFFB3E5FC), Color(0xFF81D4FA)) // light blue gradient
                    )
                ),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Text(
                    text = "EmotionApp",
                    fontSize = 36.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.DarkGray
                )
                Spacer(modifier = Modifier.height(32.dp))
                Button(onClick = onStartClick) {
                    Text(text = "Start Camera")
                }
            }
        }
    }
    @Composable
    fun CameraScreen() {
        val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
        val emotionTextState = remember { mutableStateOf("Detecting...") }
        val faceBoxState = remember { mutableStateOf<Rect?>(null) }

        val analyzer = remember {
            EmotionAnalyzer(interpreter, emotionLabels) { result ->
                when (result) {
                    is EmotionResult.Success -> {
                        faceBoxState.value = result.faceRect
                        val probs = result.probs
                        val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
                        emotionTextState.value = emotionLabels[maxIdx]
                    }
                    is EmotionResult.Error -> {
                        emotionTextState.value = result.message
                    }
                }
            }
        }

        Column(modifier = Modifier.fillMaxSize()) {
            // 1) Emotion text at the top, with padding so it sits below any notch/status bar
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.Black.copy(alpha = 0.5f))
                    .padding(top = 48.dp, bottom = 8.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = emotionTextState.value,
                    color = Color.White,
                    style = MaterialTheme.typography.headlineSmall
                )
            }

            // 2) Camera preview + overlay below the text
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
            ) {
                CameraPreviewWithAnalyzer(
                    cameraExecutor = cameraExecutor,
                    analyzer = analyzer,
                    modifier = Modifier.fillMaxSize()
                )

                // Overlay: map 480×480 face‐box into the PreviewView’s letterboxed size
                BoxWithConstraints {
                    val parentWidth = constraints.maxWidth.toFloat()
                    val parentHeight = constraints.maxHeight.toFloat()

                    faceBoxState.value?.let { box ->
                        Canvas(modifier = Modifier.fillMaxSize()) {
                            val imageWidth = 480f
                            val imageHeight = 480f
                            val scale = min(parentWidth / imageWidth, parentHeight / imageHeight)
                            val offsetX = (parentWidth - imageWidth * scale) / 2f
                            val offsetY = (parentHeight - imageHeight * scale) / 2f

                            val padX = (box.width() * 0.2f).toInt()
                            val padY = (box.height() * 0.2f).toInt()

                            // 2) Make a new Rect that’s expanded by padX/padY (clamp to [0,480])
                            val expandedLeft   = (box.left   - padX).coerceAtLeast(0)
                            val expandedTop    = (box.top    - padY).coerceAtLeast(0)
                            val expandedRight  = (box.right  + padX).coerceAtMost(imageWidth.toInt())
                            val expandedBottom = (box.bottom + padY).coerceAtMost(imageHeight.toInt())
                            val paddedBox = Rect(expandedLeft, expandedTop, expandedRight, expandedBottom)

                            // 3) Mirror and scale exactly as before, but use `paddedBox` instead of `box`
                            val mirroredLeft   = imageWidth - paddedBox.right
                            val mirroredRight  = imageWidth - paddedBox.left

                            val left   = offsetX + mirroredLeft  * scale
                            val top    = offsetY + paddedBox.top    * scale
                            val right  = offsetX + mirroredRight * scale
                            val bottom = offsetY + paddedBox.bottom * scale

                            drawRect(
                                color = Color.Red,
                                topLeft = Offset(left, top),
                                size = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                                style = Stroke(width = 4.dp.toPx())
                            )
                        }
                    }
                }
            }
        }
    }
    /**
     * Analyzer that:
     * 1) Runs ML Kit face detection on each frame,
     * 2) Crops + preprocesses the face region,
     * 3) Runs TFLite inference to get an emotion label,
     * 4) Passes the face bounding box (image coords) and emotion label back via onResult().
     */
    private class EmotionAnalyzer(
        private val interpreter: Interpreter,
        private val labels: List<String>,
        private val onResult: (EmotionResult) -> Unit
    ) : ImageAnalysis.Analyzer {


        // ML Kit face detector (fast mode)
        private val faceDetector by lazy {
            FaceDetection.getClient(
                FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                    .build()
            )
        }

        // Keep a short history of predictions for a majority vote (smoothing)
        private val predictionHistory = mutableListOf<Int>()
        private val historySize = 5

        override fun analyze(imageProxy: ImageProxy) {
            val mediaImage = imageProxy.image
            if (mediaImage == null) {
                imageProxy.close()
                return
            }

            val rotationDegrees = imageProxy.imageInfo.rotationDegrees
            val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)

            // 1) Run ML Kit face detection
            faceDetector.process(inputImage)
                .addOnSuccessListener { faces ->
                    if (faces.isNotEmpty()) {
                        val face = faces[0]
                        val boundingBox = face.boundingBox

                        // 2) Convert ImageProxy → Bitmap and rotate
                        val rawBitmap = imageProxy.toBitmap()?.rotate(rotationDegrees.toFloat())
                        if (rawBitmap != null) {
                            // 3) Crop to face and preprocess for TFLite
                            val faceCrop = cropToFace(rawBitmap, boundingBox)
                            if (faceCrop.width > 0 && faceCrop.height > 0) {
                                val inputBuffer = preprocess(faceCrop)

                                // 4) Run TFLite interpreter
                                val output = Array(1) { FloatArray(labels.size) }
                                interpreter.run(inputBuffer, output)
                                val rawProbs = output[0].copyOf()
                                labels.forEachIndexed { idx, label ->
                                    Log.d(
                                        "EmotionApp",
                                        "Probability → $label: ${"%.3f".format(rawProbs[idx])}"
                                    )
                                }
                                // 5) Find highest‐confidence index
                                val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1
                                if (maxIndex >= 0) {

                                    onResult(EmotionResult.Success(boundingBox, predictionHistory
                                        .also {
                                            it.add(maxIndex)
                                            if (it.size > historySize) it.removeAt(0)
                                        }.groupingBy { it }
                                        .eachCount()
                                        .maxByOrNull { it.value }
                                        ?.key
                                        ?.let { idx ->
                                            FloatArray(labels.size) { i -> if (i == idx) 1f else 0f }
                                        } ?: output[0]
                                    ))

                                } else {
                                    onResult(EmotionResult.Error("Unknown"))

                                }
                            } else {
                                onResult(EmotionResult.Error("No face"))

                            }
                        } else {
                            onResult(EmotionResult.Error("Bitmap conversion failed"))

                        }
                    } else {
                        onResult(EmotionResult.Error("No face crop"))

                    }
                    imageProxy.close()
                }
                .addOnFailureListener {
                    onResult(EmotionResult.Error("Detection failed"))

                    imageProxy.close()
                }
        }

        // Extension on ImageProxy to get a Bitmap
        private fun ImageProxy.toBitmap(): Bitmap? {
            val yBuffer = planes[0].buffer // Y
            val uBuffer = planes[1].buffer // U
            val vBuffer = planes[2].buffer // V

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
            val imageBytes = out.toByteArray()
            return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
        }

        // Crop a bitmap to the given face bounding box (with padding)
        private fun cropToFace(bitmap: Bitmap, rect: Rect): Bitmap {
            val padding = (rect.width() * 0.2).toInt()  // Add ~20% around
            val left = max(0, rect.left - padding)
            val top = max(0, rect.top - padding)
            val right = min(bitmap.width, rect.right + padding)
            val bottom = min(bitmap.height, rect.bottom + padding)

            return Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
        }

        // Preprocess the face crop into a ByteBuffer for TFLite (224×224 RGB float)
        private fun preprocess(bitmap: Bitmap): ByteBuffer {
            val inputSize = 224
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

            val byteBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3 * 4)
                .order(ByteOrder.nativeOrder())
            byteBuffer.rewind()

            for (y in 0 until inputSize) {
                for (x in 0 until inputSize) {
                    val pixel = scaledBitmap.getPixel(x, y)
                    val r = ((pixel shr 16) and 0xFF) / 255f
                    val g = ((pixel shr 8) and 0xFF) / 255f
                    val b = (pixel and 0xFF) / 255f
                    byteBuffer.putFloat(r)
                    byteBuffer.putFloat(g)
                    byteBuffer.putFloat(b)
                }
            }
            byteBuffer.rewind()
            return byteBuffer
        }

        // Rotate a Bitmap by the given degrees
        private fun Bitmap.rotate(degrees: Float): Bitmap {
            val matrix = Matrix().apply { postRotate(degrees) }
            return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
        }
    }

    /**
     * A Composable that binds a PreviewView (camera feed) and an ImageAnalysis use‐case
     * to the lifecycle. The provided analyzer runs on each camera frame.
     */
    @Composable
    private fun CameraPreviewWithAnalyzer(
        cameraExecutor: ExecutorService,
        analyzer: ImageAnalysis.Analyzer,
        modifier: Modifier = Modifier
    ) {
        val context = LocalContext.current
        val lifecycleOwner = LocalLifecycleOwner.current

        DisposableEffect(cameraExecutor) {
            onDispose { cameraExecutor.shutdown() }
        }

        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FIT_CENTER
                }

                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val resolutionSelector = ResolutionSelector.Builder()
                        .setAspectRatioStrategy(
                            AspectRatioStrategy(
                                androidx.camera.core.AspectRatio.RATIO_16_9,
                                AspectRatioStrategy.FALLBACK_RULE_AUTO
                            )
                        ).build()

                    val previewUseCase = Preview.Builder()
                        .setResolutionSelector(resolutionSelector)
                        .build()
                        .also {
                            it.setSurfaceProvider(previewView.surfaceProvider)
                        }

                    val analysisUseCase = ImageAnalysis.Builder()
                        .setTargetResolution(Size(480, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor, analyzer)
                        }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner as LifecycleOwner,
                            CameraSelector.DEFAULT_FRONT_CAMERA,
                            previewUseCase,
                            analysisUseCase
                        )
                    } catch (exc: Exception) {
                        Log.e("CameraPreview", "Use case binding failed", exc)
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = modifier
        )
    }
}
