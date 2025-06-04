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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size as USize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
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
            val inputSize    = remember { mutableStateOf(android.util.Size(480, 480)) }
            val faceBoxState = remember { mutableStateOf<Rect?>(null) }
            Box(modifier = Modifier.fillMaxSize()) {
                val analyzer = remember {
                    EmotionAnalyzer(interpreter, emotionLabels) { result ->
                        when (result) {
                            is EmotionResult.Success -> {
                                faceBoxState.value = result.faceRect

                                val probs = result.probs
                                val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
                                val prediction = emotionLabels[maxIdx]

                                emotionTextState.value = prediction // ✅ UPDATE HERE

                                Log.d(
                                    "EmotionApp",
                                    "Predicted=$prediction  " +
                                            probs.mapIndexed { i, p -> "${emotionLabels[i]}=${"%.2f".format(p)}" }
                                                .joinToString(", ")
                                )
                            }

                            is EmotionResult.Error -> {
                                emotionTextState.value = result.message // ✅ show errors on screen too if you want
                                Log.d("EmotionApp", "Error: ${result.message}")
                            }
                        }

                    }
                }



                CameraPreviewWithAnalyzer(
                    cameraExecutor = cameraExecutor,
                    analyzer = analyzer,
                    modifier = Modifier.fillMaxSize()
                )

                // 5) Overlay bounding box on top of the PreviewView
                BoxWithConstraints {
                    val parentWidth  = constraints.maxWidth.toFloat()
                    val parentHeight = constraints.maxHeight.toFloat()

                    faceBoxState.value?.let { box ->
                        Canvas(modifier = Modifier.fillMaxSize()) {
                            val inW = inputSize.value.width.toFloat()
                            val inH = inputSize.value.height.toFloat()
                            val scale   = if (parentWidth / inW < parentHeight / inH)
                                (parentWidth / inW)
                            else
                                (parentHeight / inH)
                            val offsetX = (parentWidth  - inW * scale) / 2f
                            val offsetY = (parentHeight - inH * scale) / 2f

                            val mirroredLeft  = (inW - box.right)
                            val mirroredRight = (inW - box.left)

                            val left   = offsetX + mirroredLeft  * scale
                            val top    = offsetY + box.top       * scale
                            val right  = offsetX + mirroredRight * scale
                            val bottom = offsetY + box.bottom    * scale

                            drawRect(
                                color   = Color.Red,
                                topLeft = Offset(left, top),
                                size    = androidx.compose.ui.geometry.Size(right - left, bottom - top),
                                style   = Stroke(width = 4.dp.toPx())
                            )
                        }
                    }
                }

                // 6) Overlay emotion label at top center
                Box(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .background(Color.Black.copy(alpha = 0.5f))
                        .padding(8.dp)
                ) {
                    Text(
                        text = emotionTextState.value,
                        color = Color.White,
                        style = MaterialTheme.typography.headlineSmall
                    )
                }
            }
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
        private val historySize = 10

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

                                // 5) Find highest‐confidence index
                                val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1
                                if (maxIndex >= 0) {
                                    // Majority vote smoothing
                                    predictionHistory.add(maxIndex)
                                    if (predictionHistory.size > historySize) {
                                        predictionHistory.removeAt(0)
                                    }
                                    val majority = predictionHistory
                                        .groupingBy { it }
                                        .eachCount()
                                        .maxByOrNull { it.value }
                                        ?.key ?: maxIndex

                                    onResult(EmotionResult.Success(boundingBox, output[0].copyOf()))
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
        private fun cropToFace(bitmap: Bitmap, boundingBox: Rect): Bitmap {
            val padX = (boundingBox.width() * 0.3f).toInt()   // 25% padding horizontally
            val padY = (boundingBox.height() * 0.5f).toInt()  // 40% padding vertically

            val x = (boundingBox.left - padX).coerceAtLeast(0)
            val y = (boundingBox.top - padY).coerceAtLeast(0)
            val right = (boundingBox.right + padX).coerceAtMost(bitmap.width)
            val bottom = (boundingBox.bottom + padY).coerceAtMost(bitmap.height)

            val width = right - x
            val height = bottom - y
            return Bitmap.createBitmap(bitmap, x, y, width, height)
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

        // Shut down executor when this composable is removed
        DisposableEffect(cameraExecutor) {
            onDispose { cameraExecutor.shutdown() }
        }

        AndroidView(
            factory = { ctx ->
                // Create the PreviewView
                val previewView = PreviewView(ctx).apply {
                    layoutParams = FrameLayout.LayoutParams(
                        ViewGroup.LayoutParams.MATCH_PARENT,
                        ViewGroup.LayoutParams.MATCH_PARENT
                    )
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }

                // Get CameraProvider
                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    // Build Preview use‐case (16:9)
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

                    // Build ImageAnalysis use‐case (480×480)
                    val analysisUseCase = ImageAnalysis.Builder()
                        .setTargetResolution(Size(480, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()
                        .also {
                            it.setAnalyzer(cameraExecutor, analyzer)
                        }

                    // Bind both to the lifecycle (front camera)
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
