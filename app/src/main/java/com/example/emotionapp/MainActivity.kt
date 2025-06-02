package com.example.emotionapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.widget.Toast
import android.view.ViewGroup
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: Interpreter
    private lateinit var cameraPermissionLauncher: ActivityResultLauncher<String>
    private lateinit var cameraExecutor: ExecutorService
    private val currentScreen = mutableStateOf("home") // "home" or "camera"
    private val emotionLabels = listOf("Happy", "Sad", "Surprised", "Fearful", "Angry", "Disgusted", "Neutral")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        interpreter = Interpreter(loadModelFile())
        try {
            interpreter = Interpreter(loadModelFile())
        } catch (e: Exception) {
            Log.e("EmotionApp", "Failed to load model.tflite: ${e.message}", e)
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_LONG).show()
            finish()
            return
        }
        cameraExecutor = Executors.newSingleThreadExecutor()
        // After `interpreter = Interpreter(loadModelFile())`:

        val inputTensor = interpreter.getInputTensor(0)
        val shape = inputTensor.shape()
        val dataType = inputTensor.dataType()
        Toast.makeText(this, "Model expects input shape: ${shape.joinToString()}, data type: $dataType", Toast.LENGTH_LONG).show()

        Log.i("EmotionApp", "Model expects shape = ${shape.joinToString()} , dataType = $dataType")
        // Temporary toast just to be 100% sure this code is executing:
        cameraPermissionLauncher = registerForActivityResult(
            ActivityResultContracts.RequestPermission()
        ) { granted ->
            setContent { AppContent(granted) }
        }
        // Initialize analytics service
        val hasCameraPermission = ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED

        if (hasCameraPermission) {
            setContent { AppContent(true) }
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    @Composable
    fun AppContent(hasPermission: Boolean) {
        if (!hasPermission) {
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center
            ) {
                Text("Camera permission required")
            }
            return
        }

        when (currentScreen.value) {
            "home" -> HomeScreen()
            "camera" -> EmotionDetectionWithPreview(
                interpreter = interpreter,
                labels = emotionLabels,
                cameraExecutor = cameraExecutor,
                modifier = Modifier.fillMaxSize()
            )
        }
    }
    @Composable
    fun HomeScreen() {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("EmotionApp", style = MaterialTheme.typography.headlineLarge)
                Spacer(modifier = Modifier.height(20.dp))
                Button(onClick = { currentScreen.value = "camera" }) {
                    Text("Start Emotion Detection")
                }
            }
        }
    }


    private fun loadModelFile(): ByteBuffer {
        val fileDescriptor = assets.openFd("model.tflite")
        val inputStream = fileDescriptor.createInputStream()
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        interpreter.close()
        cameraExecutor.shutdown()
    }
}

@OptIn(ExperimentalGetImage::class)
@Composable
fun EmotionDetectionWithPreview(
    interpreter: Interpreter,
    labels: List<String>,
    cameraExecutor: ExecutorService,         // use the one from MainActivity
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val predictionHistory = remember { mutableStateListOf<Int>() }
    val historySize = 5  // You can adjust this for more/less smoothing
    // 1) Remove the “val cameraExecutor = …” here, since you’re receiving it as a parameter.
    //    If you want an executor local to this Composable instead, drop the parameter and
    //    use this block instead:
    //
    // val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    // DisposableEffect(cameraExecutor) { onDispose { cameraExecutor.shutdown() } }

    // 2) MLKit face detector: remember + close on disposal
    val faceDetectorOptions = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .build()
    val faceDetector = remember { FaceDetection.getClient(faceDetectorOptions) }
    DisposableEffect(faceDetector) {
        onDispose { faceDetector.close() }
    }

    var emotionText by remember { mutableStateOf("Detecting...") }
    var previewBitmap by remember { mutableStateOf<Bitmap?>(null) }
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    AndroidView(
        factory = { ctx ->
            val previewView = androidx.camera.view.PreviewView(ctx).apply {
                layoutParams = android.widget.FrameLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
            }

            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                // 3) Analysis at a smaller resolution
                val analysisUseCase = ImageAnalysis.Builder()
                    .setTargetResolution(Size(480, 480))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                val preview = Preview.Builder()
                    .setTargetResolution(Size(1280, 720))
                    .build()

                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                preview.setSurfaceProvider(previewView.surfaceProvider)

                analysisUseCase.setAnalyzer(cameraExecutor) { imageProxy ->
                    val mediaImage = imageProxy.image
                    if (mediaImage == null) {
                        imageProxy.close()
                        return@setAnalyzer
                    }
                    val rotationDegrees = imageProxy.imageInfo.rotationDegrees
                    val inputImage = InputImage.fromMediaImage(mediaImage, rotationDegrees)

                    var closed = false
                    fun safeClose() {
                        if (!closed) {
                            imageProxy.close()
                            closed = true
                        }
                    }

                    faceDetector.process(inputImage)
                        .addOnSuccessListener { faces ->
                            if (faces.isNotEmpty()) {
                                val rawBitmap = imageProxy.toBitmap()
                                if (rawBitmap == null) {
                                    Log.e("EmotionApp", "toBitmap() returned null")
                                    emotionText = "Bitmap error"
                                    safeClose()
                                    return@addOnSuccessListener
                                }
                                val uprightBitmap = rawBitmap.rotate(rotationDegrees.toFloat())

                                val faceRect = faces[0].boundingBox
                                val faceCrop = cropToFace(uprightBitmap, faceRect)
                                if (faceCrop.width == 0 || faceCrop.height == 0) {
                                    Log.e("EmotionApp", "Cropped face has zero size: $faceRect")
                                    emotionText = "Crop error"
                                    safeClose()
                                    return@addOnSuccessListener
                                }

                                // ← Use the new RGB preprocess
                                val inputBuffer = preprocess(faceCrop)
                                val expectedBytes = 1 * 224 * 224 * 3 * 4  // = 49152
                                Log.d("EmotionApp", "ByteBuffer capacity = ${inputBuffer.capacity()} ; expected = $expectedBytes")
                                val output = Array(1) { FloatArray(labels.size) }
                                interpreter.run(inputBuffer, output)

                                // Log the raw probabilities for all 7 classes:
                                Log.d(
                                    "EmotionApp",
                                    "Raw probs: ${output[0].mapIndexed { idx, prob -> "${labels[idx]}:${"%.3f".format(prob)}" }.joinToString(" | ")}"
                                )


                                val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1
                                Log.d("EmotionApp", "Chosen class: $maxIndex -> ${labels[maxIndex]}")

                                if (maxIndex >= 0) {
                                    predictionHistory.add(maxIndex)
                                    if (predictionHistory.size > historySize) {
                                        predictionHistory.removeAt(0)
                                    }

                                    val majority = predictionHistory
                                        .groupingBy { it }
                                        .eachCount()
                                        .maxByOrNull { it.value }
                                        ?.key ?: maxIndex

                                    emotionText = labels[majority]
                                    previewBitmap = faceCrop
                                }


                            } else {
                                emotionText = "No face detected"
                                previewBitmap = null
                            }

                            safeClose()
                        }
                        .addOnFailureListener {
                            it.printStackTrace()
                            emotionText = "Face detection failed"
                            safeClose()
                        }
                }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,
                        preview,
                        analysisUseCase
                    )
                } catch (exc: Exception) {
                    Log.e("EmotionDetection", "Use case binding failed", exc)
                }
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = modifier
    )

    Box(modifier = Modifier.fillMaxSize()) {
        previewBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Face Preview",
                modifier = Modifier.fillMaxSize()
            )
        }
        Text(
            text = emotionText,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(16.dp),
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.primary
        )
    }
}

// (Your existing `toBitmap()`, `cropToFace()`, and `preprocess()` functions can stay the same.)



// Extension function to convert ImageProxy to Bitmap
fun ImageProxy.toBitmap(): Bitmap? {
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

    val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()
    return android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

// Crop bitmap to face bounding box, clamp to bitmap edges
fun cropToFace(bitmap: Bitmap, boundingBox: Rect): Bitmap {
    val padX = (boundingBox.width() * 0.3f).toInt()   // 30% of width
    val padY = (boundingBox.height() * 0.4f).toInt()  // 40% of height

    val x = (boundingBox.left - padX).coerceAtLeast(0)
    val y = (boundingBox.top - padY).coerceAtLeast(0)
    val right = (boundingBox.right + padX).coerceAtMost(bitmap.width)
    val bottom = (boundingBox.bottom + padY).coerceAtMost(bitmap.height)

    val width = right - x
    val height = bottom - y

    return Bitmap.createBitmap(bitmap, x, y, width, height)
}


// Preprocess face bitmap into ByteBuffer input for TFLite model
fun preprocess(bitmap: Bitmap): ByteBuffer {
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

fun Bitmap.rotate(degrees: Float): Bitmap {
    val matrix = Matrix().apply { postRotate(degrees) }
    return Bitmap.createBitmap(this, 0, 0, width, height, matrix, true)
}