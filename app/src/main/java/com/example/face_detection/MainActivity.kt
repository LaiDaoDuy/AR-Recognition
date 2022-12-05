package com.example.face_detection

import android.annotation.SuppressLint
import android.graphics.*
import android.graphics.drawable.ShapeDrawable
import android.graphics.drawable.shapes.RectShape
import android.media.Image
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.FrameLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import com.example.face_detection.tflite.SimilarityClassifier
import com.example.face_detection.tflite.TFLiteObjectDetectionAPIModel
import com.google.ar.core.ArImage
import com.google.ar.core.AugmentedFace
import com.google.ar.core.TrackingState
import com.google.ar.sceneform.ArSceneView
import com.google.ar.sceneform.Sceneform
import com.google.ar.sceneform.rendering.ModelRenderable
import com.google.ar.sceneform.rendering.Renderable
import com.google.ar.sceneform.rendering.Texture
import com.google.ar.sceneform.ux.ArFrontFacingFragment
import com.google.ar.sceneform.ux.AugmentedFaceNode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.CompletableFuture


class MainActivity : AppCompatActivity() {
    private val loaders: MutableSet<CompletableFuture<*>> = HashSet()

    private var arFragment: ArFrontFacingFragment? = null
    private var arSceneView: ArSceneView? = null

    private var faceTexture: Texture? = null
    private var faceModel: ModelRenderable? = null

    private val facesNodes: HashMap<AugmentedFace, AugmentedFaceNode> = HashMap()

    private var faceDetector: FaceDetector? = null
    private var faceBmp: Bitmap? = null
    private var detector: SimilarityClassifier? = null

    private val labelDefault: String = "Mr.Quy"
    private val bitmapSize = 112
    private val quantized = false
    private val modelFile = "mobile_face_net.tflite"
    private val labelsFile = "file:///android_asset/label_map.txt"

    private val textures = "textures/freckles.png"
    private val model3D = "models/fox.glb"

    private var arView: FrameLayout? = null
    private var btnTraining: Button? = null
    private var btnRecognize: Button? = null
    private var txtLabel: TextView? = null
    private var frameLayout: FrameLayout? = null
    private var shapeDrawable: ShapeDrawable? = null
    private var shapeView: View? = null

    private var isTraining = false
    private var isRecognize = false


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        arView = findViewById(R.id.arFragment)
        btnTraining = findViewById(R.id.btnTraining)
        btnRecognize = findViewById(R.id.btnRecognize)
        txtLabel = findViewById(R.id.txtLabel)
        frameLayout = findViewById(R.id.frameLayout)

        txtLabel?.setTextColor(Color.GREEN)
        shapeDrawable = ShapeDrawable(RectShape())
        shapeDrawable?.paint?.color = Color.BLUE
        shapeDrawable?.paint?.style = Paint.Style.STROKE
        shapeDrawable?.paint?.strokeWidth = 2f
        shapeView = View(this)
        shapeView?.background = shapeDrawable

        supportFragmentManager.addFragmentOnAttachListener(this::onAttachFragment)

        if (savedInstanceState == null) {
            if (Sceneform.isSupported(this)) {
                supportFragmentManager.beginTransaction()
                    .add(R.id.arFragment, ArFrontFacingFragment::class.java, null)
                    .commit()
            }
        }
        loadModels()
        loadTextures()

        faceBmp = Bitmap.createBitmap(
            bitmapSize,
            bitmapSize,
            Bitmap.Config.ARGB_8888
        )
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
            .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
        faceDetector = FaceDetection.getClient(options)

        try {
            detector = TFLiteObjectDetectionAPIModel.create(
                assets,
                modelFile,
                labelsFile,
                bitmapSize,
                quantized
            )
        } catch (e: IOException) {
            e.printStackTrace()
            val toast = Toast.makeText(
                applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
            )
            toast.show()
            finish()
        }
    }

    private fun onAttachFragment(fragmentManager: FragmentManager, fragment: Fragment) {
        if (fragment.id == R.id.arFragment) {
            arFragment = fragment as ArFrontFacingFragment
            arFragment!!.setOnViewCreatedListener { arSceneView: ArSceneView ->
                onViewCreated(
                    arSceneView
                )
            }
        }
    }

    @SuppressLint("SetTextI18n")
    private fun onViewCreated(arSceneView: ArSceneView) {
        arSceneView.setCameraStreamRenderPriority(Renderable.RENDER_PRIORITY_FIRST)
        this.arSceneView = arSceneView
        arFragment!!.setOnAugmentedFaceUpdateListener { augmentedFace: AugmentedFace ->
            onAugmentedFaceTrackingUpdate(
                augmentedFace
            )
        }

        btnTraining!!.setOnClickListener {
            isTraining = !isTraining
            if (isTraining) {
                btnTraining!!.text = "End training"
            } else {
                btnTraining!!.text = "Start training"
                frameLayout?.removeAllViews()
            }
        }

        btnRecognize!!.setOnClickListener {
            isRecognize = !isRecognize
            if (isRecognize) {
                btnRecognize!!.text = "End recognize"
                txtLabel?.visibility = View.VISIBLE
            } else {
                btnRecognize!!.text = "Start recognize"
                txtLabel?.visibility = View.GONE
                frameLayout?.removeAllViews()
            }

        }
    }

    override fun onDestroy() {
        super.onDestroy()
        for (loader in loaders) {
            if (!loader.isDone) {
                loader.cancel(true)
            }
        }
    }

    private fun loadModels() {
        loaders.add(ModelRenderable.builder()
            .setSource(this, Uri.parse(model3D))
            .setIsFilamentGltf(true)
            .build()
            .thenAccept { model: ModelRenderable? ->
                faceModel = model
            }
            .exceptionally {
                Toast.makeText(this, "Unable to load render able", Toast.LENGTH_LONG).show()
                null
            }
        )
    }

    private fun loadTextures() {
        loaders.add(Texture.builder()
            .setSource(this, Uri.parse(textures))
            .setUsage(Texture.Usage.COLOR_MAP)
            .build()
            .thenAccept { texture -> faceTexture = texture }
            .exceptionally {
                Toast.makeText(this, "Unable to load texture", Toast.LENGTH_LONG).show()
                null
            })
    }

    @SuppressLint("SetTextI18n")
    private fun onAugmentedFaceTrackingUpdate(augmentedFace: AugmentedFace) {
        if (faceModel == null || faceTexture == null) {
            return
        }
        try {
            val cameraImage: ArImage = arSceneView?.arFrame?.acquireCameraImage() as ArImage
            val imageWidth: Int = cameraImage.width
            val imageHeight: Int = cameraImage.height

            val yPlane: Image.Plane = cameraImage.planes[0]
            val uPlane: Image.Plane = cameraImage.planes[1]
            val vPlane: Image.Plane = cameraImage.planes[2]

            val yStride = yPlane.rowStride
            val uStride = uPlane.rowStride
            val vStride = vPlane.rowStride

            val strides: IntArray = intArrayOf(yStride, uStride, vStride)

            val yBuffer: ByteBuffer = yPlane.buffer
            val uBuffer: ByteBuffer = uPlane.buffer
            val vBuffer: ByteBuffer = vPlane.buffer

            cameraImage.close()

            val ySize: Int = yBuffer.remaining()
            val uSize: Int = uBuffer.remaining()
            val vSize: Int = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)

            val yuvImage = YuvImage(nv21, ImageFormat.YUY2, imageWidth, imageHeight, strides)

            val out = ByteArrayOutputStream()
            yuvImage.compressToJpeg(Rect(0, 0, imageWidth, imageHeight), 100, out)

            val imageBytes: ByteArray = out.toByteArray()

            val bitmapImage: Bitmap = BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)

            val rotationMatrix = Matrix()
            rotationMatrix.postRotate(270f)

            val rotatedImage: Bitmap = Bitmap.createBitmap(
                bitmapImage,
                0,
                0,
                bitmapImage.width,
                bitmapImage.height,
                rotationMatrix,
                true
            )
            val inputImage: InputImage = InputImage.fromBitmap(rotatedImage, 0)

            faceDetector!!
                .process(inputImage)
                .addOnSuccessListener { faces: List<Face> ->
                    val existingFaceNode : AugmentedFaceNode? = facesNodes[augmentedFace]
                    if (isTraining) {
                        onFacesDetected(faces, imageWidth, imageHeight)
                    }
                    if (isRecognize) {
                        recognizeFace(faces).apply {
                            if(this) {
                                viewARModel(augmentedFace, existingFaceNode)
                                txtLabel?.text = labelDefault
                            } else {
                                txtLabel?.text = "Null"
                            }
                        }
                    } else {
                        facesNodes.remove(augmentedFace)
                        if(existingFaceNode != null) {
                            arSceneView!!.scene.removeChild(existingFaceNode)
                        }
                    }
                }.addOnFailureListener { e: Any? ->
                    log(e)
                }
        } catch (e: Exception) {
            log("Exception: $e")
        }
    }

    private fun viewARModel(augmentedFace: AugmentedFace, existingFaceNode : AugmentedFaceNode?) {
        when (augmentedFace.trackingState) {
            TrackingState.TRACKING -> if (existingFaceNode == null) {
                val faceNode = AugmentedFaceNode(augmentedFace)
                val modelInstance = faceNode.setFaceRegionsRenderable(faceModel)
                modelInstance.isShadowCaster = false
                modelInstance.isShadowReceiver = true
                faceNode.faceMeshTexture = faceTexture
                arSceneView!!.scene.addChild(faceNode)
                facesNodes[augmentedFace] = faceNode
            }
            TrackingState.STOPPED -> {
                if (existingFaceNode != null) {
                    arSceneView!!.scene.removeChild(existingFaceNode)
                }
                facesNodes.remove(augmentedFace)
            }
            else -> {}
        }
    }

    private fun log(message: Any?) {
        Log.d("===== ", message.toString())
    }

    private fun recognizeFace(faces: List<Face>): Boolean {
        for (face in faces) {
            val resultsAux: List<SimilarityClassifier.Recognition> =
                detector!!.recognizeImage(faceBmp, true)
            if (resultsAux.isNotEmpty()) {
                for(result in resultsAux) {
                    if (result.title == labelDefault) {
                        return true
                    }
                }
            }
        }
        return false
    }

    private fun onFacesDetected(faces: List<Face>, w : Int, h : Int) {
        val params = FrameLayout.LayoutParams(w, h)
        for (face in faces) {
            // Draw bounding box
            val boundingBox = RectF(face.boundingBox)
            val left = boundingBox.left.toInt()
            val top = boundingBox.top.toInt()
            params.setMargins(left, top, 0, 0)
            frameLayout?.addView(shapeView, params)

            var confidence = -1f
            var color = Color.BLUE
            var extra: Any? = null
            val resultsAux: List<SimilarityClassifier.Recognition> =
                detector!!.recognizeImage(faceBmp, true)
            if (resultsAux.isNotEmpty()) {
                val result: SimilarityClassifier.Recognition = resultsAux[0]
                extra = result.extra
                val conf: Float = result.distance
                if (conf < 1.0f) {
                    confidence = conf
                    color = if (result.id.equals("0")) {
                        Color.GREEN
                    } else {
                        Color.RED
                    }
                }
            }
            val result = SimilarityClassifier.Recognition(
                "0", labelDefault, confidence, boundingBox
            )
            result.color = color
            result.location = boundingBox
            result.extra = extra
            detector?.register(labelDefault, result)
        }
    }
}
