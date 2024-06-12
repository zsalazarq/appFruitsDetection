package com.appfruitscameradetection

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.appfruitscameradetection.ml.Model
import com.google.android.material.snackbar.Snackbar
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.FloatBuffer
import java.util.Arrays
import kotlin.math.max

class MainActivity : AppCompatActivity() {
    private lateinit var selectBtn: Button
    private lateinit var predictBtn: Button
    private lateinit var captureBtn: Button
    private lateinit var result: TextView
    private lateinit var imageView: ImageView
    private lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        getPermission()

        selectBtn = findViewById(R.id.selectBtn)
        predictBtn = findViewById(R.id.predictBtn)
        captureBtn = findViewById(R.id.captureBtn)
        result = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)

        selectBtn.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK)
            intent.type = "image/*"
            startForResult.launch(intent)
        }

        captureBtn.setOnClickListener {
            openCamera()
        }

        predictBtn.setOnClickListener {
            if (::bitmap.isInitialized) { // Verifica si bitmap ha sido inicializado
                val model = Model.newInstance(this)

                // Redimensiona el bitmap a 128x128 (ajustar según sea necesario)
                val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 128, 128, true)

                // Normaliza el bitmap (ajustar según el preprocesamiento esperado por el modelo)
                val tensorImage = TensorImage(DataType.FLOAT32)
                tensorImage.load(resizedBitmap)
                val byteBuffer = tensorImage.buffer

                // Prepara el TensorBuffer con las dimensiones correctas
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 128, 128, 3), DataType.FLOAT32)
                inputFeature0.loadBuffer(byteBuffer)

                // Ejecuta la inferencia del modelo y obtiene el resultado
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                // Redondea el resultado a 4 decimales
                val resultValue = "%.20f".format(outputFeature0.floatArray[0]).toFloat()
                Log.d("MainActivity", "La pro: ${resultValue}")


                // Muestra el resultado
                val resultString = if (resultValue < 5.0E-20f) "Es apto para el consumo" else "No se recomienda para el consumo"
                result.setText(resultString)
                Snackbar.make(it, resultString, Snackbar.LENGTH_SHORT)
                    .show()

                // Libera los recursos del modelo si ya no se usan
                model.close()
            } else {
                Log.d("MainActivity", "La propiedad bitmap no ha sido inicializada.")
            }
        }




    }

    private fun getPermission() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED ||
            checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE), 1)
        }
    }

    private fun openCamera() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(imageBitmap)
            bitmap = imageBitmap // Asignar el valor de imageBitmap a bitmap
        } else if (resultCode == RESULT_OK && requestCode == 2) { //requestCode 2 por ejemplo
            val selectedImageUri: Uri? = data?.data
            imageView.setImageURI(selectedImageUri)

            // Convertir la URI en un Bitmap
            selectedImageUri?.let { uri ->
                contentResolver.openInputStream(uri)?.use { inputStream ->
                    bitmap = BitmapFactory.decodeStream(inputStream)
                }
            }
        }
    }

    companion object {
        private const val REQUEST_IMAGE_CAPTURE = 1
    }

    // Register the launcher and define the result handler
    private val startForResult = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result: ActivityResult ->
        if (result.resultCode == Activity.RESULT_OK) {
            val data: Intent? = result.data
            val selectedImageUri: Uri? = data?.data
            imageView.setImageURI(selectedImageUri)

            // Convertir la URI en un Bitmap
            selectedImageUri?.let { uri ->
                contentResolver.openInputStream(uri)?.use { inputStream ->
                    bitmap = BitmapFactory.decodeStream(inputStream)
                }
            }
        }
    }


}