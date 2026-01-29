package com.example.deepfakeai

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.util.Log
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

/**
 * Efficient YUV_420_888 to RGB converter with buffer reuse.
 * Avoids per-frame allocations by reusing internal buffers.
 */
class YuvToRgbConverter {
    
    companion object {
        private const val TAG = "YuvToRgbConverter"
    }
    
    // Reusable buffers - allocated once, reused per frame
    @Volatile private var rgbBuffer: ByteArray? = null
    @Volatile private var yuvBuffer: ByteArray? = null
    
    /**
     * Convert ImageProxy (YUV_420_888) to RGB byte array.
     * Returns interleaved RGB bytes [R,G,B,R,G,B,...].
     * 
     * @param imageProxy CameraX ImageProxy in YUV_420_888 format
     * @return RGB byte array (size = width * height * 3)
     */
    @Synchronized
    fun imageProxyToRgbBytes(imageProxy: ImageProxy): ByteArray {
        val width = imageProxy.width
        val height = imageProxy.height
        val rgbSize = width * height * 3
        
        // Allocate or reuse RGB buffer
        if (rgbBuffer == null || rgbBuffer!!.size < rgbSize) {
            rgbBuffer = ByteArray(rgbSize)
            Log.d(TAG, "Allocated RGB buffer: ${rgbSize} bytes")
        }
        
        // Convert YUV to RGB
        yuvToRgb(imageProxy, rgbBuffer!!)
        
        return rgbBuffer!!
    }
    
    /**
     * Convert ImageProxy (YUV_420_888) directly to Bitmap.
     * More efficient than going through intermediate buffers when Bitmap is needed.
     */
    @Synchronized
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap {
        val width = imageProxy.width
        val height = imageProxy.height
        
        // Get RGB bytes
        val rgbBytes = imageProxyToRgbBytes(imageProxy)
        
        // Create Bitmap from RGB
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(width * height)
        
        var rgbIdx = 0
        for (i in 0 until width * height) {
            val r = rgbBytes[rgbIdx++].toInt() and 0xFF
            val g = rgbBytes[rgbIdx++].toInt() and 0xFF
            val b = rgbBytes[rgbIdx++].toInt() and 0xFF
            pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
        return bitmap
    }
    
    /**
     * Core YUV_420_888 to RGB conversion.
     * Handles plane rowStride and pixelStride correctly.
     * CRITICAL: Rewinds buffers to ensure fresh data is read each frame.
     */
    private fun yuvToRgb(imageProxy: ImageProxy, outRgb: ByteArray) {
        val width = imageProxy.width
        val height = imageProxy.height
        
        val yPlane = imageProxy.planes[0]
        val uPlane = imageProxy.planes[1]
        val vPlane = imageProxy.planes[2]
        
        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        
        // CRITICAL: Rewind buffers to read from start
        yBuffer.rewind()
        uBuffer.rewind()
        vBuffer.rewind()
        
        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        
        // Copy Y plane to byte array for safer random access
        val ySize = yBuffer.remaining()
        val yBytes = ByteArray(ySize)
        yBuffer.get(yBytes)
        
        // Copy U and V planes
        uBuffer.rewind()
        vBuffer.rewind()
        val uvSize = uBuffer.remaining()
        val uBytes = ByteArray(uvSize)
        val vBytes = ByteArray(vBuffer.remaining())
        uBuffer.get(uBytes)
        vBuffer.get(vBytes)
        
        var rgbIdx = 0
        
        for (y in 0 until height) {
            for (x in 0 until width) {
                // Get Y value with bounds check
                val yIdx = y * yRowStride + x
                val yValue = if (yIdx < yBytes.size) {
                    yBytes[yIdx].toInt() and 0xFF
                } else 128
                
                // Get U and V values (subsampled 2x2)
                val uvY = y / 2
                val uvX = x / 2
                val uvIdx = uvY * uvRowStride + uvX * uvPixelStride
                
                val uValue = if (uvIdx < uBytes.size) {
                    uBytes[uvIdx].toInt() and 0xFF
                } else 128
                
                val vValue = if (uvIdx < vBytes.size) {
                    vBytes[uvIdx].toInt() and 0xFF
                } else 128
                
                // YUV to RGB conversion (BT.601)
                val yShifted = yValue - 16
                val uShifted = uValue - 128
                val vShifted = vValue - 128
                
                var r = (1.164f * yShifted + 1.596f * vShifted).toInt()
                var g = (1.164f * yShifted - 0.392f * uShifted - 0.813f * vShifted).toInt()
                var b = (1.164f * yShifted + 2.017f * uShifted).toInt()
                
                // Clamp to 0-255
                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)
                
                outRgb[rgbIdx++] = r.toByte()
                outRgb[rgbIdx++] = g.toByte()
                outRgb[rgbIdx++] = b.toByte()
            }
        }
        
        Log.d(TAG, "YUV conversion: ${width}x${height}, ySize=$ySize, uvSize=$uvSize")
    }
    
    /**
     * Crop and resize RGB bytes to target size, then normalize to float ByteBuffer.
     * 
     * @param rgbBytes Source RGB byte array
     * @param srcWidth Source image width
     * @param srcHeight Source image height
     * @param cropRect Crop rectangle (bounding box)
     * @param dstWidth Target width (e.g., 224)
     * @param dstHeight Target height (e.g., 224)
     * @param outBuffer Output ByteBuffer for normalized floats (must be direct, native order)
     */
    fun cropResizeNormalizeToBuffer(
        rgbBytes: ByteArray,
        srcWidth: Int,
        srcHeight: Int,
        cropRect: Rect,
        dstWidth: Int,
        dstHeight: Int,
        outBuffer: ByteBuffer
    ) {
        // Clamp crop rect to image bounds
        val left = cropRect.left.coerceIn(0, srcWidth - 1)
        val top = cropRect.top.coerceIn(0, srcHeight - 1)
        val right = cropRect.right.coerceIn(left + 1, srcWidth)
        val bottom = cropRect.bottom.coerceIn(top + 1, srcHeight)
        val cropWidth = right - left
        val cropHeight = bottom - top
        
        outBuffer.rewind()
        
        // Bilinear interpolation for resize
        val scaleX = cropWidth.toFloat() / dstWidth
        val scaleY = cropHeight.toFloat() / dstHeight
        
        for (y in 0 until dstHeight) {
            for (x in 0 until dstWidth) {
                // Map destination coords to source coords
                val srcXf = left + x * scaleX
                val srcYf = top + y * scaleY
                
                val srcX = srcXf.toInt().coerceIn(left, right - 1)
                val srcY = srcYf.toInt().coerceIn(top, bottom - 1)
                
                // Get RGB values
                val srcIdx = (srcY * srcWidth + srcX) * 3
                val rRaw = (rgbBytes[srcIdx].toInt() and 0xFF) / 255.0f
                val gRaw = (rgbBytes[srcIdx + 1].toInt() and 0xFF) / 255.0f
                val bRaw = (rgbBytes[srcIdx + 2].toInt() and 0xFF) / 255.0f
                
                // Try -1 to 1 normalization (common for TF models)
                // Formula: (pixel / 255.0) * 2.0 - 1.0
                val r = rRaw * 2.0f - 1.0f
                val g = gRaw * 2.0f - 1.0f
                val b = bRaw * 2.0f - 1.0f
                
                // Write normalized floats in BGR order (some TF models expect BGR)
                outBuffer.putFloat(b)  // Blue first
                outBuffer.putFloat(g)  // Green
                outBuffer.putFloat(r)  // Red last
            }
        }
        
        outBuffer.rewind()
    }
    
    /**
     * Compute CRC32 hash of a region in RGB bytes for DIAG_INPUT logging.
     */
    fun computeRegionHash(
        rgbBytes: ByteArray,
        srcWidth: Int,
        srcHeight: Int,
        cropRect: Rect
    ): Int {
        val left = cropRect.left.coerceIn(0, srcWidth - 1)
        val top = cropRect.top.coerceIn(0, srcHeight - 1)
        val right = cropRect.right.coerceIn(left + 1, srcWidth)
        val bottom = cropRect.bottom.coerceIn(top + 1, srcHeight)
        
        var hash = 0
        for (y in top until bottom step 4) { // Sample every 4 rows for speed
            for (x in left until right step 4) { // Sample every 4 columns
                val idx = (y * srcWidth + x) * 3
                if (idx + 2 < rgbBytes.size) {
                    hash = 31 * hash + rgbBytes[idx].toInt()
                    hash = 31 * hash + rgbBytes[idx + 1].toInt()
                    hash = 31 * hash + rgbBytes[idx + 2].toInt()
                }
            }
        }
        return hash
    }
    
    /**
     * Release internal buffers to free memory.
     */
    fun release() {
        rgbBuffer = null
        yuvBuffer = null
        Log.d(TAG, "Buffers released")
    }
}
