func @gemver(%arg0: memref<4x4xf32>, %arg1: f32, %arg2: f32, %arg3: memref<4xf32>, %arg4: memref<4xf32>, %arg5: memref<4xf32>, %arg6: memref<4xf32>, %arg7: memref<4xf32>, %arg8: memref<4xf32>, %arg9: memref<4xf32>, %arg10: memref<4xf32>) {
    affine.for %arg11 = 0 to 4 {
      affine.for %arg12 = 0 to 4 {
        %0 = affine.load %arg0[%arg11, %arg12] : memref<4x4xf32>
        %1 = affine.load %arg3[%arg11] : memref<4xf32>
        %2 = affine.load %arg5[%arg12] : memref<4xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        %5 = affine.load %arg4[%arg11] : memref<4xf32>
        %6 = affine.load %arg6[%arg12] : memref<4xf32>
        %7 = mulf %5, %6 : f32
        %8 = addf %4, %7 : f32
        affine.store %8, %arg0[%arg11, %arg12] : memref<4x4xf32>
      }
    }
    affine.for %arg11 = 0 to 4 {
      affine.for %arg12 = 0 to 4 {
        %0 = affine.load %arg8[%arg11] : memref<4xf32>
        %1 = affine.load %arg0[%arg12, %arg11] : memref<4x4xf32>
        %2 = mulf %arg2, %1 : f32
        %3 = affine.load %arg9[%arg12] : memref<4xf32>
        %4 = mulf %2, %3 : f32
        %5 = addf %0, %4 : f32
        affine.store %5, %arg8[%arg11] : memref<4xf32>
      }
    }
    affine.for %arg11 = 0 to 4 {
      %0 = affine.load %arg8[%arg11] : memref<4xf32>
      %1 = affine.load %arg10[%arg11] : memref<4xf32>
      %2 = addf %0, %1 : f32
      affine.store %2, %arg8[%arg11] : memref<4xf32>
    }
    affine.for %arg11 = 0 to 4 {
      affine.for %arg12 = 0 to 4 {
        %0 = affine.load %arg7[%arg11] : memref<4xf32>
        %1 = affine.load %arg0[%arg11, %arg12] : memref<4x4xf32>
        %2 = mulf %arg1, %1 : f32
        %3 = affine.load %arg8[%arg12] : memref<4xf32>
        %4 = mulf %2, %3 : f32
        %5 = addf %0, %4 : f32
        affine.store %5, %arg7[%arg11] : memref<4xf32>
      }
    }
    return
  }
