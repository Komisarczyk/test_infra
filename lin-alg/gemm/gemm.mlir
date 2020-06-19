func @mm(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>, %arg2: memref<4x4xf32>, %arg3: f32, %arg4: f32) {
    affine.for %arg5 = 0 to 4 {
      affine.for %arg6 = 0 to 4 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<4x4xf32>
        %1 = mulf %arg4, %0 : f32
        affine.store %1, %arg2[%arg5, %arg6] : memref<4x4xf32>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to 4 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<4x4xf32>
          %1 = mulf %arg3, %0 : f32
          %2 = affine.load %arg1[%arg6, %arg7] : memref<4x4xf32>
          %3 = mulf %1, %2 : f32
          %4 = affine.load %arg2[%arg5, %arg7] : memref<4x4xf32>
          %5 = addf %3, %4 : f32
          affine.store %5, %arg2[%arg5, %arg7] : memref<4x4xf32>
        }
      }
    }
    return
 }
