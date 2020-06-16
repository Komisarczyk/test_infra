  func @mm(%arg0: memref<1000x1000xf32>, %arg1: memref<1000x1000xf32>, %arg2: memref<1000x1000xf32>, %arg3: f32, %arg4: f32) {
    affine.for %arg5 = 0 to 1000 {
      affine.for %arg6 = 0 to 1000 {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<1000x1000xf32>
        %1 = mulf %arg4, %0 : f32
        affine.store %1, %arg2[%arg5, %arg6] : memref<1000x1000xf32>
      }
      affine.for %arg6 = 0 to 1000 {
        affine.for %arg7 = 0 to 1000 {
          %0 = affine.load %arg0[%arg5, %arg6] : memref<1000x1000xf32>
          %1 = mulf %arg3, %0 : f32
          %2 = affine.load %arg1[%arg6, %arg7] : memref<1000x1000xf32>
          %3 = mulf %1, %2 : f32
          %4 = affine.load %arg2[%arg5, %arg7] : memref<1000x1000xf32>
          %5 = addf %3, %4 : f32
          affine.store %5, %arg2[%arg5, %arg7] : memref<1000x1000xf32>
        }
      }
    }
    return
 }





