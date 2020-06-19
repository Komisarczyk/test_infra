func @mvt(%arg0: memref<12x12xf32>, %arg1: memref<12xf32>, %arg2: memref<12xf32>, %arg3: memref<12xf32>, %arg4: memref<12xf32>) {
    affine.for %arg5 = 0 to 12 {
      affine.for %arg6 = 0 to 12 {
        %0 = affine.load %arg1[%arg5] : memref<12xf32>
        %1 = affine.load %arg0[%arg5, %arg6] : memref<12x12xf32>
        %2 = affine.load %arg3[%arg6] : memref<12xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        affine.store %4, %arg1[%arg5] : memref<12xf32>
      }
    }
    affine.for %arg5 = 0 to 12 {
      affine.for %arg6 = 0 to 12 {
        %0 = affine.load %arg2[%arg5] : memref<12xf32>
        %1 = affine.load %arg0[%arg6, %arg5] : memref<12x12xf32>
        %2 = affine.load %arg4[%arg6] : memref<12xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        affine.store %4, %arg2[%arg5] : memref<12xf32>
      }
    }
    return
  }
