func @atax(%arg0: memref<19x21xf32>, %arg1: memref<19xf32>, %arg2: memref<21xf32>, %arg3: memref<21xf32>) {
    affine.for %arg4 = 0 to 21 {
      %cst = constant 0.000000e+00 : f32
      affine.store %cst, %arg3[%arg4] : memref<21xf32>
    }
    affine.for %arg4 = 0 to 19 {
      %cst = constant 0.000000e+00 : f32
      affine.store %cst, %arg1[%arg4] : memref<19xf32>
      affine.for %arg5 = 0 to 21 {
        %0 = affine.load %arg1[%arg4] : memref<19xf32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<19x21xf32>
        %2 = affine.load %arg2[%arg5] : memref<21xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        affine.store %4, %arg1[%arg4] : memref<19xf32>
      }
      affine.for %arg5 = 0 to 21 {
        %0 = affine.load %arg3[%arg5] : memref<21xf32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<19x21xf32>
        %2 = affine.load %arg1[%arg4] : memref<19xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        affine.store %4, %arg3[%arg5] : memref<21xf32>
      }
    }
    return
  }

