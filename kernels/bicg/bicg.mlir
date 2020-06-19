 func @bicg(%arg0: memref<21x19xf32>, %arg1: memref<19xf32>, %arg2: memref<21xf32>, %arg3: memref<21xf32>, %arg4: memref<19xf32>) {
    affine.for %arg5 = 0 to 19 {
      %cst = constant 0.000000e+00 : f32
      affine.store %cst, %arg4[%arg5] : memref<19xf32>
    }
    affine.for %arg5 = 0 to 21 {
      %cst = constant 0.000000e+00 : f32
      affine.store %cst, %arg2[%arg5] : memref<21xf32>
      affine.for %arg6 = 0 to 19 {
        %0 = affine.load %arg4[%arg6] : memref<19xf32>
        %1 = affine.load %arg3[%arg5] : memref<21xf32>
        %2 = affine.load %arg0[%arg5, %arg6] : memref<21x19xf32>
        %3 = mulf %1, %2 : f32
        %4 = addf %0, %3 : f32
        affine.store %4, %arg4[%arg6] : memref<19xf32>
        %5 = affine.load %arg2[%arg5] : memref<21xf32>
        %6 = affine.load %arg0[%arg5, %arg6] : memref<21x19xf32>
        %7 = affine.load %arg1[%arg6] : memref<19xf32>
        %8 = mulf %6, %7 : f32
        %9 = addf %5, %8 : f32
        affine.store %9, %arg2[%arg5] : memref<21xf32>
      }
    }
    return
  }