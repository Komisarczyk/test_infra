#map0 = affine_map<(d0) -> (d0)>
func @symm(%arg0: memref<10x10xf32>, %arg1: memref<10x12xf32>, %arg2: memref<10x12xf32>, %arg3: f32, %arg4: f32) {
    %0 = alloc() : memref<1xf32>
    affine.for %arg5 = 0 to 10 {
      affine.for %arg6 = 0 to 12 {
        %cst = constant 0.000000e+00 : f32
        %c0 = constant 0 : index
        affine.store %cst, %0[%c0] : memref<1xf32>
        affine.for %arg7 = 0 to #map0(%arg5) {
          %11 = affine.load %arg1[%arg5, %arg6] : memref<10x12xf32>
          %12 = mulf %arg3, %11 : f32
          %13 = affine.load %arg0[%arg5, %arg7] : memref<10x10xf32>
          %14 = mulf %12, %13 : f32
          %15 = affine.load %arg2[%arg7, %arg6] : memref<10x12xf32>
          %16 = addf %14, %15 : f32
          affine.store %16, %arg2[%arg7, %arg6] : memref<10x12xf32>
          %17 = affine.load %arg1[%arg7, %arg6] : memref<10x12xf32>
          %18 = affine.load %arg0[%arg5, %arg7] : memref<10x10xf32>
          %19 = mulf %17, %18 : f32
          %c0_1 = constant 0 : index
          %20 = affine.load %0[%c0_1] : memref<1xf32>
          %21 = addf %19, %20 : f32
          %c0_2 = constant 0 : index
          affine.store %21, %0[%c0_2] : memref<1xf32>
        }
        %1 = affine.load %arg2[%arg5, %arg6] : memref<10x12xf32>
        %2 = mulf %arg4, %1 : f32
        %3 = affine.load %arg1[%arg5, %arg6] : memref<10x12xf32>
        %4 = mulf %arg3, %3 : f32
        %5 = affine.load %arg0[%arg5, %arg5] : memref<10x10xf32>
        %6 = mulf %4, %5 : f32
        %7 = addf %2, %6 : f32
        %c0_0 = constant 0 : index
        %8 = affine.load %0[%c0_0] : memref<1xf32>
        %9 = mulf %arg3, %8 : f32
        %10 = addf %7, %9 : f32
        affine.store %10, %arg2[%arg5, %arg6] : memref<10x12xf32>
      }
    }
    return
  }


