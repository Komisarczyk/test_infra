func @threemm(%arg0: memref<8x10xf32>, %arg1: memref<10x9xf32>, %arg2: memref<9x12xf32>, %arg3: memref<12x11xf32>, %arg4: memref<8x9xf32>, %arg5: memref<9x11xf32>, %arg6: memref<8x11xf32>) {
    affine.for %arg7 = 0 to 8 {
      affine.for %arg8 = 0 to 9 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg4[%arg7, %arg8] : memref<8x9xf32>
        affine.for %arg9 = 0 to 10 {
          %0 = affine.load %arg0[%arg7, %arg9] : memref<8x10xf32>
          %1 = affine.load %arg1[%arg9, %arg8] : memref<10x9xf32>
          %2 = mulf %0, %1 : f32
          %3 = affine.load %arg4[%arg7, %arg8] : memref<8x9xf32>
          %4 = addf %2, %3 : f32
          affine.store %4, %arg4[%arg7, %arg8] : memref<8x9xf32>
        }
      }
    }
    affine.for %arg7 = 0 to 9 {
      affine.for %arg8 = 0 to 11 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg5[%arg7, %arg8] : memref<9x11xf32>
        affine.for %arg9 = 0 to 12 {
          %0 = affine.load %arg2[%arg7, %arg9] : memref<9x12xf32>
          %1 = affine.load %arg3[%arg9, %arg8] : memref<12x11xf32>
          %2 = mulf %0, %1 : f32
          %3 = affine.load %arg5[%arg7, %arg8] : memref<9x11xf32>
          %4 = addf %2, %3 : f32
          affine.store %4, %arg5[%arg7, %arg8] : memref<9x11xf32>
        }
      }
    }
    affine.for %arg7 = 0 to 8 {
      affine.for %arg8 = 0 to 11 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg6[%arg7, %arg8] : memref<8x11xf32>
        affine.for %arg9 = 0 to 9 {
          %0 = affine.load %arg4[%arg7, %arg9] : memref<8x9xf32>
          %1 = affine.load %arg5[%arg9, %arg8] : memref<9x11xf32>
          %2 = mulf %0, %1 : f32
          %3 = affine.load %arg6[%arg7, %arg8] : memref<8x11xf32>
          %4 = addf %2, %3 : f32
          affine.store %4, %arg6[%arg7, %arg8] : memref<8x11xf32>
        }
      }
    }
    return
  }
