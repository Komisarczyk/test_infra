#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (11)>
#map4 = affine_map<() -> (9)>
#map5 = affine_map<() -> (8)>
#map6 = affine_map<() -> (12)>
#map7 = affine_map<() -> (1190)>
#map8 = affine_map<() -> (798)>


func @twomm(%arg0: memref<8x11xf32>, %arg1: memref<11x9xf32>, %arg2: memref<9x12xf32>, %arg3: memref<8x12xf32>, %arg4: f32, %arg5: f32, %arg6: memref<8x9xf32>) {
    affine.for %arg7 = 0 to 8 {
      affine.for %arg8 = 0 to 9 {
        %cst = constant 0.000000e+00 : f32
        affine.store %cst, %arg6[%arg7, %arg8] : memref<8x9xf32>
        affine.for %arg9 = 0 to 11 {
          %0 = affine.load %arg0[%arg7, %arg9] : memref<8x11xf32>
          %1 = mulf %arg4, %0 : f32
          %2 = affine.load %arg1[%arg9, %arg8] : memref<11x9xf32>
          %3 = mulf %1, %2 : f32
          %4 = affine.load %arg6[%arg7, %arg8] : memref<8x9xf32>
          %5 = addf %3, %4 : f32
          affine.store %5, %arg6[%arg7, %arg8] : memref<8x9xf32>
        }
      }
    }
    affine.for %arg7 = 0 to 8 {
      affine.for %arg8 = 0 to 12 {
        %0 = affine.load %arg3[%arg7, %arg8] : memref<8x12xf32>
        %1 = mulf %arg5, %0 : f32
        affine.store %1, %arg3[%arg7, %arg8] : memref<8x12xf32>
        affine.for %arg9 = 0 to 9 {
          %2 = affine.load %arg6[%arg7, %arg9] : memref<8x9xf32>
          %3 = affine.load %arg2[%arg9, %arg8] : memref<9x12xf32>
          %4 = mulf %2, %3 : f32
          %5 = affine.load %arg3[%arg7, %arg8] : memref<8x12xf32>
          %6 = addf %4, %5 : f32
          affine.store %6, %arg3[%arg7, %arg8] : memref<8x12xf32>
        }
      }
    }
    return
  }

