#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<() -> (4)>
#map4 = affine_map<() -> (6)>



  func @syr2k(%arg0: memref<6x4xf32>, %arg1: memref<6x4xf32>, %arg2: memref<6x6xf32>, %arg3: f32, %arg4: f32) {
    affine.for %arg5 = 0 to 6 {
      affine.for %arg6 = 0 to #map2(%arg5) {
        %0 = affine.load %arg2[%arg5, %arg6] : memref<6x6xf32>
        %1 = mulf %arg4, %0 : f32
        affine.store %1, %arg2[%arg5, %arg6] : memref<6x6xf32>
      }
      affine.for %arg6 = 0 to 4 {
        affine.for %arg7 = 0 to #map2(%arg5) {
          %0 = affine.load %arg0[%arg7, %arg6] : memref<6x4xf32>
          %1 = mulf %0, %arg3 : f32
          %2 = affine.load %arg1[%arg5, %arg6] : memref<6x4xf32>
          %3 = mulf %1, %2 : f32
          %4 = affine.load %arg1[%arg7, %arg6] : memref<6x4xf32>
          %5 = mulf %4, %arg3 : f32
          %6 = affine.load %arg0[%arg5, %arg6] : memref<6x4xf32>
          %7 = mulf %5, %6 : f32
          %8 = addf %3, %7 : f32
          %9 = affine.load %arg2[%arg5, %arg7] : memref<6x6xf32>
          %10 = addf %8, %9 : f32
          affine.store %10, %arg2[%arg5, %arg7] : memref<6x6xf32>
        }
      }
    }
    return
  }

