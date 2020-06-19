#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<() -> (4)>
#map4 = affine_map<() -> (6)>



  func @syrk(%arg0: memref<6x4xf32>, %arg1: memref<6x6xf32>, %arg2: f32, %arg3: f32) {
    affine.for %arg4 = 0 to 6 {
      affine.for %arg5 = 0 to #map2(%arg4) {
        %0 = affine.load %arg1[%arg4, %arg5] : memref<6x6xf32>
        %1 = mulf %arg3, %0 : f32
        affine.store %1, %arg1[%arg4, %arg5] : memref<6x6xf32>
      }
      affine.for %arg5 = 0 to 4 {
        affine.for %arg6 = 0 to #map2(%arg4) {
          %0 = affine.load %arg0[%arg4, %arg5] : memref<6x4xf32>
          %1 = mulf %arg2, %0 : f32
          %2 = affine.load %arg0[%arg6, %arg5] : memref<6x4xf32>
          %3 = mulf %1, %2 : f32
          %4 = affine.load %arg1[%arg4, %arg6] : memref<6x6xf32>
          %5 = addf %3, %4 : f32
          affine.store %5, %arg1[%arg4, %arg6] : memref<6x6xf32>
        }
      }
    }
    return
  }



