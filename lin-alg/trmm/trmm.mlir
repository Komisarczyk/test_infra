#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<() -> (4)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<() -> (6)>

func @trmm(%arg0: memref<4x4xf32>, %arg1: memref<4x6xf32>, %arg2: f32) {
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 6 {
        affine.for %arg5 = #map1(%arg3) to 4 {
          %2 = affine.load %arg0[%arg5, %arg3] : memref<4x4xf32>
          %3 = affine.load %arg1[%arg5, %arg4] : memref<4x6xf32>
          %4 = mulf %2, %3 : f32
          %5 = affine.load %arg1[%arg3, %arg4] : memref<4x6xf32>
          %6 = addf %4, %5 : f32
          affine.store %6, %arg1[%arg3, %arg4] : memref<4x6xf32>
        }
        %0 = affine.load %arg1[%arg3, %arg4] : memref<4x6xf32>
        %1 = mulf %arg2, %0 : f32
        affine.store %1, %arg1[%arg3, %arg4] : memref<4x6xf32>
      }
    }
    return
  }
