#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<() -> (0)>
#map2 = affine_map<(d0) -> (d0)>
#map3 = affine_map<() -> (5)>
#map4 = affine_map<() -> (1)>


  func @lu(%arg0: memref<5x5xf32>) {
    affine.for %arg1 = 1 to 5 {
      affine.for %arg2 = 0 to #map2(%arg1) {
        affine.for %arg3 = 0 to #map2(%arg2) {
          %3 = affine.load %arg0[%arg1, %arg3] : memref<5x5xf32>
          %4 = affine.load %arg0[%arg3, %arg2] : memref<5x5xf32>
          %5 = mulf %3, %4 : f32
          %6 = affine.load %arg0[%arg1, %arg2] : memref<5x5xf32>
          %7 = subf %6, %5 : f32
          affine.store %7, %arg0[%arg1, %arg2] : memref<5x5xf32>
        }
        %0 = affine.load %arg0[%arg2, %arg2] : memref<5x5xf32>
        %1 = affine.load %arg0[%arg1, %arg2] : memref<5x5xf32>
        %2 = divf %1, %0 : f32
        affine.store %2, %arg0[%arg1, %arg2] : memref<5x5xf32>
      }
      affine.for %arg2 = #map2(%arg1) to 5 {
        affine.for %arg3 = 0 to #map2(%arg1) {
          %0 = affine.load %arg0[%arg1, %arg3] : memref<5x5xf32>
          %1 = affine.load %arg0[%arg3, %arg2] : memref<5x5xf32>
          %2 = mulf %0, %1 : f32
          %3 = affine.load %arg0[%arg1, %arg2] : memref<5x5xf32>
          %4 = subf %3, %2 : f32
          affine.store %4, %arg0[%arg1, %arg2] : memref<5x5xf32>
        }
      }
    }
    return
  }
