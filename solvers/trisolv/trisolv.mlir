#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<() -> (0)>
#map3 = affine_map<() -> (5)>
#map4 = affine_map<() -> (990)>
#map5 = affine_map<() -> (1002)>

  func @trisolv(%arg0: memref<5x5xf32>, %arg1: memref<5xf32>, %arg2: memref<5xf32>) {
    affine.for %arg3 = 0 to 5 {
      %0 = affine.load %arg1[%arg3] : memref<5xf32>
      affine.store %0, %arg2[%arg3] : memref<5xf32>
      affine.for %arg4 = 0 to #map0(%arg3) {
        %4 = affine.load %arg0[%arg3, %arg4] : memref<5x5xf32>
        %5 = affine.load %arg2[%arg4] : memref<5xf32>
        %6 = mulf %4, %5 : f32
        %7 = affine.load %arg2[%arg3] : memref<5xf32>
        %8 = subf %7, %6 : f32
        affine.store %8, %arg2[%arg3] : memref<5xf32>
      }
      %1 = affine.load %arg2[%arg3] : memref<5xf32>
      %2 = affine.load %arg0[%arg3, %arg3] : memref<5x5xf32>
      %3 = divf %1, %2 : f32
      affine.store %3, %arg2[%arg3] : memref<5xf32>
    }
    return
  }
