#map0 = affine_map<(d0) -> (d0 - 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<() -> (1)>
#map4 = affine_map<() -> (1999)>
#map5 = affine_map<() -> (0)>
#map6 = affine_map<() -> (500)>


  func @jacobioned(%arg0: memref<12xf32>, %arg1: memref<12xf32>) {
    affine.for %arg2 = 0 to 500 {
      affine.for %arg3 = 1 to 1999 {
        %cst = constant 3.333300e-01 : f32
        %0 = affine.apply #map0(%arg3)
        %1 = affine.load %arg0[%0] : memref<12xf32>
        %2 = affine.load %arg0[%arg3] : memref<12xf32>
        %3 = addf %1, %2 : f32
        %4 = affine.apply #map2(%arg3)
        %5 = affine.load %arg0[%4] : memref<12xf32>
        %6 = addf %3, %5 : f32
        %7 = mulf %cst, %6 : f32
        affine.store %7, %arg1[%arg3] : memref<12xf32>
      }
      affine.for %arg3 = 1 to 1999 {
        %cst = constant 3.333300e-01 : f32
        %0 = affine.apply #map0(%arg3)
        %1 = affine.load %arg1[%0] : memref<12xf32>
        %2 = affine.load %arg1[%arg3] : memref<12xf32>
        %3 = addf %1, %2 : f32
        %4 = affine.apply #map2(%arg3)
        %5 = affine.load %arg1[%4] : memref<12xf32>
        %6 = addf %3, %5 : f32
        %7 = mulf %cst, %6 : f32
        affine.store %7, %arg0[%arg3] : memref<12xf32>
      }
    }
    return
  }

  %A = alloc() : memref<12xf32>
  %B = alloc() : memref<12xf32>

