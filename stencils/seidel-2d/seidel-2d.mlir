#map0 = affine_map<(d0) -> (d0 - 1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0) -> (d0 + 1)>
#map3 = affine_map<() -> (1)>
#map4 = affine_map<() -> (19)>
#map5 = affine_map<() -> (0)>
#map6 = affine_map<() -> (5)>


 
  func @seideltwod(%arg0: memref<20x20xf32>) {
    affine.for %arg1 = 0 to 5 {
      affine.for %arg2 = 1 to 19 {
        affine.for %arg3 = 1 to 19 {
          %0 = affine.apply #map0(%arg2)
          %1 = affine.apply #map0(%arg3)
          %2 = affine.load %arg0[%0, %1] : memref<20x20xf32>
          %3 = affine.apply #map0(%arg2)
          %4 = affine.load %arg0[%3, %arg3] : memref<20x20xf32>
          %5 = addf %2, %4 : f32
          %6 = affine.apply #map0(%arg2)
          %7 = affine.apply #map2(%arg3)
          %8 = affine.load %arg0[%6, %7] : memref<20x20xf32>
          %9 = addf %5, %8 : f32
          %10 = affine.apply #map0(%arg3)
          %11 = affine.load %arg0[%arg2, %10] : memref<20x20xf32>
          %12 = addf %9, %11 : f32
          %13 = affine.load %arg0[%arg2, %arg3] : memref<20x20xf32>
          %14 = addf %12, %13 : f32
          %15 = affine.apply #map2(%arg3)
          %16 = affine.load %arg0[%arg2, %15] : memref<20x20xf32>
          %17 = addf %14, %16 : f32
          %18 = affine.apply #map2(%arg2)
          %19 = affine.apply #map0(%arg3)
          %20 = affine.load %arg0[%18, %19] : memref<20x20xf32>
          %21 = addf %17, %20 : f32
          %22 = affine.apply #map2(%arg2)
          %23 = affine.load %arg0[%22, %arg3] : memref<20x20xf32>
          %24 = addf %21, %23 : f32
          %25 = affine.apply #map2(%arg2)
          %26 = affine.apply #map2(%arg3)
          %27 = affine.load %arg0[%25, %26] : memref<20x20xf32>
          %28 = addf %24, %27 : f32
          %cst = constant 9.000000e+00 : f32
          %29 = divf %28, %cst : f32
          affine.store %29, %arg0[%arg2, %arg3] : memref<20x20xf32>
        }
      }
    }
    return
  }
