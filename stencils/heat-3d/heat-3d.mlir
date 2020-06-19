#map0 = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map2 = affine_map<(d0) -> (d0 - 1)>
#map3 = affine_map<() -> (1)>
#map4 = affine_map<() -> (2)>
#map5 = affine_map<() -> (6)>

  func @heat(%arg0: memref<3x3x3xf32>, %arg1: memref<3x3x3xf32>) {
    affine.for %arg2 = 1 to 6 {
      affine.for %arg3 = 1 to 2 {
        affine.for %arg4 = 1 to 2 {
          affine.for %arg5 = 1 to 2 {
            %cst = constant 1.250000e-01 : f32
            %0 = affine.apply #map0(%arg3)
            %1 = affine.load %arg0[%0, %arg4, %arg5] : memref<3x3x3xf32>
            %cst_0 = constant 2.000000e+00 : f32
            %2 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %3 = mulf %cst_0, %2 : f32
            %4 = subf %1, %3 : f32
            %5 = affine.apply #map2(%arg3)
            %6 = affine.load %arg0[%5, %arg4, %arg5] : memref<3x3x3xf32>
            %7 = addf %4, %6 : f32
            %8 = mulf %cst, %7 : f32
            %cst_1 = constant 1.250000e-01 : f32
            %9 = affine.apply #map0(%arg4)
            %10 = affine.load %arg0[%arg3, %9, %arg5] : memref<3x3x3xf32>
            %cst_2 = constant 2.000000e+00 : f32
            %11 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %12 = mulf %cst_2, %11 : f32
            %13 = subf %10, %12 : f32
            %14 = affine.apply #map2(%arg4)
            %15 = affine.load %arg0[%arg3, %14, %arg5] : memref<3x3x3xf32>
            %16 = addf %13, %15 : f32
            %17 = mulf %cst_1, %16 : f32
            %18 = addf %8, %17 : f32
            %cst_3 = constant 1.250000e-01 : f32
            %19 = affine.apply #map0(%arg5)
            %20 = affine.load %arg0[%arg3, %arg4, %19] : memref<3x3x3xf32>
            %cst_4 = constant 2.000000e+00 : f32
            %21 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %22 = mulf %cst_4, %21 : f32
            %23 = subf %20, %22 : f32
            %24 = affine.apply #map2(%arg5)
            %25 = affine.load %arg0[%arg3, %arg4, %24] : memref<3x3x3xf32>
            %26 = addf %23, %25 : f32
            %27 = mulf %cst_3, %26 : f32
            %28 = addf %18, %27 : f32
            %29 = affine.load %arg0[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %30 = addf %28, %29 : f32
            affine.store %30, %arg1[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
          }
        }
      }
      affine.for %arg3 = 1 to 2 {
        affine.for %arg4 = 1 to 2 {
          affine.for %arg5 = 1 to 2 {
            %cst = constant 1.250000e-01 : f32
            %0 = affine.apply #map0(%arg3)
            %1 = affine.load %arg1[%0, %arg4, %arg5] : memref<3x3x3xf32>
            %cst_0 = constant 2.000000e+00 : f32
            %2 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %3 = mulf %cst_0, %2 : f32
            %4 = subf %1, %3 : f32
            %5 = affine.apply #map2(%arg3)
            %6 = affine.load %arg1[%5, %arg4, %arg5] : memref<3x3x3xf32>
            %7 = addf %4, %6 : f32
            %8 = mulf %cst, %7 : f32
            %cst_1 = constant 1.250000e-01 : f32
            %9 = affine.apply #map0(%arg4)
            %10 = affine.load %arg1[%arg3, %9, %arg5] : memref<3x3x3xf32>
            %cst_2 = constant 2.000000e+00 : f32
            %11 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %12 = mulf %cst_2, %11 : f32
            %13 = subf %10, %12 : f32
            %14 = affine.apply #map2(%arg4)
            %15 = affine.load %arg1[%arg3, %14, %arg5] : memref<3x3x3xf32>
            %16 = addf %13, %15 : f32
            %17 = mulf %cst_1, %16 : f32
            %18 = addf %8, %17 : f32
            %cst_3 = constant 1.250000e-01 : f32
            %19 = affine.apply #map0(%arg5)
            %20 = affine.load %arg1[%arg3, %arg4, %19] : memref<3x3x3xf32>
            %cst_4 = constant 2.000000e+00 : f32
            %21 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %22 = mulf %cst_4, %21 : f32
            %23 = subf %20, %22 : f32
            %24 = affine.apply #map2(%arg5)
            %25 = affine.load %arg1[%arg3, %arg4, %24] : memref<3x3x3xf32>
            %26 = addf %23, %25 : f32
            %27 = mulf %cst_3, %26 : f32
            %28 = addf %18, %27 : f32
            %29 = affine.load %arg1[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
            %30 = addf %28, %29 : f32
            affine.store %30, %arg0[%arg3, %arg4, %arg5] : memref<3x3x3xf32>
          }
        }
      }
    }
    return
  }
 