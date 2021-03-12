import numpy as np
def Warp(output, target, reference, H, x_offset, y_offset):

    (h1, w1, ch) = target.shape

    target_bounds = np.array([[x_offset, y_offset, 1], [h1 + x_offset, y_offset, 1],
                                [x_offset, w1 + y_offset, 1], [h1 + x_offset, w1 + y_offset, 1]]).T
    
    transformed_bounds = np.dot(H, target_bounds)
    
    transformed_bounds /= transformed_bounds[2]
    
    xs = transformed_bounds[0]  # for i in transformed_bounds]
    ys = transformed_bounds[1]  # for i in transformed_bounds]
    minx = int(np.min(xs))
    maxx = int(np.max(xs))
    miny = int(np.min(ys))
    maxy = int(np.max(ys))

    invH = np.linalg.inv(H)

    for i in range(minx, maxx):
        for j in range(miny, maxy):
            point = np.array([j+x_offset, i+y_offset, 1]).T
            untransformed_point = invH.dot(point)
            untransformed_point /= untransformed_point[0, 2]
            print(untransformed_point)
            break
            
            print(np.array(untransformed_point))
            try:
                output[i][j] = img1[untransformed_point[0]
                                    ][untransformed_point[1]]
            except:
                output[i][j] = 0