import matplotlib.pyplot as plt
import numpy as np



def main():
    """
    v0 = np.array([1.2, 1.2])
    v1 = np.array([5.9, 7.6])
    v2 = np.array([14.6, 1.2])
    """
    v0 = np.array([3.5, 4]) * 20
    v1 = np.array([4.75, 1]) * 20
    v2 = np.array([6, 6]) * 20

    plt.plot(*v0, *v1, 'bo')  # plot x and y using blue circle markers
    plt.plot(*v1, *v2, 'bo')
    plt.plot(*v2, *v0, 'bo')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)

    # Values at vertices (for example purposes, use values of your choice)
    values = np.array([v0, v1, v2]).T

    # Get the bounding box of the quadrilateral
    min_x = int(min(v0[0], v1[0], v2[0]))
    max_x = int(max(v0[0]+1, v1[0]+1, v2[0]+1))
    min_y = int(min(v0[1], v1[1], v2[1]))
    max_y = int(max(v0[1]+1, v1[1]+1, v2[1]+1))

    # Iterate over each pixel in the bounding box
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            # Compute barycentric coordinates
            # bc = compute_barycentric(x + 0.5, y + 0.5, v0, v1, v2)  # Using pixel center
            bc = compute_barycentric(x, y, v0, v1, v2)  # Using pixel center

            #print("BC SHAPE:", bc.shape)
            # Check if the pixel is inside the quadrilateral
            if all(0 <= bc) and all(bc <= 1):
                # Perform bilinear interpolation
                interpolated_value = values @ bc
                #print("TPOSE:", interpolated_value.T)
                x, y = interpolated_value.T[0]
                plt.plot(x, y, 'k.')
                # plt.plot(*v0, *v1, 'bo')
                # Do something with the interpolated value (e.g., store or process it)
                #print(f"Pixel ({x}, {y}): Interpolated Value = {interpolated_value}")
    plt.show()



def bilinear_interpolation(p, x1, x2, y1, y2, q11, q12, q21, q22):
    """
    From the book, Numerical Recipes in C: the Art of Scientific Computing
    """
    x, y = p
    term1 = 1/((x2-x1)*(y2-y1))
    term2 = np.array([[x2-x, x-x1]])
    term3 = np.array([[q11, q12], [q21, q22]])
    term4 = np.array([[y2-y], [y-y1]])
    solution = term1 * term2 @ term3 @ term4
    return solution


def unitcube_bilinear_interpolation(p, q11, q12, q21, q22):
    """
    Uses same algorithm as routine above, but adjusted around a cube at (0, 0, 1, 1).
    This simplifies a lot of things.
    It ends up looking like a straightforward bilinear interpolation mathematically.
    """
    #setup point location
    x, y = p
    assert x >= 0 and y >= 0, "Point must be in first quadrant."
    if x > 1:
        x, y = x - int(x), y - int(y)

    #interpolate
    _x = 1-x
    left_term = (1-y) * (q11*_x + q21*x)
    right_term = y*(q12*_x + q22*x)
    solution = left_term + right_term
    return solution




# Function to compute barycentric coordinates of a point (px, py) in a triangle
def compute_barycentric(px, py, v0, v1, v2):
    # Setup Ax=b
    A = np.array([
        [v0[0], v1[0], v2[0]],
        [v0[1], v1[1], v2[1]],
        [1, 1, 1]
    ])
    b = np.array([[px, py, 1]]).T


    # Solve linear system.
    # lstsq returns a 4-tuple of (x, residuals, rank, svd_of_A). I only need x, labeled l for lambda in this case.
    l, _, _, _ = np.linalg.lstsq(A, b, rcond=-1)

    return l




if __name__ == "__main__":
    main()
