// src/fdtd_solver.rs
use ndarray::{Array3, Array4};

/// Calculate Laplacian using 7-point stencil with periodic boundaries
pub fn laplacian_3d(field: &Array3<f64>, i: usize, j: usize, k: usize, dx: f64) -> f64 {
    let shape = field.shape();
    let nx = shape[0];
    let ny = shape[1];
    let nz = shape[2];

    let inv_dx2 = 1.0 / (dx * dx);

    // Periodic boundary conditions
    let im = if i == 0 { nx - 1 } else { i - 1 };
    let ip = if i == nx - 1 { 0 } else { i + 1 };
    let jm = if j == 0 { ny - 1 } else { j - 1 };
    let jp = if j == ny - 1 { 0 } else { j + 1 };
    let km = if k == 0 { nz - 1 } else { k - 1 };
    let kp = if k == nz - 1 { 0 } else { k + 1 };

    let center = field[[i, j, k]];
    let sum_neighbors = field[[ip, j, k]] + field[[im, j, k]] +
                       field[[i, jp, k]] + field[[i, jm, k]] +
                       field[[i, j, kp]] + field[[i, j, km]];

    (sum_neighbors - 6.0 * center) * inv_dx2
}

/// Calculate Laplacian for a specific component of a vector field
pub fn laplacian_component(field: &Array4<f64>, comp: usize, i: usize, j: usize, k: usize, dx: f64) -> f64 {
    laplacian_3d(&field.slice(ndarray::s![comp, .., .., ..]).to_owned(), i, j, k, dx)
}

/// Calculate curl for a specific component
pub fn curl_component(field: &Array4<f64>, comp: usize, i: usize, j: usize, k: usize, dx: f64) -> f64 {
    let shape = field.shape();
    let nx = shape[1];
    let ny = shape[2];
    let nz = shape[3];

    let inv_2dx = 0.5 / dx;

    match comp {
        // (∇ × A)_x = ∂_y A_z - ∂_z A_y
        0 => {
            let jp = if j == ny - 1 { 0 } else { j + 1 };
            let jm = if j == 0 { ny - 1 } else { j - 1 };
            let kp = if k == nz - 1 { 0 } else { k + 1 };
            let km = if k == 0 { nz - 1 } else { k - 1 };

            let dydz = (field[[2, i, jp, k]] - field[[2, i, jm, k]]) * inv_2dx;
            let dzdy = (field[[1, i, j, kp]] - field[[1, i, j, km]]) * inv_2dx;

            dydz - dzdy
        },
        // (∇ × A)_y = ∂_z A_x - ∂_x A_z
        1 => {
            let kp = if k == nz - 1 { 0 } else { k + 1 };
            let km = if k == 0 { nz - 1 } else { k - 1 };
            let ip = if i == nx - 1 { 0 } else { i + 1 };
            let im = if i == 0 { nx - 1 } else { i - 1 };

            let dzdx = (field[[0, i, j, kp]] - field[[0, i, j, km]]) * inv_2dx;
            let dxdz = (field[[2, ip, j, k]] - field[[2, im, j, k]]) * inv_2dx;

            dzdx - dxdz
        },
        // (∇ × A)_z = ∂_x A_y - ∂_y A_x
        2 => {
            let ip = if i == nx - 1 { 0 } else { i + 1 };
            let im = if i == 0 { nx - 1 } else { i - 1 };
            let jp = if j == ny - 1 { 0 } else { j + 1 };
            let jm = if j == 0 { ny - 1 } else { j - 1 };

            let dxdy = (field[[1, ip, j, k]] - field[[1, im, j, k]]) * inv_2dx;
            let dydx = (field[[0, i, jp, k]] - field[[0, i, jm, k]]) * inv_2dx;

            dxdy - dydx
        },
        _ => 0.0,
    }
}

/// Calculate divergence of vector field (for Gauss constraint if needed)
pub fn divergence(field: &Array4<f64>, i: usize, j: usize, k: usize, dx: f64) -> f64 {
    let shape = field.shape();
    let nx = shape[1];
    let ny = shape[2];
    let nz = shape[3];

    let inv_2dx = 0.5 / dx;

    let ip = if i == nx - 1 { 0 } else { i + 1 };
    let im = if i == 0 { nx - 1 } else { i - 1 };
    let jp = if j == ny - 1 { 0 } else { j + 1 };
    let jm = if j == 0 { ny - 1 } else { j - 1 };
    let kp = if k == nz - 1 { 0 } else { k + 1 };
    let km = if k == 0 { nz - 1 } else { k - 1 };

    let dAdx = (field[[0, ip, j, k]] - field[[0, im, j, k]]) * inv_2dx;
    let dBdy = (field[[1, i, jp, k]] - field[[1, i, jm, k]]) * inv_2dx;
    let dCdz = (field[[2, i, j, kp]] - field[[2, i, j, km]]) * inv_2dx;

    dAdx + dBdy + dCdz
}
