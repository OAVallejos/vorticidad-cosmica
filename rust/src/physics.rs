// src/physics.rs - VERSIÓN COMPLETA CON ONDAS GRAVITATORIAS
use ndarray::{Array3, Array4, ArrayView3, ArrayView4, Zip};
use crate::CosmologicalParameters;

/// Calculates the Laplacian of a 3D scalar field using a 7-point stencil
/// with periodic boundary conditions.
/// ∇²φ = (φ(x+dx) + φ(x-dx) + φ(y+dy) + ... - 6φ(x)) / dx²
fn compute_laplacian_periodic(field: &ArrayView3<f64>, dx: f64) -> Array3<f64> {
    let (nx, ny, nz) = field.dim();
    let mut laplacian = Array3::zeros((nx, ny, nz));
    let inv_dx2 = 1.0 / (dx * dx);

    for i in 0..nx {
        let im = if i == 0 { nx - 1 } else { i - 1 };
        let ip = if i == nx - 1 { 0 } else { i + 1 };

        for j in 0..ny {
            let jm = if j == 0 { ny - 1 } else { j - 1 };
            let jp = if j == ny - 1 { 0 } else { j + 1 };

            for k in 0..nz {
                let km = if k == 0 { nz - 1 } else { k - 1 };
                let kp = if k == nz - 1 { 0 } else { k + 1 };

                let center = field[[i, j, k]];

                let sum_neighbors = field[[ip, j, k]] + field[[im, j, k]] +
                                    field[[i, jp, k]] + field[[i, jm, k]] +
                                    field[[i, j, kp]] + field[[i, j, km]];

                laplacian[[i, j, k]] = (sum_neighbors - 6.0 * center) * inv_dx2;
            }
        }
    }

    laplacian
}

/// Calculates the curl of a vector field A at a specific point for a specific component
/// (∇ × A)_i = ε_{ijk} ∂_j A_k
fn compute_curl_component(
    a_i: &ArrayView4<f64>,
    comp: usize,
    i: usize, j: usize, k: usize,
    dx: f64
) -> f64 {
    let (_, nx, ny, nz) = a_i.dim();
    let inv_dx = 1.0 / dx;

    match comp {
        // (∇ × A)_x = ∂_y A_z - ∂_z A_y
        0 => {
            let jp = if j == ny - 1 { 0 } else { j + 1 };
            let jm = if j == 0 { ny - 1 } else { j - 1 };
            let kp = if k == nz - 1 { 0 } else { k + 1 };
            let km = if k == 0 { nz - 1 } else { k - 1 };

            let dydz = (a_i[[2, i, jp, k]] - a_i[[2, i, jm, k]]) * inv_dx;
            let dzdy = (a_i[[1, i, j, kp]] - a_i[[1, i, j, km]]) * inv_dx;

            dydz - dzdy
        },
        // (∇ × A)_y = ∂_z A_x - ∂_x A_z
        1 => {
            let kp = if k == nz - 1 { 0 } else { k + 1 };
            let km = if k == 0 { nz - 1 } else { k - 1 };
            let ip = if i == nx - 1 { 0 } else { i + 1 };
            let im = if i == 0 { nx - 1 } else { i - 1 };

            let dzdx = (a_i[[0, i, j, kp]] - a_i[[0, i, j, km]]) * inv_dx;
            let dxdz = (a_i[[2, ip, j, k]] - a_i[[2, im, j, k]]) * inv_dx;

            dzdx - dxdz
        },
        // (∇ × A)_z = ∂_x A_y - ∂_y A_x
        2 => {
            let ip = if i == nx - 1 { 0 } else { i + 1 };
            let im = if i == 0 { nx - 1 } else { i - 1 };
            let jp = if j == ny - 1 { 0 } else { j + 1 };
            let jm = if j == 0 { ny - 1 } else { j - 1 };

            let dxdy = (a_i[[1, ip, j, k]] - a_i[[1, im, j, k]]) * inv_dx;
            let dydx = (a_i[[0, i, jp, k]] - a_i[[0, i, jm, k]]) * inv_dx;

            dxdy - dydx
        },
        _ => 0.0,
    }
}

/// Calculates E·B term for the scalar field evolution
fn compute_eb_term(
    a_i: &ArrayView4<f64>,
    phi_dot: &ArrayView3<f64>,
    i: usize, j: usize, k: usize,
    dx: f64,
    a: f64,
) -> f64 {
    let inv_dx = 1.0 / dx;
    let (_, nx, ny, nz) = a_i.dim();

    // Calculate electric field E = -∂_t A = -(A_{n+1} - A_n)/dt
    // But we need the curl of A for B = ∇ × A
    let b_x = compute_curl_component(a_i, 0, i, j, k, dx);
    let b_y = compute_curl_component(a_i, 1, i, j, k, dx);
    let b_z = compute_curl_component(a_i, 2, i, j, k, dx);

    // Simplified version: use spatial derivatives as proxy for E
    // E_i ≈ -∂_i φ_dot for coupling term
    let e_x = {
        let ip = if i == nx - 1 { 0 } else { i + 1 };
        let im = if i == 0 { nx - 1 } else { i - 1 };
        (phi_dot[[im, j, k]] - phi_dot[[ip, j, k]]) * 0.5 * inv_dx
    };

    let e_y = {
        let jp = if j == ny - 1 { 0 } else { j + 1 };
        let jm = if j == 0 { ny - 1 } else { j - 1 };
        (phi_dot[[i, jm, k]] - phi_dot[[i, jp, k]]) * 0.5 * inv_dx
    };

    let e_z = {
        let kp = if k == nz - 1 { 0 } else { k + 1 };
        let km = if k == 0 { nz - 1 } else { k - 1 };
        (phi_dot[[i, j, km]] - phi_dot[[i, j, kp]]) * 0.5 * inv_dx
    };

    // E·B = E_x B_x + E_y B_y + E_z B_z
    (e_x * b_x + e_y * b_y + e_z * b_z) / (a * a * a) // Factors of a for FLRW metric
}

/// Evolves the scalar field φ according to the Klein-Gordon equation in FLRW metric:
/// φ'' + 3Hφ' - (1/a²)∇²φ + m_φ²φ + g² A² φ = 0
/// With axial coupling: ∂L/∂φ = (g/2) F_{μν} F̃^{μν} = g E·B
pub fn evolve_phi_field(
    phi: &Array3<f64>,
    phi_prev: &Array3<f64>,
    a_i: &Array4<f64>,
    params: &CosmologicalParameters,
    dx: f64,
    dt: f64,
    a: f64,
    h: f64,
) -> Array3<f64> {
    let shape = phi.dim();
    let phi_view = phi.view();
    let a_i_view = a_i.view();

    // 1. Calculate Laplacian ∇²φ
    let laplacian = compute_laplacian_periodic(&phi_view, dx);

    // 2. Calculate magnitude of vector field |A|² = Σ A_i² at each point
    let mut a_squared = Array3::<f64>::zeros(shape);
    for comp in 0..3 {
        let component = a_i.slice(ndarray::s![comp, .., .., ..]);
        Zip::from(&mut a_squared)
            .and(&component)
            .for_each(|acc, &val| *acc += val * val);
    }

    // 3. Calculate temporal derivative of φ for E·B term
    let phi_dot = {
        let mut dot = Array3::zeros(shape);
        Zip::from(&mut dot)
            .and(phi)
            .and(phi_prev)
            .for_each(|d, &curr, &prev| *d = (curr - prev) / dt);
        dot
    };

    // 4. Evolve using central difference for second derivative and friction
    // Eq: (φ_{n+1} - 2φ_n + φ_{n-1})/dt² + 3H(φ_{n+1} - φ_{n-1})/(2dt) = RHS
    // RHS = (1/a²)∇²φ - m²φ - (g²/a²)|A|²φ - (g/4) E·B

    let dt2 = dt * dt;
    let damping_factor = 1.5 * h * dt;  // 3H Δt/2
    let denom = 1.0 + damping_factor;   // (1 + 3HΔt/2)
    let alpha = 1.0 - damping_factor;   // (1 - 3HΔt/2)

    let mut phi_next = Array3::zeros(shape);

    // Pre-calculate constants
    let m2 = params.m_phi * params.m_phi;
    let g2 = params.g * params.g;
    let g = params.g;
    let a2 = a * a;
    let inv_a2 = 1.0 / a2;

    let (nx, ny, nz) = shape;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let curr = phi[[i, j, k]];
                let prev = phi_prev[[i, j, k]];
                let lap = laplacian[[i, j, k]];
                let vec_mag = a_squared[[i, j, k]];

                // Calculate E·B term for axial coupling
                let eb_term = compute_eb_term(&a_i_view, &phi_dot.view(), i, j, k, dx, a);

                // Gradient term: (1/a²)∇²φ
                let grad_term = lap * inv_a2;

                // Mass term: -m²φ
                let mass_term = -m2 * curr;

                // Vector field interaction: -(g²/a²)|A|²φ
                let vec_interaction = -g2 * vec_mag * curr * inv_a2;

                // Axial coupling: -(g/4) E·B
                let axial_coupling = -0.25 * g * eb_term;

                let force = grad_term + mass_term + vec_interaction + axial_coupling;

                // Verlet integration step with Hubble friction
                phi_next[[i, j, k]] = (2.0 * curr - alpha * prev + dt2 * force) / denom;
            }
        }
    }

    phi_next
}

/// Evolves the vector field A_μ according to the Proca equation in FLRW metric
/// A'' + H A' - (1/a²)∇²A + m_A² A + g²φ² A = 0
/// With axial coupling: ∂_μ(g φ) F̃^{μν}
pub fn evolve_vector_field(
    a_i: &Array4<f64>,
    a_i_prev: &Array4<f64>,
    phi: &Array3<f64>,
    params: &CosmologicalParameters,
    dx: f64,
    dt: f64,
    a: f64,
    h: f64,
) -> Array4<f64> {
    let (_, nx, ny, nz) = a_i.dim();
    let mut a_next = Array4::zeros((3, nx, ny, nz));

    let dt2 = dt * dt;
    // Vector fields have H damping in comoving coordinates
    let damping_factor = 0.5 * h * dt;
    let denom = 1.0 + damping_factor;
    let alpha = 1.0 - damping_factor;

    let m_a2 = params.m_a * params.m_a;
    let g = params.g;
    let a2 = a * a;
    let inv_a2 = 1.0 / a2;

    // Calculate gradient of φ for axial coupling
    let (phi_grad_x, phi_grad_y, phi_grad_z) = {
        let mut grad_x = Array3::zeros((nx, ny, nz));
        let mut grad_y = Array3::zeros((nx, ny, nz));
        let mut grad_z = Array3::zeros((nx, ny, nz));

        let inv_2dx = 0.5 / dx;

        for i in 0..nx {
            let ip = if i == nx - 1 { 0 } else { i + 1 };
            let im = if i == 0 { nx - 1 } else { i - 1 };

            for j in 0..ny {
                let jp = if j == ny - 1 { 0 } else { j + 1 };
                let jm = if j == 0 { ny - 1 } else { j - 1 };

                for k in 0..nz {
                    let kp = if k == nz - 1 { 0 } else { k + 1 };
                    let km = if k == 0 { nz - 1 } else { k - 1 };

                    grad_x[[i, j, k]] = (phi[[ip, j, k]] - phi[[im, j, k]]) * inv_2dx;
                    grad_y[[i, j, k]] = (phi[[i, jp, k]] - phi[[i, jm, k]]) * inv_2dx;
                    grad_z[[i, j, k]] = (phi[[i, j, kp]] - phi[[i, j, km]]) * inv_2dx;
                }
            }
        }

        (grad_x, grad_y, grad_z)
    };

    // Evolve each component (x, y, z) independently
    for comp in 0..3 {
        let current_comp = a_i.slice(ndarray::s![comp, .., .., ..]);
        let prev_comp = a_i_prev.slice(ndarray::s![comp, .., .., ..]);

        // 1. Laplacian of this component
        let laplacian = compute_laplacian_periodic(&current_comp, dx);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let curr = a_i[[comp, i, j, k]];
                    let prev = a_i_prev[[comp, i, j, k]];
                    let lap = laplacian[[i, j, k]];

                    // Calculate axial coupling term: g ∇φ × B
                    let axial_term = match comp {
                        // (∇φ × B)_x = ∂_y φ B_z - ∂_z φ B_y
                        0 => {
                            let b_y = compute_curl_component(&a_i.view(), 1, i, j, k, dx);
                            let b_z = compute_curl_component(&a_i.view(), 2, i, j, k, dx);
                            g * (phi_grad_y[[i, j, k]] * b_z - phi_grad_z[[i, j, k]] * b_y)
                        },
                        // (∇φ × B)_y = ∂_z φ B_x - ∂_x φ B_z
                        1 => {
                            let b_x = compute_curl_component(&a_i.view(), 0, i, j, k, dx);
                            let b_z = compute_curl_component(&a_i.view(), 2, i, j, k, dx);
                            g * (phi_grad_z[[i, j, k]] * b_x - phi_grad_x[[i, j, k]] * b_z)
                        },
                        // (∇φ × B)_z = ∂_x φ B_y - ∂_y φ B_x
                        2 => {
                            let b_x = compute_curl_component(&a_i.view(), 0, i, j, k, dx);
                            let b_y = compute_curl_component(&a_i.view(), 1, i, j, k, dx);
                            g * (phi_grad_x[[i, j, k]] * b_y - phi_grad_y[[i, j, k]] * b_x)
                        },
                        _ => 0.0,
                    };

                    // Gradient term: (1/a²)∇²A
                    let grad_term = lap * inv_a2;

                    // Mass term: -m_A² A
                    let mass_term = -m_a2 * curr;

                    // Axial coupling term: g ∇φ × B (divided by a³ for FLRW)
                    let axial_coupling = axial_term / (a * a * a);

                    let force = grad_term + mass_term + axial_coupling;

                    // Verlet step
                    a_next[[comp, i, j, k]] = (2.0 * curr - alpha * prev + dt2 * force) / denom;
                }
            }
        }
    }

    a_next
}

/// Calcula el tensor de energía-momento T_ij para el campo escalar φ
/// que es la fuente de ondas gravitatorias
/// T_ij ≈ ∂_i φ ∂_j φ - (1/2) δ_ij [∂_k φ ∂_k φ + m²φ²]
pub fn compute_stress_tensor_ij(
    phi: &Array3<f64>,
    phi_dot: &Array3<f64>,
    params: &CosmologicalParameters,
    dx: f64,
    a: f64,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let (nx, ny, nz) = phi.dim();
    let mut t_xx = Array3::zeros((nx, ny, nz));
    let mut t_xy = Array3::zeros((nx, ny, nz));
    let mut t_xz = Array3::zeros((nx, ny, nz));

    let inv_dx = 1.0 / dx;
    let m2 = params.m_phi * params.m_phi;
    let a2 = a * a;
    let inv_a2 = 1.0 / a2;  // Factor métrico para derivadas espaciales

    for i in 0..nx {
        let ip = if i == nx - 1 { 0 } else { i + 1 };
        let im = if i == 0 { nx - 1 } else { i - 1 };

        for j in 0..ny {
            let jp = if j == ny - 1 { 0 } else { j + 1 };
            let jm = if j == 0 { ny - 1 } else { j - 1 };

            for k in 0..nz {
                let kp = if k == nz - 1 { 0 } else { k + 1 };
                let km = if k == 0 { nz - 1 } else { k - 1 };

                // Derivadas espaciales
                let dphi_dx = (phi[[ip, j, k]] - phi[[im, j, k]]) * 0.5 * inv_dx;
                let dphi_dy = (phi[[i, jp, k]] - phi[[i, jm, k]]) * 0.5 * inv_dx;
                let dphi_dz = (phi[[i, j, kp]] - phi[[i, j, km]]) * 0.5 * inv_dx;

                // Derivada temporal
                let phi_dot_val = phi_dot[[i, j, k]];

                // Componentes del tensor de energía-momento (espacial)
                let kinetic = 0.5 * phi_dot_val * phi_dot_val;
                let gradient = 0.5 * inv_a2 * (dphi_dx*dphi_dx + dphi_dy*dphi_dy + dphi_dz*dphi_dz);
                let potential = 0.5 * m2 * phi[[i, j, k]] * phi[[i, j, k]];

                // Energía total (para traza)
                let energy_total = kinetic + gradient + potential;

                // T_ij = ∂_i φ ∂_j φ - (1/2) δ_ij [∂_k φ ∂_k φ + m²φ²]
                // Nota: simplificamos - usamos solo la parte anisotrópica
                t_xx[[i, j, k]] = dphi_dx * dphi_dx * inv_a2 - energy_total / 3.0;
                t_xy[[i, j, k]] = dphi_dx * dphi_dy * inv_a2;
                t_xz[[i, j, k]] = dphi_dx * dphi_dz * inv_a2;
            }
        }
    }

    (t_xx, t_xy, t_xz)
}

/// Calcula la densidad de energía para normalización
pub fn compute_energy_density(
    phi: &Array3<f64>,
    phi_dot: &Array3<f64>,
    params: &CosmologicalParameters,
    dx: f64,
    a: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = phi.dim();
    let mut rho = Array3::zeros((nx, ny, nz));

    let inv_dx = 1.0 / dx;
    let m2 = params.m_phi * params.m_phi;
    let a2 = a * a;
    let inv_a2 = 1.0 / a2;

    for i in 0..nx {
        let ip = if i == nx - 1 { 0 } else { i + 1 };
        let im = if i == 0 { nx - 1 } else { i - 1 };

        for j in 0..ny {
            let jp = if j == ny - 1 { 0 } else { j + 1 };
            let jm = if j == 0 { ny - 1 } else { j - 1 };

            for k in 0..nz {
                let kp = if k == nz - 1 { 0 } else { k + 1 };
                let km = if k == 0 { nz - 1 } else { k - 1 };

                let dphi_dx = (phi[[ip, j, k]] - phi[[im, j, k]]) * 0.5 * inv_dx;
                let dphi_dy = (phi[[i, jp, k]] - phi[[i, jm, k]]) * 0.5 * inv_dx;
                let dphi_dz = (phi[[i, j, kp]] - phi[[i, j, km]]) * 0.5 * inv_dx;

                let phi_val = phi[[i, j, k]];
                let phi_dot_val = phi_dot[[i, j, k]];

                // Densidad de energía del campo escalar
                rho[[i, j, k]] = 0.5 * phi_dot_val * phi_dot_val
                    + 0.5 * inv_a2 * (dphi_dx*dphi_dx + dphi_dy*dphi_dy + dphi_dz*dphi_dz)
                    + 0.5 * m2 * phi_val * phi_val;
            }
        }
    }

    rho
}

/// Calcula la parte anarmónica del tensor de energía-momento
/// que es la fuente de ondas gravitatorias
pub fn compute_anisotropic_stress(
    phi: &Array3<f64>,
    phi_dot: &Array3<f64>,
    params: &CosmologicalParameters,
    dx: f64,
    a: f64,
) -> Array3<f64> {
    let (nx, ny, nz) = phi.dim();
    let mut stress = Array3::zeros((nx, ny, nz));

    let inv_dx = 1.0 / dx;
    let a2 = a * a;
    let inv_a2 = 1.0 / a2;

    for i in 0..nx {
        let ip = if i == nx - 1 { 0 } else { i + 1 };
        let im = if i == 0 { nx - 1 } else { i - 1 };

        for j in 0..ny {
            let jp = if j == ny - 1 { 0 } else { j + 1 };
            let jm = if j == 0 { ny - 1 } else { j - 1 };

            for k in 0..nz {
                let kp = if k == nz - 1 { 0 } else { k + 1 };
                let km = if k == 0 { nz - 1 } else { k - 1 };

                // Derivadas espaciales
                let dphi_dx = (phi[[ip, j, k]] - phi[[im, j, k]]) * 0.5 * inv_dx;
                let dphi_dy = (phi[[i, jp, k]] - phi[[i, jm, k]]) * 0.5 * inv_dx;
                let dphi_dz = (phi[[i, j, kp]] - phi[[i, j, km]]) * 0.5 * inv_dx;

                // Magnitud cuadrada del gradiente
                let grad_sq = dphi_dx*dphi_dx + dphi_dy*dphi_dy + dphi_dz*dphi_dz;

                // Estimación de la anisotropía (parte TT del tensor)
                // Para ondas gravitatorias: Π_ij^TT ~ (∂_i φ ∂_j φ)^TT
                stress[[i, j, k]] = grad_sq * inv_a2;  // Simplificado
            }
        }
    }

    stress
              }
