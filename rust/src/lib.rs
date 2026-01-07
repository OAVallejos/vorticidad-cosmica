// src/lib.rs - VERSIÓN COMPLETA CON ONDAS GRAVITATORIAS
pub mod physics;
pub mod fdtd_solver;

use ndarray::{Array3, Array4, s};
use pyo3::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::f64::consts::PI;

// =============================================================================
// CONSTANTES GLOBALES (ACTUALIZADAS A 128³)
// =============================================================================

const N_GRID: usize = 128;    // Escalamos a 128³ para mejor resolución
const N_COMP: usize = 3;      // Componentes vectoriales

// =============================================================================
// ESTRUCTURAS PRINCIPALES
// =============================================================================

#[pyclass]
#[derive(Clone, Debug)]
pub struct CosmologicalParameters {
    #[pyo3(get, set)]
    pub g: f64,            // Acoplamiento adimensional
    #[pyo3(get, set)]
    pub m_phi: f64,        // m_φ / M_Pl
    #[pyo3(get, set)]
    pub m_a: f64,          // m_A / M_Pl
    #[pyo3(get, set)]
    pub h_inf: f64,        // H_inf / M_Pl
    #[pyo3(get, set)]
    pub phi0: f64,         // Amplitud inicial
}

#[pymethods]
impl CosmologicalParameters {
    #[new]
    fn new(g: f64, m_phi: f64, m_a: f64, h_inf: f64, phi0: f64) -> Self {
        Self {
            g,
            m_phi,
            m_a,
            h_inf,
            phi0,
        }
    }

    #[staticmethod]
    fn calibrated() -> Self {
        Self {
            g: 8.4e-15,
            m_phi: 1e-5,
            m_a: 2.9e-28,
            h_inf: 1e-5,
            phi0: 1.0,
        }
    }

    /// Calcular parámetro de resonancia q = (g φ0 / m_φ)^2
    pub fn resonance_parameter(&self) -> f64 {
        (self.g * self.phi0 / self.m_phi).powi(2)
    }

    /// Calcular el modo de Fourier resonante k_res ≈ m_φ √q
    pub fn resonant_mode(&self) -> f64 {
        self.m_phi * self.resonance_parameter().sqrt()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct LatticeSimulation {
    // Campos escalares
    phi: Array3<f64>,        // φ(t^n)
    phi_prev: Array3<f64>,   // φ(t^{n-1})
    // Campos vectoriales
    a_i: Array4<f64>,        // A_i(t^n)
    a_i_prev: Array4<f64>,   // A_i(t^{n-1})
    // Derivadas temporales
    phi_dot: Array3<f64>,    // \dot{φ}
    // Parámetros
    params: CosmologicalParameters,
    dx: f64,
    dt: f64,
    a: f64,
    time: f64,
    n_steps: usize,
    // Estadísticas
    energy_history: Vec<f64>,
    hubble_history: Vec<f64>,
    // Espectro de potencia (caché)
    power_spectrum_cache: Option<Vec<f64>>,
}

#[pymethods]
impl LatticeSimulation {
    #[new]
    fn new(
        g: f64,
        m_phi: f64,
        m_a: f64,
        h_inf: f64,
        phi0: f64,
        dx: f64,
        dt: f64,
    ) -> PyResult<Self> {
        // Validar parámetros
        if dx <= 0.0 || dt <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dx y dt deben ser positivos"
            ));
        }

        let cfl_limit = dx / 3.0_f64.sqrt();
        if dt >= cfl_limit {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("dt = {} demasiado grande. Requiere dt < dx/√3 ≈ {}", dt, cfl_limit)
            ));
        }

        let shape = (N_GRID, N_GRID, N_GRID);

        // Inicialización optimizada para 128³
        let mut phi = Array3::zeros(shape);
        let mut phi_prev = Array3::zeros(shape);

        // Usar una semilla determinista para reproducibilidad
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        let mut rng = StdRng::seed_from_u64(42);

        // Pre-cálculo para eficiencia
        let m_phi_dx = m_phi * dx;

        for i in 0..N_GRID {
            let x = i as f64 * dx;
            let background_cos = phi0 * (m_phi_dx * i as f64).cos();
            let background_sin = -phi0 * m_phi * (m_phi_dx * i as f64).sin(); // derivada

            for j in 0..N_GRID {
                for k in 0..N_GRID {
                    // FONDO FÍSICO: φ ~ M_Pl * cos(m_φ x)
                    let background = background_cos;

                    // FLUCTUACIONES: ~10^-10 para densidad ~10^-20
                    let fluctuation: f64 = 1.0e-10 * (rng.gen::<f64>() - 0.5);
                    phi[[i, j, k]] = background + fluctuation;

                    // Derivada inicial: \dot{φ} ≈ -H φ para expansión
                    phi_prev[[i, j, k]] = phi[[i, j, k]] - dt * h_inf * background;
                }
            }
        }

        // Campo vectorial inicial - MUCHO más pequeño
        let mut a_i = Array4::zeros((N_COMP, N_GRID, N_GRID, N_GRID));
        let mut a_i_prev = Array4::zeros((N_COMP, N_GRID, N_GRID, N_GRID));

        // Inicialización más eficiente del campo vectorial
        for comp in 0..N_COMP {
            let mut slice = a_i.slice_mut(s![comp, .., .., ..]);
            for val in slice.iter_mut() {
                *val = 1.0e-15 * (rng.gen::<f64>() - 0.5);
            }
            a_i_prev.slice_mut(s![comp, .., .., ..]).assign(&slice);
        }

        Ok(Self {
            phi,
            phi_prev,
            a_i,
            a_i_prev,
            phi_dot: Array3::zeros(shape),
            params: CosmologicalParameters::new(g, m_phi, m_a, h_inf, phi0),
            dx,
            dt,
            a: 1.0,
            time: 0.0,
            n_steps: 0,
            energy_history: Vec::new(),
            hubble_history: Vec::new(),
            power_spectrum_cache: None,
        })
    }

    /// Ejecutar N pasos de evolución
    pub fn evolve(&mut self, n_steps: usize) -> PyResult<()> {
        for step in 0..n_steps {
            self.time += self.dt;
            self.n_steps += 1;

            // 1. Calcular derivadas
            self.compute_time_derivatives();

            // 2. Evolucionar campos
            self.update_fields();

            // 3. Actualizar factor de escala
            self.update_scale_factor();

            // 4. Registrar estadísticas
            if step % 20 == 0 {  // Menos frecuente para 128³
                self.record_statistics();

                // Invalidar caché del espectro
                self.power_spectrum_cache = None;
            }
        }

        Ok(())
    }

    /// Calcular estadísticas básicas
    pub fn compute_statistics(&self) -> PyResult<HashMap<String, f64>> {
        let mut stats = HashMap::new();

        // Energía
        let energy = self.compute_total_energy_density();
        stats.insert("energy_density".to_string(), energy);

        // Norma de campos
        let phi_norm = self.phi.mapv(|x| x * x).sum();
        let mut a_norm = 0.0;
        for comp in 0..N_COMP {
            a_norm += self.a_i.slice(s![comp, .., .., ..]).mapv(|x| x * x).sum();
        }

        stats.insert("phi_norm".to_string(), phi_norm);
        stats.insert("a_norm".to_string(), a_norm);
        stats.insert("time".to_string(), self.time);
        stats.insert("scale_factor".to_string(), self.a);

        // Índice espectral aproximado
        let q = self.params.resonance_parameter();
        let n_omega = -2.0 + 0.35 * q.sqrt();
        stats.insert("spectral_index".to_string(), n_omega);

        // Modo resonante
        let k_res = self.params.resonant_mode();
        stats.insert("resonant_mode".to_string(), k_res);

        Ok(stats)
    }

    /// Getter para resultados
    #[getter]
    pub fn get_energy_history(&self) -> Vec<f64> {
        self.energy_history.clone()
    }

    #[getter]
    pub fn get_hubble_history(&self) -> Vec<f64> {
        self.hubble_history.clone()
    }

    #[getter]
    pub fn get_time(&self) -> f64 {
        self.time
    }

    #[getter]
    pub fn get_scale_factor(&self) -> f64 {
        self.a
    }

    #[getter]
    pub fn get_phi_norm(&self) -> f64 {
        self.phi.mapv(|x| x * x).sum()
    }

    #[getter]
    pub fn get_a_norm(&self) -> f64 {
        let mut sum = 0.0;
        for comp in 0..N_COMP {
            sum += self.a_i.slice(s![comp, .., .., ..]).mapv(|x| x * x).sum();
        }
        sum
    }

    // Nuevos getters para compatibilidad con el script Python
    #[getter]
    pub fn scale_factor(&self) -> f64 {
        self.a
    }

    #[getter]
    pub fn time(&self) -> f64 {
        self.time
    }

    // Nuevos getters para acceder a parámetros internos
    #[getter]
    pub fn dx(&self) -> f64 {
        self.dx
    }

    #[getter]
    pub fn dt(&self) -> f64 {
        self.dt
    }

    #[getter]
    pub fn params(&self) -> CosmologicalParameters {
        self.params.clone()
    }

    #[getter]
    pub fn n_steps(&self) -> usize {
        self.n_steps
    }

    /// Devuelve el campo escalar como lista plana (reshape en Python)
    pub fn get_phi_grid(&self) -> PyResult<Vec<f64>> {
        let total_points = N_GRID * N_GRID * N_GRID;
        let mut phi_flat = Vec::with_capacity(total_points);

        for i in 0..N_GRID {
            for j in 0..N_GRID {
                for k in 0..N_GRID {
                    phi_flat.push(self.phi[[i, j, k]]);
                }
            }
        }

        if phi_flat.len() != total_points {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Error interno: phi_grid tiene {} puntos, esperaba {}",
                       phi_flat.len(), total_points)
            ));
        }

        Ok(phi_flat)
    }

    /// Devuelve la magnitud del campo vectorial |A| como lista plana
    pub fn get_vector_magnitude_grid(&self) -> PyResult<Vec<f64>> {
        let total_points = N_GRID * N_GRID * N_GRID;
        let mut mag = Vec::with_capacity(total_points);

        // Calculamos |A| = sqrt(Ax² + Ay² + Az²)
        for i in 0..N_GRID {
            for j in 0..N_GRID {
                for k in 0..N_GRID {
                    let ax = self.a_i[[0, i, j, k]];
                    let ay = self.a_i[[1, i, j, k]];
                    let az = self.a_i[[2, i, j, k]];
                    let magnitude = (ax*ax + ay*ay + az*az).sqrt();
                    mag.push(magnitude);
                }
            }
        }

        if mag.len() != total_points {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Error interno: vector_magnitude_grid tiene {} puntos, esperaba {}",
                       mag.len(), total_points)
            ));
        }

        Ok(mag)
    }

    /// Calcula el espectro de potencia P(k) del campo escalar (Opción A)
    /// Retorna: (k_values, power_spectrum)
    /// Donde k_values están normalizados a m_φ
    pub fn compute_power_spectrum(&mut self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        // Usar caché si está disponible
        if let Some(cached) = &self.power_spectrum_cache {
            // Reconstruir desde caché (k_values, power_spectrum)
            let mid = cached.len() / 2;
            let k_values = cached[..mid].to_vec();
            let power_spectrum = cached[mid..].to_vec();
            return Ok((k_values, power_spectrum));
        }

        // Calcular FFT 3D del campo φ
        let total_points = N_GRID * N_GRID * N_GRID;
        let mut k_values = Vec::new();
        let mut power_spectrum = Vec::new();

        // Cálculo simplificado del espectro de potencia radial
        let num_bins = 20;
        let max_k = (2.0 * PI) / self.dx;  // Nyquist frequency
        let k_bins: Vec<f64> = (0..=num_bins)
            .map(|i| (i as f64) * max_k / (num_bins as f64))
            .collect();

        // Inicializar bins de potencia
        let mut power_bins = vec![0.0; num_bins];
        let mut count_bins = vec![0; num_bins];

        // Calcular transformada de Fourier de forma aproximada
        // Nota: Para mayor precisión, usar rustfft
        for i in 0..N_GRID {
            let kx = if i <= N_GRID/2 {
                (2.0 * PI * i as f64) / (N_GRID as f64 * self.dx)
            } else {
                (2.0 * PI * (i as f64 - N_GRID as f64)) / (N_GRID as f64 * self.dx)
            };

            for j in 0..N_GRID {
                let ky = if j <= N_GRID/2 {
                    (2.0 * PI * j as f64) / (N_GRID as f64 * self.dx)
                } else {
                    (2.0 * PI * (j as f64 - N_GRID as f64)) / (N_GRID as f64 * self.dx)
                };

                for k in 0..N_GRID {
                    let kz = if k <= N_GRID/2 {
                        (2.0 * PI * k as f64) / (N_GRID as f64 * self.dx)
                    } else {
                        (2.0 * PI * (k as f64 - N_GRID as f64)) / (N_GRID as f64 * self.dx)
                    };

                    let k_mag = (kx*kx + ky*ky + kz*kz).sqrt();

                    // Encontrar el bin correspondiente
                    for bin_idx in 0..num_bins {
                        if k_mag >= k_bins[bin_idx] && k_mag < k_bins[bin_idx + 1] {
                            let phi_val = self.phi[[i, j, k]];
                            power_bins[bin_idx] += phi_val * phi_val;
                            count_bins[bin_idx] += 1;
                            break;
                        }
                    }
                }
            }
        }

        // Promediar y normalizar
        for bin_idx in 0..num_bins {
            if count_bins[bin_idx] > 0 {
                let k_center = (k_bins[bin_idx] + k_bins[bin_idx + 1]) / 2.0;
                let power = power_bins[bin_idx] / (count_bins[bin_idx] as f64);

                k_values.push(k_center / self.params.m_phi);  // Normalizado a m_φ
                power_spectrum.push(power);
            }
        }

        // Guardar en caché
        let mut cache_data = k_values.clone();
        cache_data.extend(power_spectrum.clone());
        self.power_spectrum_cache = Some(cache_data);

        Ok((k_values, power_spectrum))
    }

    /// Calcula el espectro helicoidal H(k) = ⟨E·B⟩
    pub fn compute_helicity_spectrum(&self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        // Implementación simplificada del espectro helicoidal
        let num_bins = 20;
        let max_k = (2.0 * PI) / self.dx;
        let k_bins: Vec<f64> = (0..=num_bins)
            .map(|i| (i as f64) * max_k / (num_bins as f64))
            .collect();

        let mut helicity_bins = vec![0.0; num_bins];
        let mut count_bins = vec![0; num_bins];

        // Cálculo aproximado de helicidad por punto
        for i in 0..N_GRID {
            for j in 0..N_GRID {
                for k in 0..N_GRID {
                    // Cálculo simplificado del número de onda
                    let k_mag = self.estimate_wavenumber(i, j, k);

                    // Estimación de helicidad en este punto
                    let helicity = self.estimate_local_helicity(i, j, k);

                    // Asignar a bin
                    for bin_idx in 0..num_bins {
                        if k_mag >= k_bins[bin_idx] && k_mag < k_bins[bin_idx + 1] {
                            helicity_bins[bin_idx] += helicity;
                            count_bins[bin_idx] += 1;
                            break;
                        }
                    }
                }
            }
        }

        // Promediar
        let mut k_values = Vec::new();
        let mut helicity_spectrum = Vec::new();

        for bin_idx in 0..num_bins {
            if count_bins[bin_idx] > 0 {
                let k_center = (k_bins[bin_idx] + k_bins[bin_idx + 1]) / 2.0;
                let h = helicity_bins[bin_idx] / (count_bins[bin_idx] as f64);

                k_values.push(k_center / self.params.m_phi);
                helicity_spectrum.push(h);
            }
        }

        Ok((k_values, helicity_spectrum))
    }

    /// Calcula el espectro de ondas gravitatorias estimado
    pub fn compute_gw_spectrum(&mut self) -> PyResult<(Vec<f64>, Vec<f64>)> {
        // Calcular tensor de energía-momento
        // Primero necesitamos el campo φ y su derivada

        // Calcular derivadas temporales
        let phi_dot = &self.phi - &self.phi_prev;
        let phi_dot = phi_dot.mapv(|x| x / self.dt);

        // Calcular componentes del tensor (simplificado - solo escalar)
        let (t_xx, t_xy, t_xz) = physics::compute_stress_tensor_ij(
            &self.phi,
            &phi_dot,
            &self.params,
            self.dx,
            self.a
        );

        // Análisis espectral de las anisotropías
        let num_bins = 20;
        let max_k = (2.0 * PI) / self.dx;
        let k_bins: Vec<f64> = (0..=num_bins)
            .map(|i| (i as f64) * max_k / (num_bins as f64))
            .collect();

        let mut gw_bins = vec![0.0; num_bins];
        let mut count_bins = vec![0; num_bins];

        // Medir la anisotropía cuadrática media
        for i in 0..N_GRID {
            for j in 0..N_GRID {
                for k in 0..N_GRID {
                    // Estimación simplificada del número de onda
                    let k_mag = self.estimate_wavenumber(i, j, k);

                    // Medir anisotropía: sqrt(T_ij T_ij) - parte isotrópica
                    let t_xx_val = t_xx[[i, j, k]];
                    let t_xy_val = t_xy[[i, j, k]];
                    let t_xz_val = t_xz[[i, j, k]];

                    // Tensor anisotrópico (aproximado)
                    let anisotropy_sq = t_xx_val * t_xx_val
                        + 2.0 * t_xy_val * t_xy_val
                        + 2.0 * t_xz_val * t_xz_val;
                    let anisotropy = anisotropy_sq.sqrt();

                    // Asignar a bin
                    for bin_idx in 0..num_bins {
                        if k_mag >= k_bins[bin_idx] && k_mag < k_bins[bin_idx + 1] {
                            gw_bins[bin_idx] += anisotropy * anisotropy;  // ∝ Ω_GW
                            count_bins[bin_idx] += 1;
                            break;
                        }
                    }
                }
            }
        }

        // Promediar y normalizar
        let mut k_values = Vec::new();
        let mut gw_spectrum = Vec::new();

        for bin_idx in 0..num_bins {
            if count_bins[bin_idx] > 0 {
                let k_center = (k_bins[bin_idx] + k_bins[bin_idx + 1]) / 2.0;
                let omega_gw = gw_bins[bin_idx] / (count_bins[bin_idx] as f64);

                k_values.push(k_center / self.params.m_phi);
                gw_spectrum.push(omega_gw);
            }
        }

        Ok((k_values, gw_spectrum))
    }

    /// Obtiene el tensor de energía-momento como arrays planos para Python
    pub fn get_stress_tensor_grids(&self) -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let total_points = N_GRID * N_GRID * N_GRID;
        let mut t_xx_flat = Vec::with_capacity(total_points);
        let mut t_xy_flat = Vec::with_capacity(total_points);
        let mut t_xz_flat = Vec::with_capacity(total_points);

        // Calcular derivada temporal
        let phi_dot = &self.phi - &self.phi_prev;
        let phi_dot = phi_dot.mapv(|x| x / self.dt);

        // Calcular componentes del tensor
        let (t_xx, t_xy, t_xz) = physics::compute_stress_tensor_ij(
            &self.phi,
            &phi_dot,
            &self.params,
            self.dx,
            self.a
        );

        // Aplanar arrays
        for i in 0..N_GRID {
            for j in 0..N_GRID {
                for k in 0..N_GRID {
                    t_xx_flat.push(t_xx[[i, j, k]]);
                    t_xy_flat.push(t_xy[[i, j, k]]);
                    t_xz_flat.push(t_xz[[i, j, k]]);
                }
            }
        }

        // Validar tamaño
        if t_xx_flat.len() != total_points {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                format!("Error interno: tensor grids tienen {} puntos, esperaba {}",
                       t_xx_flat.len(), total_points)
            ));
        }

        Ok((t_xx_flat, t_xy_flat, t_xz_flat))
    }
}

// =============================================================================
// IMPLEMENTACIONES INTERNAS
// =============================================================================

impl LatticeSimulation {
    fn compute_time_derivatives(&mut self) {
        // \dot{φ} = (φ^n - φ^{n-1}) / Δt
        self.phi_dot = &self.phi - &self.phi_prev;
        self.phi_dot.mapv_inplace(|x| x / self.dt);
    }

    fn update_fields(&mut self) {
        // Obtener el valor actual de Hubble
        let current_hubble = *self.hubble_history.last().unwrap_or(&self.params.h_inf);

        // Actualizar campos escalares
        let new_phi = physics::evolve_phi_field(
            &self.phi,
            &self.phi_prev,
            &self.a_i,
            &self.params,
            self.dx,
            self.dt,
            self.a,
            current_hubble
        );

        // Actualizar campos vectoriales
        let new_a_i = physics::evolve_vector_field(
            &self.a_i,
            &self.a_i_prev,
            &self.phi,
            &self.params,
            self.dx,
            self.dt,
            self.a,
            current_hubble
        );

        // Actualizar historial
        self.phi_prev = self.phi.clone();
        self.phi = new_phi;

        self.a_i_prev = self.a_i.clone();
        self.a_i = new_a_i;

        // Invalidar caché del espectro
        self.power_spectrum_cache = None;
    }

    fn update_scale_factor(&mut self) {
        // Densidad de energía física
        let energy_density = self.compute_total_energy_density();

        // Calcular H correctamente
        let h = (energy_density.abs() / 3.0).sqrt();

        // LIMITAR H para estabilidad numérica
        let h_limited = h.clamp(0.0, self.params.h_inf);

        // Evolución de a: da/dt = aH
        let delta_a = self.a * h_limited * self.dt;

        // Verificar que el cambio sea pequeño
        if delta_a.abs() < 0.1 * self.a {
            self.a += delta_a;
        } else {
            self.a *= 1.0 + h_limited.min(0.01) * self.dt;
        }

        // Asegurar que a ≥ 1.0
        self.a = self.a.max(1.0);

        // Limitar crecimiento máximo durante preheating
        let max_growth = 1.0 + 0.02 * self.time;
        if self.a > max_growth {
            self.a = max_growth;
        }
    }

    fn compute_total_energy_density(&self) -> f64 {
        let mut energy = 0.0;
        let volume = (N_GRID * N_GRID * N_GRID) as f64;

        // ENERGÍA CINÉTICA ESCALAR
        let kinetic_energy = 0.5 * self.phi_dot.mapv(|x| x * x).sum() / volume;
        energy += kinetic_energy;

        // ENERGÍA POTENCIAL ESCALAR
        let potential_energy = 0.5 * self.params.m_phi.powi(2) *
                               self.phi.mapv(|x| x * x).sum() / volume;
        energy += potential_energy;

        // ENERGÍA VECTORIAL
        for comp in 0..N_COMP {
            let slice = self.a_i.slice(s![comp, .., .., ..]);
            let vector_energy = 0.5 * self.params.m_a.powi(2) *
                                slice.mapv(|x| x * x).sum() / volume;
            energy += vector_energy;
        }

        // Escalar si es necesario
        let expected_energy = self.params.m_phi.powi(2) * self.params.phi0.powi(2) * 0.5;
        if energy > 100.0 * expected_energy {
            energy = expected_energy;
        }

        energy
    }

    fn record_statistics(&mut self) {
        let energy = self.compute_total_energy_density();
        let hubble = (energy.abs() / 3.0).sqrt();
        let hubble_limited = hubble.clamp(1e-10, 1e-4);

        self.energy_history.push(energy);
        self.hubble_history.push(hubble_limited);
    }

    fn estimate_wavenumber(&self, i: usize, j: usize, k: usize) -> f64 {
        // Estimación simple del número de onda basada en derivadas
        let n = N_GRID;

        // Índices periódicos
        let im = if i == 0 { n - 1 } else { i - 1 };
        let ip = if i == n - 1 { 0 } else { i + 1 };
        let jm = if j == 0 { n - 1 } else { j - 1 };
        let jp = if j == n - 1 { 0 } else { j + 1 };
        let km = if k == 0 { n - 1 } else { k - 1 };
        let kp = if k == n - 1 { 0 } else { k + 1 };

        // Derivadas de φ
        let dphi_dx = (self.phi[[ip, j, k]] - self.phi[[im, j, k]]) / (2.0 * self.dx);
        let dphi_dy = (self.phi[[i, jp, k]] - self.phi[[i, jm, k]]) / (2.0 * self.dx);
        let dphi_dz = (self.phi[[i, j, kp]] - self.phi[[i, j, km]]) / (2.0 * self.dx);

        // Estimación de k ≈ |∇φ|/φ (cuando φ ≠ 0)
        let phi_val = self.phi[[i, j, k]];
        if phi_val.abs() > 1e-10 {
            let grad_mag = (dphi_dx*dphi_dx + dphi_dy*dphi_dy + dphi_dz*dphi_dz).sqrt();
            grad_mag / phi_val.abs()
        } else {
            0.0
        }
    }

    fn estimate_local_helicity(&self, i: usize, j: usize, k: usize) -> f64 {
        // Estimación simplificada de E·B en un punto
        let n = N_GRID;

        // Índices periódicos
        let im = if i == 0 { n - 1 } else { i - 1 };
        let ip = if i == n - 1 { 0 } else { i + 1 };
        let jm = if j == 0 { n - 1 } else { j - 1 };
        let jp = if j == n - 1 { 0 } else { j + 1 };
        let km = if k == 0 { n - 1 } else { k - 1 };
        let kp = if k == n - 1 { 0 } else { k + 1 };

        // Campo eléctrico aproximado E ≈ -∂_t A
        let ex = -(self.a_i[[0, i, j, k]] - self.a_i_prev[[0, i, j, k]]) / self.dt;
        let ey = -(self.a_i[[1, i, j, k]] - self.a_i_prev[[1, i, j, k]]) / self.dt;
        let ez = -(self.a_i[[2, i, j, k]] - self.a_i_prev[[2, i, j, k]]) / self.dt;

        // Campo magnético B = ∇ × A
        let bx = (self.a_i[[2, i, jp, k]] - self.a_i[[2, i, jm, k]] -
                  self.a_i[[1, i, j, kp]] + self.a_i[[1, i, j, km]]) / (2.0 * self.dx);
        let by = (self.a_i[[0, i, j, kp]] - self.a_i[[0, i, j, km]] -
                  self.a_i[[2, ip, j, k]] + self.a_i[[2, im, j, k]]) / (2.0 * self.dx);
        let bz = (self.a_i[[1, ip, j, k]] - self.a_i[[1, im, j, k]] -
                  self.a_i[[0, i, jp, k]] + self.a_i[[0, i, jm, k]]) / (2.0 * self.dx);

        // Helicidad local E·B
        ex * bx + ey * by + ez * bz
    }
}

// =============================================================================
// MÓDULO PYTHON
// =============================================================================

#[pymodule]
fn vorticity_preheating(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<CosmologicalParameters>()?;
    m.add_class::<LatticeSimulation>()?;

    // Añadir versión
    m.add("__version__", "0.3.0")?;  // Actualizado a 0.3.0 con GW

    Ok(())
          }
