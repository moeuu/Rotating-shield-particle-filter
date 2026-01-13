# Online Radiation Source Term Estimation Using a Particle Filter with Rotating Shields
<a id="chap:chap3"></a>

<!-- ================================================== % -->
<!-- section -->
<!-- ================================================== % -->
## Introduction
<a id="chap_pf_introduction"></a>

In this chapter, I propose an online method for estimating the three-dimensional distribution of
multiple $\gamma$-ray sources using a particle filter (PF) combined with actively rotating shields.
A mobile robot equipped with an energy-resolving, non-directional detector and lightweight iron
and lead shields moves in a high-dose indoor environment and acquires radiation measurements
while continuously changing the shield orientations. By exploiting geometric spreading and
controlled attenuation due to the shields, the method recovers pseudo-directional information
from non-directional measurements and estimates the locations and strengths of multiple $\gamma$-ray
sources.

Section~\ref{chap3_pf_model} introduces the physical and mathematical model of non-directional count measurements with a shielded detector and discusses the design of lightweight lead shielding under robot payload constraints.

Section~\ref{chap3_pf_pf} formulates multi-isotope source-term estimation as Bayesian inference with parallel PFs, including the definition of precomputed geometric and shielding kernels, state representation, prediction, log-domain weight update for Poisson observations, resampling, regularization (including birth/death moves), and spurious-source rejection.

Section~\ref{chap3_pf_rotation_section} presents the shield-rotation strategy, which selects informative shield orientations using information-gain-based criteria and then selects the next robot pose by trading off expected uncertainty reduction and motion cost together with short-time measurements.

Finally, Section~\ref{chap3_summary} presents a summary of this chapter.



<!-- ================================================== % -->
<!-- section -->
<!-- ================================================== % -->
## Measurement Model and Shield Design
<a id="chap3_pf_model"></a>

### Macroscopic Attenuation and Shield Thickness
<a id="chap3_shield"></a>

Although $\gamma$ rays are electromagnetic waves, their high energy also gives them particle-like properties.
A macroscopic view of the interaction between $\gamma$ rays and matter considers an incident beam on a flat slab of material, as illustrated in Fig.~\ref{fig:chap2_attenuation}.
When $\gamma$ rays are incident on the material, some are absorbed and some undergo scattering, changing their direction and energy.

Consider a thin layer of thickness $dX$ inside the material, through which $N$ $\gamma$ rays are passing.
Let $N'$ be the number of $\gamma$ rays that pass through this layer without interaction.
The number of $\gamma$ rays that interact in the layer, $-dN = N - N'$, is proportional to both the thickness $dX$ of the layer and the number $N$ of incident $\gamma$ rays.
Using the proportionality constant $\mu$, this relationship can be written as
<a id="eq:chap2_attenuation_differential"></a>
$$
        -dN = \mu \, N \, dX ,
$$
where $\mu$ is called the linear attenuation coefficient and represents the ease with which interactions occur.

If $N_{0}$ is the number of $\gamma$ rays incident on the slab, solving Eq.~(\ref{eq:chap2_attenuation_differential}) yields
<a id="eq:chap2_attenuation"></a>
$$
        N = N_{0} e^{-\mu X} .
$$

The radiation dose measured by a detector obeys the inverse-square law with respect to the distance between a point source and the detector~\cite{ref_Tsoulfanidis1995}.
Let the detector position at the $k$-th measurement be
$$
\bm{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^{\top},
$$
and let a point source of intensity $q_j$ be located at
$$
\bm{s}_j = [x_j, y_j, z_j]^{\top}.
$$

Defining the distance
$$
    d_{k,j} = \sqrt{(x_k^{\mathrm{det}}-x_j)^{2} + (y_k^{\mathrm{det}}-y_j)^{2} + (z_k^{\mathrm{det}}-z_j)^{2}},
$$
the radiation intensity $I(\bm{q}_k)$ measured at $\bm{q}_k$ is given by
<a id="eq:chap2_inverse_square_law"></a>
$$
        I(\bm{q}_k) = \frac{S q_{j}}{4\pi d_{k,j}^{2}} e^{-\mu_{\mathrm{air}} d_{k,j}} ,
$$
where $S$ is the detector area and $\mu_{\mathrm{air}}$ is the linear attenuation coefficient of air.
For the $\gamma$ rays emitted by \ce{^{137}Cs}, the linear attenuation coefficient in air at \SI{20}{^\circ C} is approximately $9.7\times10^{-3}$~\si{\per\metre}~\cite{ref_Tsoulfanidis1995}.
In this study, measurements are performed relatively close to the radiation sources, and thus I approximate $e^{-\mu_{\mathrm{air}} d_{k,j}} \approx 1$ and neglect attenuation in air.
I also neglect attenuation by environmental obstacles, such as walls and equipment, and explicitly model only the attenuation due to the lightweight shields.

The thickness of a shielding material required to reduce the dose rate by half is called the half-value layer, and the thickness required to reduce the dose rate to one-tenth is called the tenth-value layer.
Table~\ref{table:half_shield} lists the half-value and tenth-value layers of lead, iron, and concrete for $\gamma$ rays emitted by \ce{^{137}Cs}~\cite{ref_ICRP21}.

<a id="fig:chap2_attenuation"></a>
![Interaction of $\gamma$ rays with matter.](Figures/chap3/attenuation.pdf)

<a id="table:half_shield"></a>
*Shield thickness and gamma-ray attenuation (unit: mm)*

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">Lead</th>
      <th colspan="2">Iron</th>
    </tr>
    <tr>
      <th></th>
      <th>Half-value layer (HVL)</th>
      <th>Tenth-value layer (TVL)</th>
      <th>Half-value layer (HVL)</th>
      <th>Tenth-value layer (TVL)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Cs-137</td>
      <td>7.0</td>
      <td>22.0</td>
      <td>15.0</td>
      <td>50.0</td>
    </tr>
    <tr>
      <td>Co-60</td>
      <td>12.0</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>Eu-154</td>
      <td>7.4</td>
      <td>24.6</td>
      <td>13.8</td>
      <td>45.8</td>
    </tr>
  </tbody>
</table>


### Mathematical Model of Non-directional Count Measurements
<a id="chap3_radation"></a>

I assume that $M \geq 1$ unknown radiation point sources exist in the environment.
Let the position of the $j$-th source be $\bm{s}_j = [x_{j}, y_{j}, z_{j}]^{\top}$ and its strength be $q_{j} \ge 0$.
The radiation source distribution is represented by the vector
<a id="eq:chap3_thetaj"></a>
$$
        \bm{q} = (q_{1}, q_{2}, \dots, q_{M})^{\top} .
$$

A non-directional detector mounted on the robot acquires $N_{\mathrm{meas}}$ measurements at positions
$$
\bm{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^{\top},
\qquad k = 1,\dots,N_{\mathrm{meas}}.
$$

Using the distance $d_{k,j}$ defined in Eq.~(\ref{eq:chap2_inverse_square_law}), and neglecting attenuation in air and environmental obstacles, the inverse-square law implies that the contribution of a unit-strength source at $\bm{s}_j$ to the detector at pose $\bm{q}_k$ is proportional to $1/d_{k,j}^{2}$.

I absorb the detector area, acquisition time, and conversion factors between source strength and count rate into a single constant $\Gamma$ and define
<a id="eq:chap3_bij"></a>
$$
        A_{k,j} = \frac{\Gamma}{d_{k,j}^{2}} ,
$$
which represents the expected count at pose $k$ from a unit-strength source at source $j$.

The expected total count at pose $k$ due to the full source distribution $\bm{q}$ is then
<a id="eq:chap3_bi"></a>
$$
        \Lambda_{k}(\bm{q}) = \sum_{j=1}^{M} A_{k,j} q_{j} .
$$

Collecting all measurement poses, I define the vector of expected counts
<a id="eq:chap3_bq"></a>
$$
        \boldsymbol{\Lambda}(\bm{q})
        = [\Lambda_{1}(\bm{q}), \Lambda_{2}(\bm{q}), \dots, \Lambda_{N_{\mathrm{meas}}}(\bm{q})]^{\top}.
$$

Let $\bm{A} \in \mathbb{R}^{N_{\mathrm{meas}}\times M}$ be the matrix with elements $A_{k,j}$.
In matrix form, the measurement model can be written compactly as
$$
\bm{A} =
\begin{pmatrix}
  A_{1,1} & \dots & A_{1,M} \\
  \vdots  &       & \vdots  \\
  A_{N_{\mathrm{meas}},1} & \dots & A_{N_{\mathrm{meas}},M}
\end{pmatrix},
$$
and
<a id="eq:chap3_b_Aq"></a>
$$
        \boldsymbol{\Lambda}(\bm{q}) = \bm{A}\bm{q} .
$$

As discussed in Section~\ref{chap2_radiation}, radiation is emitted by stochastic radioactive decay processes, and the number of counts recorded by the detector follows a Poisson distribution~\cite{ref_Tsoulfanidis1995}.
Let $z_{k}$ denote the observed count at the $k$-th measurement.
The likelihood of observing $z_{k}$ given the source distribution $\bm{q}$ is
<a id="eq:chap3_equation4"></a>
$$
    p(z_{k} \mid \bm{q})
      = \frac{\Lambda_{k}(\bm{q})^{\,z_{k}} \exp\!\big(-\Lambda_{k}(\bm{q})\big)}{z_{k}!} .
$$

This single-isotope, count-only Poisson model is extended in Section~\ref{subsec:spectrum_isotope_counts} and in Section~\ref{chap3_pf_pf} to isotope-wise counts $z_{k,h}$ and shield-dependent kernels $K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$.


### Shield Mass and Robot Payload
<a id="chap3_shield_weight"></a>

The densities of candidate shielding materials at \SI{20}{^\circ C} are listed in Table~\ref{table:density}.
For a given attenuation level (e.g., a specified number of half-value or tenth-value layers), the required shield mass is largely governed by the areal density (mass per unit area) and thus does not strongly depend on the material: denser materials require thinner shields, whereas less dense materials require thicker shields.
Accordingly, when mounting a shield on a mobile robot with a limited payload capacity, materials with high density are advantageous because they can achieve the required attenuation with smaller thickness.

In this study, lead and iron are selected as the shielding materials and are used *simultaneously*.
Specifically, both materials are fabricated as lightweight one-eighth spherical shells (1/8-shells) that partially cover a non-directional detector.
This configuration provides sufficient attenuation while keeping the total shield mass within the payload constraints of the mobile robot, making the system feasible for deployment in real environments.

Figure~\ref{fig:chap3_detector_shield} illustrates the detector and shield configuration assumed in this chapter.
The non-directional detector is mounted at the center of the assembly, and two partial 1/8 spherical shells---one made of lead and the other made of iron---surround the detector.
During operation, these shells are actively rotated around the detector so that the attenuation factor in \eqref{eq:pf_shield_attn} varies with time.

<!-- -------------------------------------------------- % -->
<!-- Payload feasibility check (robot + shield mass) -->
<!-- -------------------------------------------------- % -->

#### Representative robot platform and payload constraint

To make the payload constraint explicit, we consider a representative compact UGV, the Clearpath
*Jackal*. According to the manufacturer specifications, Jackal can carry a maximum payload
of \SI{20}{kg}.\cite{ref_clearpath_jackal}
This value is used as a practical reference when designing the detector--shield module
(detector head, digitizer, shields, rotation mechanism, and mounting structure).

#### Detector envelope and shield geometry

As a robot-mountable energy-resolving spectrometer, we assume a compact \ce{CeBr3} scintillation detector.
Commercial \ce{CeBr3} detectors are available up to \SI{102}{mm} in diameter and \SI{127}{mm} in length.\cite{ref_scionix_cebr3}
To enclose a cylindrical detector of diameter $D$ and length $L$ inside a spherical shield assembly,
the minimum inner radius satisfies
$$
  R_{\mathrm{in}} \ge \frac{1}{2}\sqrt{D^{2}+L^{2}} .
$$
Using the above maximum dimensions ($D=\SI{102}{mm}$, $L=\SI{127}{mm}$) gives
$R_{\mathrm{in}} \ge \SI{81.5}{mm}$.
In the following calculation, we conservatively set $R_{\mathrm{in}}=\SI{85}{mm}$ to account for the detector
housing and mechanical clearance.

#### Mass of 1/8 spherical shields for a tenth-value layer of \ce{^{137}Cs}

For the dominant \SI{662}{keV} $\gamma$ ray of \ce{^{137}Cs}, the tenth-value layer (TVL) thicknesses
are $X_{\mathrm{Pb}}=\SI{22}{mm}$ and $X_{\mathrm{Fe}}=\SI{50}{mm}$ (Table~\ref{table:half_shield}), and the material densities are
$\rho_{\mathrm{Pb}}=\SI{11.36}{g/cm^{3}}$ and $\rho_{\mathrm{Fe}}=\SI{7.87}{g/cm^{3}}$
(Table~\ref{table:density}).
The mass of a one-eighth spherical shell (an octant) of inner radius $R_{\mathrm{in}}$ and thickness $X$ is
<a id="eq:octant_shell_mass"></a>
$$
  m(\rho,R_{\mathrm{in}},X)
  = \rho \frac{\pi}{6}\left[(R_{\mathrm{in}}+X)^{3}-R_{\mathrm{in}}^{3}\right],
$$
where $\rho$ is in \si{kg/m^{3}} and $R_{\mathrm{in}},X$ are in \si{m}.
Converting the densities in Table~\ref{table:density} to SI units yields
$\rho_{\mathrm{Pb}}=\SI{11360}{kg/m^{3}}$ and $\rho_{\mathrm{Fe}}=\SI{7870}{kg/m^{3}}$.
Substituting $R_{\mathrm{in}}=\SI{85}{mm}$, $X_{\mathrm{Pb}}=\SI{22}{mm}$, and $X_{\mathrm{Fe}}=\SI{50}{mm}$ into
\eqref{eq:octant_shell_mass} gives
$$
\begin{aligned}
  m_{\mathrm{Pb}} &\approx \SI{3.63}{kg}, &
  m_{\mathrm{Fe}} &\approx \SI{7.61}{kg}, &
  m_{\mathrm{shields}} &\approx \SI{11.24}{kg}.
\end{aligned}
$$

Therefore, even a conservative TVL design for \ce{^{137}Cs} results in a total shield mass of
approximately \SI{11}{kg}, which is below the \SI{20}{kg} maximum payload of Jackal.\cite{ref_clearpath_jackal}
This leaves roughly \SI{9}{kg} of payload margin for the detector head, a compact digitizer
(e.g., CAEN DT5730, \SI{670}{g}),\cite{ref_caen_dt5730}
the rotation mechanism, and mounting hardware, supporting the feasibility of the proposed
robot-mountable rotating-shield configuration.

<a id="fig:chap3_detector_shield"></a>
![Configuration of the non-directional detector and lightweight shields used in this thesis.](Figures/chap3/Detector.pdf)

*Configuration of the non-directional detector and lightweight shields used in this thesis.
A compact non-directional $\gamma$-ray detector is placed at the center, and lightweight lead
and iron shields partially surround the detector. By rotating these shields during measurement,
the incident $\gamma$-ray flux from each direction is modulated, providing pseudo-directional
information while keeping the total payload within the robot limits.*

<a id="table:density"></a>
*Material properties~\cite{ref_NIST_XCOM}*

| Material | Density |
| --- | --- |
| Lead | \SI{11.36}{g/cm^{3}} |
| Iron | \SI{7.87}{g/cm^{3}} |
| Concrete | \SI{2.1}{g/cm^{3}} |



<!-- ================================================== % -->
<!-- section -->
<!-- ================================================== % -->
## Particle Filter Formulation for Multi-Isotope Source-Term Estimation
<a id="chap3_pf_pf"></a>

In this section, I formulate the estimation of multiple three-dimensional point sources as Bayesian inference with parallel PFs, one PF for each isotope.
The PFs use the isotope-wise counts defined in Eq.~(\ref{eq:pf_isotope_vector}) of Section~\ref{subsec:spectrum_isotope_counts} and the precomputed geometric and shielding kernels to infer the number, locations, and strengths of sources and the background rate.

### Precomputed Geometric and Shielding Kernels
<a id="chap3_pf_kernel"></a>

To efficiently evaluate expected counts for different robot poses and shield orientations,
I precompute attenuation kernels that encode geometric spreading and shielding attenuation
for point sources. In contrast to grid--based methods, the PF in this chapter represents the
radiation field directly as a finite set of point sources whose positions are state variables;
no voxelisation of the environment is required.

For isotope $h$, let the $j$-th source position be
$$
  \bm{s}_{h,j} = [x_{h,j}, y_{h,j}, z_{h,j}]^{\top},
  \qquad j = 1,\dots,r_h ,
$$
where $r_h$ is the (unknown) number of sources of isotope $h$.
The robot measurement poses (detector positions) are denoted by
$$
  \bm{q}_k = [x^{\mathrm{det}}_k, y^{\mathrm{det}}_k, z^{\mathrm{det}}_k]^{\top},
  \qquad k = 1,\dots,N_{\mathrm{meas}}.
$$

The distance and direction from source $j$ to pose $k$ are
<a id="eq:pf_distance_direction"></a>
$$
\begin{aligned}
  d_{k,j} &= \|\bm{q}_k - \bm{s}_{h,j}\|_2, \\
  \hat{\bm{u}}_{k,j} &= \frac{\bm{q}_k - \bm{s}_{h,j}}{d_{k,j}} .
\end{aligned}
$$

The basic geometric contribution from a point source to a non--directional detector is given by the inverse--square law~\cite{ref_Tsoulfanidis1995},
<a id="eq:pf_geometry"></a>
$$
  G_{k,j} = \frac{1}{4\pi d_{k,j}^2}.
$$

In the absence of shielding, the linear model in Eq.~(\ref{eq:chap3_b_Aq}) reduces to this
geometric term.

The detector is surrounded by lightweight iron and lead shields.
At time $k$, their orientations are represented by rotation matrices
$$
  \bm{R}^{\mathrm{Fe}}_k, \quad \bm{R}^{\mathrm{Pb}}_k \in SO(3).
$$

Let the shield thicknesses be $X^{\mathrm{Fe}}$ and $X^{\mathrm{Pb}}$, and let
$\mu^{\mathrm{Fe}}(E_h)$ and $\mu^{\mathrm{Pb}}(E_h)$ denote the linear attenuation
coefficients of iron and lead at the representative energy $E_h$ of isotope $h$.
For direction $\hat{\bm{u}}_{k,j}$, the effective path lengths through the shields are
$$
  T^{\mathrm{Fe}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Fe}}_k),\quad
  T^{\mathrm{Pb}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Pb}}_k),
$$
which are approximated by an octant-based blocking test.
If the source-to-detector direction lies within the active octant of each shield,
the path length is set to the nominal thickness; otherwise it is zero.

$$
  T^{\mathrm{Fe}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Fe}}_k)
  =
  \begin{cases}
    X^{\mathrm{Fe}}, & \text{if the ray is blocked by the Fe octant},\\
    0, & \text{otherwise},
  \end{cases}
$$

$$
  T^{\mathrm{Pb}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Pb}}_k)
  =
  \begin{cases}
    X^{\mathrm{Pb}}, & \text{if the ray is blocked by the Pb octant},\\
    0, & \text{otherwise}.
  \end{cases}
$$

The shielding attenuation factor becomes
<a id="eq:pf_shield_attn"></a>
$$
  A^{\mathrm{sh}}_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = \exp\left(
      - \mu^{\mathrm{Fe}}(E_h)
        T^{\mathrm{Fe}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Fe}}_k)
      - \mu^{\mathrm{Pb}}(E_h)
        T^{\mathrm{Pb}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Pb}}_k)
    \right).
$$

The combined kernel for isotope $h$ and source $j$ is defined as
<a id="eq:pf_kernel"></a>
$$
  K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = G_{k,j}\,
    A^{\mathrm{sh}}_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
$$

For later use, I also regard the kernel as a function of a generic source position
$$
  K_{k,h}(\bm{s},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k),
$$
so that
$K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
 = K_{k,h}(\bm{s}_{h,j},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$.
Note that attenuation by environmental obstacles is not considered; only geometric spreading
and attenuation by the lightweight shields are modeled.

Let $q_{h,j} \ge 0$ denote the strength of the $j$-th source of isotope $h$, and let $b_h$
denote the background count rate for isotope $h$. I collect the source strengths in the vector
$$
  \bm{q}_h = (q_{h,1},\dots,q_{h,r_h})^{\top}.
$$

For a given shield orientation, the expected count rate (per unit time) for isotope $h$ at pose $k$ is
<a id="eq:pf_lambda_rate"></a>
$$
  \lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = b_h
    + \sum_{j=1}^{r_h}
        K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)\,q_{h,j}.
$$
For acquisition time $T_k$, the expected total count is
<a id="eq:pf_lambda_total"></a>
$$
  \Lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = T_k\,\lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
$$

In practice, because source locations are continuous, the kernel is evaluated on-the-fly
for each particle and pose.
Static quantities such as octant normals, shield thicknesses, and isotope-specific attenuation coefficients are cached, while the geometric term $G_{k,j}$ and the attenuation factor $A^{\mathrm{sh}}_{k,j,h}$ are computed at each update.

### PF State Representation and Initialization
<a id="chap_pf_state_init"></a>

I construct an independent PF for each isotope $h\in\mathcal{H}$.
The state vector for isotope $h$ is defined as
<a id="eq:pf_state"></a>
$$
  \bm{\theta}_h
  = \left(
      r_h,\;
      \{\bm{s}_{h,m}\}_{m=1}^{r_h},\;
      \{q_{h,m}\}_{m=1}^{r_h},\;
      b_h
    \right),
$$
where $r_h$ is the number of sources of isotope $h$, $\bm{s}_{h,m}$ is the location of the $m$-th source, $q_{h,m}$ is its strength, and $b_h$ is the background rate for isotope $h$.

The posterior distribution $p(\bm{\theta}_h\mid \{\bm{z}_k\})$ is approximated by $N_{\mathrm{p}}$ weighted particles
<a id="eq:pf_particles"></a>
$$
  \left\{
    \bm{\theta}_h^{(n)}, w_{h}^{(n)}
  \right\}_{n=1}^{N_{\mathrm{p}}},
$$
where $\bm{\theta}_h^{(n)}$ is the $n$-th particle and $w_{h}^{(n)}$ is its normalized weight.

For initialization, I assume a broad prior over the number of sources, their locations, and strengths.
The source locations are sampled from a uniform distribution over the explored volume, while source strengths are sampled from non--negative distributions (e.g., uniform or log--normal)~\cite{ref_Arulampalam2002}.
In this study, the background rate is fixed to zero and is not updated during filtering.
The initial weights are uniform,
<a id="eq:pf_init_weights"></a>
$$
  w_{h,0}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
$$

If the number of sources $r_h$ is unknown and may change during the exploration, I incorporate birth/death moves into the PF~\cite{ref_khadanga2020_ari,ref_pinkam2020_irc} so that different particles can maintain different values of $r_h$.
In the implementation, these trans-dimensional moves are applied after resampling as part of the regularization step described in Section~\ref{chap_pf_resample}.

### Prediction and Log--Domain Weight Update for Poisson Observations
<a id="chap_pf_weight_update"></a>

At time step $k$, the robot is at pose $\bm{q}_k$ with shield orientation $(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$.
For isotope $h$ and particle $n$, let $r_h^{(n)}$ be the number of sources in
$\bm{\theta}_h^{(n)}$. Using the kernel in Eq.~(\ref{eq:pf_kernel}), the expected total
count is
<a id="eq:pf_expected_counts_particle"></a>
$$
  \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = T_k\left(
      b_h^{(n)} + \sum_{j=1}^{r_h^{(n)}}
        K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
        q_{h,j}^{(n)}
    \right),
$$
where $q_{h,j}^{(n)}$ is the strength of the $j$-th source of isotope $h$ in particle $n$.
Here $K_{k,j,h}(\cdot)$ is evaluated at the source position $\bm{s}^{(n)}_{h,j}$ of that particle.

The observed isotope--wise count $z_{k,h}$ is modeled as a Poisson random variable,
<a id="eq:pf_poisson_model"></a>
$$
  z_{k,h} \sim \mathrm{Poisson}\left(
    \Lambda_{k,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  \right),
$$
consistent with the likelihood in Eq.~(\ref{eq:chap3_equation4}).

The likelihood of observing $z_{k,h}$ for particle $n$ is~\cite{ref_Tsoulfanidis1995}
<a id="eq:pf_poisson_likelihood"></a>
$$
  p(z_{k,h} \mid \bm{\theta}_h^{(n)},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = \frac{
      \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)^{z_{k,h}}
      \exp\!\Big(-\Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)\Big)
    }{z_{k,h}!}.
$$

To avoid numerical underflow, I perform the weight update in the logarithmic domain~\cite{ref_kemp2023_tns}.
Let $\log w_{h,k-1}^{(n)}$ be the previous log weight.
The unnormalized new log weight is
<a id="eq:pf_log_weight_raw"></a>
$$
  \log \tilde{w}_{h,k}^{(n)}
  = \log w_{h,k-1}^{(n)}
    + z_{k,h}\log \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
    - \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
$$

The normalized weight $w_{h,k}^{(n)}$ is obtained by
<a id="eq:pf_log_weight_norm"></a>
$$
  w_{h,k}^{(n)}
  = \frac{
      \exp\Big(\log \tilde{w}_{h,k}^{(n)} - \max_{m} \log \tilde{w}_{h,k}^{(m)}\Big)
    }{
      \sum_{m=1}^{N_{\mathrm{p}}}
        \exp\Big(\log \tilde{w}_{h,k}^{(m)} - \max_{m'} \log \tilde{w}_{h,k}^{(m')}\Big)
    }.
$$

Subtracting the maximum log weight improves numerical stability without changing the normalized weights.

### Resampling, Regularization, and Particle Count Adaptation
<a id="chap_pf_resample"></a>

When the weights become highly imbalanced, the effective number of particles decreases and the PF may degenerate.
The effective sample size for isotope $h$ is defined as~\cite{ref_Arulampalam2002}
<a id="eq:pf_neff"></a>
$$
  N_{\mathrm{eff},h}
  = \frac{1}{\sum_{n=1}^{N_{\mathrm{p}}} \big(w_{h,k}^{(n)}\big)^2}.
$$

If $N_{\mathrm{eff},h}$ drops below a threshold $N_{\mathrm{th}}$, resampling is performed.
I adopt low--variance (systematic) resampling or similar algorithms~\cite{ref_Arulampalam2002} to draw a new set of particles from the discrete distribution defined by $\{w_{h,k}^{(n)}\}$.
After resampling, the weights are reset to the uniform value
<a id="eq:pf_weights_after_resample"></a>
$$
  w_{h,k}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
$$

To prevent premature convergence to local optima and to support trans-dimensional updates in $r_h$, I regularize the resampled particles by perturbing the source locations and strengths~\cite{ref_khadanga2020_ari,ref_pinkam2020_irc}.
For each particle $n$ and each source $m=1,\dots,r_h^{(n)}$, I apply
<a id="eq:pf_regularization"></a>
$$
\begin{aligned}
  \bm{s}_{h,m}^{(n)} &\leftarrow
  \mathrm{clip}\!\left(
    \bm{s}_{h,m}^{(n)} + \bm{\epsilon}_{\mathrm{pos}},
    \bm{s}_{\min},\bm{s}_{\max}
  \right),
  \qquad
  \bm{\epsilon}_{\mathrm{pos}} \sim \mathcal{N}(\bm{0},\sigma_{\mathrm{pos}}^2\bm{I}),
  \\
  q_{h,m}^{(n)} &\leftarrow \max\!\left(q_{h,m}^{(n)} + \epsilon_{\mathrm{int}},\,0\right),
  \qquad
  \epsilon_{\mathrm{int}} \sim \mathcal{N}(0,\sigma_{\mathrm{int}}^2),
\end{aligned}
$$
where $\mathrm{clip}(\cdot,\bm{s}_{\min},\bm{s}_{\max})$ confines the location to a bounding box of the explored volume and the strength is kept non-negative by clipping at zero.

After jittering, I apply birth/death moves to allow the source count $r_h$ to vary across particles.
A death move removes weak sources stochastically: if $q_{h,m}^{(n)} < q_{\min}$, the source is removed with probability $p_{\mathrm{kill}}$.
A birth move is applied with probability $p_{\mathrm{birth}}$ (only if $r_h^{(n)} < r_{\max}$), where a new source is added with
$$
  \bm{s}_{\mathrm{new}} \sim \mathrm{Unif}(\mathcal{V}),
$$
and
$$
  q_{\mathrm{new}} \sim \left|\mathcal{N}(\mu_b, \sigma_b^2)\right|.
$$
In the implementation, $(\mu_b,\sigma_b)=(0.1,\,0.05)$ and the absolute value enforces non-negativity.

To balance computational cost and estimation accuracy, I adapt the particle count online using the effective sample size (ESS).
At each update, the ESS ratio $\rho_h = N_{\mathrm{eff},h}/N_{\mathrm{p}}$ is evaluated.
If $\rho_h$ falls below a lower threshold, the number of particles is increased (up to a prescribed maximum) to better represent a diffuse posterior.
Conversely, if $\rho_h$ exceeds an upper threshold, the number of particles is reduced (down to a prescribed minimum) to save computation when the posterior has largely converged.
This ESS-driven adaptation is combined with resampling, and additional jitter is applied only when the particle set is expanded.

#### Label alignment for multi-source particles

When multiple sources are present, the ordering of sources inside each particle is arbitrary and
can permute across particles (label switching). If the posterior mean is computed by index-wise
averaging, such permutations blur the estimated source locations and strengths.

To mitigate this, I align the source ordering of each particle to a common reference ordering
after resampling/regularization. Let $(\bm{s}_i, q_i)$ be the $i$-th source in a particle, and
$(\bar{\bm{s}}_j, \bar{q}_j)$ the $j$-th reference source. I define a cost matrix
$$
  C_{ij} =
  \lambda_s \frac{\|\bm{s}_i - \bar{\bm{s}}_j\|_2}{d_s}
  + \lambda_q \frac{|q_i - \bar{q}_j|}{d_q},
$$
where $d_s$ and $d_q$ are scale factors and $\lambda_s, \lambda_q$ are weights. The assignment
$\pi$ is obtained by the Hungarian algorithm,
$$
  \pi = \arg\min_{\pi} \sum_{i} C_{i,\pi(i)}.
$$
If the number of sources differs, the cost matrix is padded with dummy rows/columns with a
fixed penalty to allow unassigned sources. After reordering by $\pi$, unmatched sources are
kept at the end. The reference is refined by recomputing the weighted mean after alignment;
a small number of iterations is sufficient in practice.

### Mixing of Parallel PFs and Convergence Criteria
<a id="chap_pf_mixing"></a>

The independent PFs for each isotope yield separate estimates of source locations and strengths.
However, some inferred sources may be spurious.
To obtain a consistent multi--isotope source map, I aggregate the PF outputs and remove spurious sources.

Following the ``best--case measurement'' test proposed in~\cite{ref_kemp2023_tns}, I proceed as follows.
For isotope $h$, let the set of candidate sources obtained from the PF (e.g., using the MMSE estimate) be
<a id="eq:pf_candidate_sources"></a>
$$
  \hat{\mathcal{S}}_h
  = \left\{
      (\hat{\bm{s}}_{h,m}, \hat{q}_{h,m})
    \right\}_{m=1}^{\hat{r}_h}.
$$

For each candidate source $(\hat{\bm{s}}_{h,m}, \hat{q}_{h,m})$, I compute its predicted
contribution to the count at all measurement poses $k$,
<a id="eq:pf_predicted_single"></a>
$$
  \hat{\Lambda}_{k,h,m}
  = T_k\,
    K_{k,h}\big(\hat{\bm{s}}_{h,m},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k\big)\,
    \hat{q}_{h,m},
$$

Let $k^{\star}$ denote the measurement pose where the candidate source is expected to be most visible, e.g.,
<a id="eq:pf_best_measurement_for_source"></a>
$$
  k^{\star}
  = \arg\max_{k} \frac{\hat{\Lambda}_{k,h,m}}{z_{k,h} + \epsilon},
$$
where $\epsilon$ is a small positive constant to avoid division by zero.

I then test whether the candidate source can explain a sufficient fraction of the observed count at $k^{\star}$,
<a id="eq:pf_spurious_test"></a>
$$
  \frac{\hat{\Lambda}_{k^{\star},h,m}}{z_{k^{\star},h}}
  \ge \tau_{\mathrm{mix}},
$$
where $\tau_{\mathrm{mix}}$ is a threshold (e.g., $\tau_{\mathrm{mix}}=0.9$).
If the inequality is not satisfied, the candidate is considered spurious and removed from $\hat{\mathcal{S}}_h$.

Possible convergence criteria for terminating the online estimation include:

- The volume of the $95\%$ credible region of the source locations for each isotope $h$ falls below a predefined threshold.
- The change in the estimated source strengths or locations between consecutive time steps is small, e.g.,
  $$
    \left\|
      \hat{\bm{q}}_{h,k} - \hat{\bm{q}}_{h,k-1}
    \right\|
    < \tau_{\mathrm{conv}},
  $$
  for several successive $k$, where $\hat{\bm{q}}_{h,k}$ denotes a vector of estimated source strengths or positions at time $k$.
- The information--gain criterion used in the shield--rotation strategy (Section~\ref{chap3_pf_rotation_section}) becomes small for all candidate poses, indicating that additional measurements are unlikely to significantly improve the estimate.

Once the convergence criteria are satisfied, the exploration is terminated and the final estimates are reported.

For each isotope $h$ and each estimated source index $m$, the method outputs the estimated source location $\hat{\bm{s}}_{h,m}$, the estimated strength $\hat{q}_{h,m}$, and the associated covariance matrix $\mathrm{Cov}(\bm{s}_{h,m}, q_{h,m})$.
For visualization and practical use, the estimated source distribution can be rendered as radiation maps (e.g., heat maps or point-source markers) overlaid with the robot trajectory and known obstacles, enabling operators to interpret the spatial distribution of radiation in the environment.

I define a global uncertainty measure as
<a id="eq:pf_global_uncertainty"></a>
$$
  U
  = \sum_{h\in\mathcal{H}}\sum_{m=1}^{\hat{r}_h} \mathrm{Var}(q_{h,m}),
$$
where $\hat{r}_h$ is the current estimated number of sources of isotope $h$.



<!-- ================================================== % -->
<!-- section -->
<!-- ================================================== % -->
## Shield Rotation Strategy
<a id="chap3_pf_rotation_section"></a>

In this section I describe the proposed method that actively rotates the lightweight shields to obtain pseudo-directional information from the non--directional detector.
The strategy consists of generating candidate shield orientations, predicting the measurement value of each orientation using the PF particles and kernels, executing short--time measurements while rotating the shields, and selecting the next robot pose.

### Generation of Candidate Shield Orientations
<a id="chap_pf_rotation_candidates"></a>

While the robot is stopped at pose $\bm{q}_k$, it can rotate the shields and perform several measurements with different orientations.
I discretize the azimuth angles of the iron and lead shields as
$$
\begin{aligned}
  \Phi^{\mathrm{Fe}} = \{\phi^{\mathrm{Fe}}_1,\dots,\phi^{\mathrm{Fe}}_{N_{\mathrm{Fe}}}\}, \\
  \Phi^{\mathrm{Pb}} = \{\phi^{\mathrm{Pb}}_1,\dots,\phi^{\mathrm{Pb}}_{N_{\mathrm{Pb}}}\},
\end{aligned}
$$
while keeping elevation and roll fixed.

The set of candidate shield orientations is
<a id="eq:pf_orientation_candidates"></a>
$$
  \mathcal{R}
  = \left\{
      \big(\bm{R}^{\mathrm{Fe}}_u,\bm{R}^{\mathrm{Pb}}_v\big)
      \;\middle|\;
      \phi^{\mathrm{Fe}}_u\in\Phi^{\mathrm{Fe}},\;
      \phi^{\mathrm{Pb}}_v\in\Phi^{\mathrm{Pb}}
    \right\},
$$
where $\bm{R}^{\mathrm{Fe}}_u$ and $\bm{R}^{\mathrm{Pb}}_v$ are the rotation matrices corresponding to the azimuth angles $\phi^{\mathrm{Fe}}_u$ and $\phi^{\mathrm{Pb}}_v$, respectively.
In general, $|\mathcal{R}| = N_{\mathrm{Fe}}N_{\mathrm{Pb}}$, but symmetry and mechanical constraints can be used to reduce the number of effective patterns to a smaller set $N_{\mathrm{R}}$.

For each candidate orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}$, the kernels $K_{k,j,h}$ defined in~\eqref{eq:pf_kernel} are used to predict expected counts and to evaluate the measurement value as described next.

### Measurement Value Prediction and Orientation Selection
<a id="chap_pf_pose_value"></a>

For a candidate shield orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}$ and acquisition time $T_k$, the expected total count for isotope $h$ and particle $n$ is given by
<a id="eq:pf_expected_counts_particle_rotation"></a>
$$
  \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  = T_k\left(
      b_h^{(n)} + \sum_{j=1}^{r_h^{(n)}}
        K_{k,j,h}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
        q_{h,j}^{(n)}
    \right),
$$
consistent with Eq.~(\ref{eq:pf_expected_counts_particle}).

To actively select the orientation, I evaluate the ``measurement value'' of each candidate using the current particle set.
In this study, I use expected information gain (EIG), defined as the expected reduction of the particle-weight entropy.

**Expected information gain.**

Let $\bm{w}_h = (w_{h}^{(1)},\dots,w_{h}^{(N_{\mathrm{p}})})$ be the current weight vector for isotope $h$.
Its Shannon entropy is
<a id="eq:pf_entropy"></a>
$$
  H(\bm{w}_h) = -\sum_{n=1}^{N_{\mathrm{p}}} w_{h}^{(n)} \log w_{h}^{(n)}.
$$

Suppose that measurement $z_{k,h}$ is hypothetically obtained under orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$.
After updating the weights (Section~\ref{chap_pf_weight_update}), the new weight vector becomes $\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$.
The expected posterior entropy is
$$
  \mathbb{E}_{z_{k,h}}
  \big[
    H\big(\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\big)
  \big].
$$

The expected information gain (EIG) for isotope $h$ is defined as
<a id="eq:pf_ig_single"></a>
$$
  \mathrm{IG}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  = H(\bm{w}_h)
    - \mathbb{E}_{z_{k,h}}
      \big[
        H\big(\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\big)
      \big],
$$
which measures the expected reduction in uncertainty for isotope $h$~\cite{ref_ristic2010,ref_pinkam2020_irc,ref_lazna2025_net}.

To combine multiple isotopes, I use a weighted sum
<a id="eq:pf_ig_total"></a>
$$
\begin{aligned}
  \mathrm{IG}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  = \sum_{h\in\mathcal{H}} \alpha_h\,
    \mathrm{IG}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}}), \\
  \sum_{h\in\mathcal{H}} \alpha_h = 1,
\end{aligned}
$$
where $\alpha_h$ reflects the relative importance of isotope $h$.

### Short--Time Measurements with Rotating Shields
<a id="chap_pf_rotation_measurement"></a>

At pose $\bm{q}_k$, the shield orientation for the next measurement is chosen by maximizing the measurement value over all candidates,
<a id="eq:pf_best_orientation"></a>
$$
    (\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
      = \arg\max_{(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}}
        \mathrm{IG}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}}),
$$

The robot rotates the iron and lead shields to $(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$ and acquires a spectrum for a short time interval $T_k$.
The spectrum is processed according to Section~\ref{subsec:spectrum_isotope_counts} to obtain the isotope--wise counts $z_{k,h}$ in Eq.~(\ref{eq:pf_isotope_vector}).
These counts serve as the observations for updating the PF at time step $k$.

The acquisition time $T_k$ per rotation is selected considering the trade--off between signal--to--noise ratio and total exploration time.

### Stopping Shield Rotation and Active Selection of the Next Pose
<a id="chap_pf_rotation_stop_nextpose"></a>

At a single robot pose $\bm{q}_k$, performing multiple measurements with different shield orientations increases information but also increases total measurement time.
Let $\Delta \mathrm{IG}_k^{(r)}$ denote the information gain obtained by the $r$-th rotation at pose $\bm{q}_k$,
<a id="eq:pf_delta_ig"></a>
$$
  \Delta \mathrm{IG}_k^{(r)}
  = \mathrm{IG}\!\left(
      (\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})^{(r)}
    \right).
$$

When $\Delta \mathrm{IG}_k^{(r)}$ falls below a threshold $\tau_{\mathrm{IG}}$, I stop rotating at pose $\bm{q}_k$.
In addition, a maximum dwell time $T_{\max}$ per pose is imposed for safety and scheduling reasons,
<a id="eq:pf_dwell_limit"></a>
$$
  \sum_{r} T_k^{(r)} \le T_{\max},
$$
where $T_k^{(r)}$ is the acquisition time of the $r$-th rotation at pose $\bm{q}_k$.

The next robot pose is chosen actively based on the current PF state.
Let $\{\bm{q}^{\mathrm{cand}}_1,\dots,\bm{q}^{\mathrm{cand}}_L\}$ be a set of candidate future poses, generated for example by sampling reachable positions while avoiding obstacles.

Using the global uncertainty measure $U$ defined in Eq.~(\ref{eq:pf_global_uncertainty}), I approximate, for each candidate pose $\bm{q}^{\mathrm{cand}}_{\ell}$, the expected uncertainty after one hypothetical measurement as $\mathbb{E}[U\mid \bm{q}^{\mathrm{cand}}_{\ell}]$ using the attenuation kernels and the PF particles~\cite{ref_ristic2010,ref_pinkam2020_irc,ref_lazna2025_net}.

I then choose the next pose by minimizing the expected uncertainty plus a motion cost,
<a id="eq:pf_next_pose"></a>
$$
  \bm{q}_{k+1}
  = \arg\min_{\bm{q}^{\mathrm{cand}}_{\ell}}
      \left(
        \mathbb{E}[U\mid \bm{q}^{\mathrm{cand}}_{\ell}]
        + \lambda_{\mathrm{cost}}
          C(\bm{q}_k,\bm{q}^{\mathrm{cand}}_{\ell})
      \right),
$$
where $C(\bm{q}_k,\bm{q}^{\mathrm{cand}}_{\ell})$ is a cost function that encodes travel distance and obstacle avoidance, and $\lambda_{\mathrm{cost}}$ is a weighting parameter balancing information gain against motion cost~\cite{ref_lazna2025_net}.

The overall online procedure, combining the measurement model of Section~\ref{chap3_pf_model}, the isotope-wise count processing of Section~\ref{subsec:spectrum_isotope_counts}, the particle-filter-based inference of Section~\ref{chap3_pf_pf}, and the shield-rotation strategy of this section, is summarised in Fig.~\ref{fig:pf_flow_rotshield}.
The procedure for selecting the next measurement pose is summarised in Fig.~\ref{fig:chap3_pose_selection}.

<a id="fig:chap3_shield_strategy"></a>
![Concept of the proposed shield--rotation strategy.](Figures/chap3/Shield-Selection.pdf)

*Concept of the proposed shield--rotation strategy.
At each robot pose, multiple shield orientations are evaluated using information--theoretic criteria, and short-time measurements are acquired with the most informative orientations.*

<a id="fig:chap3_pose_selection"></a>
![Procedure for active measurement pose selection.](Figures/chap3/Pose-Selection.pdf)

*Procedure for active measurement pose selection.
Given the current posterior represented by the particle filters, the robot enumerates a set of reachable candidate poses and predicts the expected observation at each pose.
An information-based objective is evaluated for each candidate, optionally combined with a motion cost.
The next pose minimising the objective is selected and the cycle is repeated until a stopping criterion is met.*

<a id="fig:pf_flow_rotshield"></a>
![Flowchart of the proposed particle-filter-based radiation source distribution estimation with rotating shields.](Figures/chap3/Flowchart.pdf)



<!-- ================================================== % -->
<!-- section -->
<!-- ================================================== % -->
## Summary
<a id="chap3_summary"></a>

In this chapter, an online radiation source distribution estimation method using a particle filter with rotating shields was described.

Section~\ref{chap3_pf_model} presented the mathematical measurement model for non-directional count observations and the design of lightweight shielding that satisfies the payload constraints of the mobile robot.

Section~\ref{chap3_pf_pf} formulated multi-isotope three-dimensional source-term estimation as Bayesian inference with parallel particle filters, including the construction of precomputed geometric and shielding kernels, state representation, prediction, log-domain weight update, resampling, regularization (including birth/death moves), and removal of spurious sources.

Section~\ref{chap3_pf_rotation_section} described the shield-rotation strategy that actively selects shield orientations and subsequent robot poses based on information-gain-based criteria, balancing measurement informativeness and motion cost.

The resulting estimates provide the inferred source locations, strengths, and uncertainties, which can be visualised as radiation maps overlaid on the environment for practical use.
