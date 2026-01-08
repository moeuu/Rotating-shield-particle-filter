\chapter{Online Radiation Source Term Estimation Using a Particle Filter with Rotating Shields}
\thispagestyle{empty}
\label{chap:chap3}
\graphicspath{{Figures/chap3/}}
\lhead[Chapter 3]{}
\minitoc

\newpage

% ================================================== %
% section
% ================================================== %
\section{Introduction}
\label{chap_pf_introduction}

In this chapter, I propose an online method for estimating the three-dimensional distribution of
multiple $\gamma$-ray sources using a particle filter (PF) combined with actively rotating shields.
A mobile robot equipped with an energy-resolving, non-directional detector and lightweight iron
and lead shields moves in a high-dose indoor environment and acquires radiation measurements
while continuously changing the shield orientations. By exploiting geometric spreading and
controlled attenuation due to the shields, the method recovers pseudo-directional information
from non-directional measurements and estimates the locations and strengths of multiple $\gamma$-ray
sources.

Section~\ref{chap3_pf_model} introduces the physical and mathematical model of non-directional count measurements with a shielded detector and discusses the design of lightweight lead shielding under robot payload constraints.\par
Section~\ref{chap3_pf_pf} formulates multi-isotope source-term estimation as Bayesian inference with parallel PFs, including the definition of precomputed geometric and shielding kernels, state
representation, prediction, log-domain weight update for Poisson observations, resampling, regularization, and spurious-source rejection.\par
Section~\ref{chap3_pf_rotation_section} presents the shield-rotation strategy, which selects informative shield orientations and next robot poses using information-gain and Fisher-information-based criteria together with short-time measurements.\par
Section~\ref{chap3_pf_conclusion} summarizes the convergence criteria and the final outputs of the algorithm, which provide radiation maps for subsequent visualization and decision making.\par
Finally, Section~\ref{chap3_summary} presents a summary of this chapter.\par


\clearpage
\newpage

% ================================================== %
% section
% ================================================== %
\section{Measurement Model and Shield Design}
\label{chap3_pf_model}

\subsection{Macroscopic Attenuation and Shield Thickness}
\label{chap3_shield}

Although $\gamma$ rays are electromagnetic waves, their high energy also gives them particle-like properties.
A macroscopic view of the interaction between $\gamma$ rays and matter considers an incident beam on a flat slab of material, as illustrated in Fig.~\ref{fig:chap2_attenuation}.
When $\gamma$ rays are incident on the material, some are absorbed and some undergo scattering, changing their direction and energy.

Consider a thin layer of thickness $dX$ inside the material, through which $N$ $\gamma$ rays are passing.
Let $N'$ be the number of $\gamma$ rays that pass through this layer without interaction.
The number of $\gamma$ rays that interact in the layer, $-dN = N - N'$, is proportional to both the thickness $dX$ of the layer and the number $N$ of incident $\gamma$ rays.
Using the proportionality constant $\mu$, this relationship can be written as
\begin{align}
    \label{eq:chap2_attenuation_differential}
        -dN = \mu \, N \, dX ,
\end{align}
where $\mu$ is called the linear attenuation coefficient and represents the ease with which interactions occur.

If $N_{0}$ is the number of $\gamma$ rays incident on the slab, solving Eq.~(\ref{eq:chap2_attenuation_differential}) yields
\begin{align}
    \label{eq:chap2_attenuation}
        N = N_{0} e^{-\mu X} .
\end{align}

The radiation dose measured by a detector obeys the inverse-square law with respect to the distance between a point source and the detector~\cite{ref_Tsoulfanidis1995}.
Let the detector position at the $k$-th measurement be
\begin{align}
\bm{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^{\top},
\end{align}
and let a point source of intensity $q_j$ be located at
\begin{align}
\bm{s}_j = [x_j, y_j, z_j]^{\top}.
\end{align}

Defining the distance
\begin{align}
    d_{k,j} = \sqrt{(x_k^{\mathrm{det}}-x_j)^{2} + (y_k^{\mathrm{det}}-y_j)^{2} + (z_k^{\mathrm{det}}-z_j)^{2}},
\end{align}
the radiation intensity $I(\bm{q}_k)$ measured at $\bm{q}_k$ is given by
\begin{align}
    \label{eq:chap2_inverse_square_law}
        I(\bm{q}_k) = \frac{S q_{j}}{4\pi d_{k,j}^{2}} e^{-\mu_{\mathrm{air}} d_{k,j}} ,
\end{align}
where $S$ is the detector area and $\mu_{\mathrm{air}}$ is the linear attenuation coefficient of air.
For the $\gamma$ rays emitted by \ce{^{137}Cs}, the linear attenuation coefficient in air at \SI{20}{^\circ C} is approximately $9.7\times10^{-3}$~\si{\per\metre}~\cite{ref_Tsoulfanidis1995}.
In this study, measurements are performed relatively close to the radiation sources, and thus I approximate $e^{-\mu_{\mathrm{air}} d_{k,j}} \approx 1$ and neglect attenuation in air.
I also neglect attenuation by environmental obstacles, such as walls and equipment, and explicitly model only the attenuation due to the lightweight shields.

The thickness of a shielding material required to reduce the dose rate by half is called the half-value layer, and the thickness required to reduce the dose rate to one-tenth is called the tenth-value layer.
Table~\ref{table:half_shield} lists the half-value and tenth-value layers of lead, iron, and concrete for $\gamma$ rays emitted by \ce{^{137}Cs}~\cite{ref_ICRP21}.

\begin{table}[t]
  \caption{Shield thickness and gamma-ray attenuation (unit: mm)}
  \label{table:half_shield}
  \centering
  \setlength{\tabcolsep}{4pt}
  \renewcommand{\arraystretch}{1.15}
  \begin{tabularx}{\linewidth}{c||>{\centering\arraybackslash}X >{\centering\arraybackslash}X|>{\centering\arraybackslash}X >{\centering\arraybackslash}X}
    \hline
     & \multicolumn{2}{c|}{Lead} & \multicolumn{2}{c}{Iron} \\
     & \makecell[c]{Half-value\\layer (HVL)} & \makecell[c]{Tenth-value\\layer (TVL)}
     & \makecell[c]{Half-value\\layer (HVL)} & \makecell[c]{Tenth-value\\layer (TVL)} \\
    \hline\hline
    Cs-137 & 7.0  & 22.0  & 15.0 & 50.0 \\
    Co-60  & 12.0 & 40.0  & 20.0 & 67.0 \\
    Eu-154 & 7.4  & 24.6  & 13.8 & 45.8 \\
    \hline
  \end{tabularx}
\end{table}


\subsection{Mathematical Model of Non-directional Count Measurements}
\label{chap3_radation}

This thesis assumes $M \geq 1$ unknown radiation point sources exist in the environment.
Let the position of the $j$-th source be $\bm{s}_j = [x_{j}, y_{j}, z_{j}]^{\top}$ and its strength be $q_{j} \ge 0$.
The radiation source distribution is represented by the vector
\begin{align}
    \label{eq:chap3_thetaj}
        \bm{q} = (q_{1}, q_{2}, \dots, q_{M})^{\top} .
\end{align}

A non-directional detector mounted on the robot acquires $N_{\mathrm{meas}}$ measurements at positions
\begin{align}
\bm{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^{\top},
\qquad k = 1,\dots,N_{\mathrm{meas}}.
\end{align}

Using the distance $d_{k,j}$ defined in Eq.~(\ref{eq:chap2_inverse_square_law}), and neglecting attenuation in air and environmental obstacles, the inverse-square law implies that the contribution of a unit-strength source at $\bm{s}_j$ to the detector at pose $\bm{q}_k$ is proportional to $1/d_{k,j}^{2}$.

This thesis absorb the detector area, acquisition time, and conversion factors between source strength and count rate into a single constant $\Gamma$ and define
\begin{align}
    \label{eq:chap3_bij}
        A_{k,j} = \frac{\Gamma}{d_{k,j}^{2}} ,
\end{align}
which represents the expected count at pose $k$ from a unit-strength source at source $j$.

The expected total count at pose $k$ due to the full source distribution $\bm{q}$ is then
\begin{align}
    \label{eq:chap3_bi}
        \Lambda_{k}(\bm{q}) = \sum_{j=1}^{M} A_{k,j} q_{j} .
\end{align}

Collecting all measurement poses, I define the vector of expected counts
\begin{align}
    \label{eq:chap3_bq}
        \boldsymbol{\Lambda}(\bm{q})
        = [\Lambda_{1}(\bm{q}), \Lambda_{2}(\bm{q}), \dots, \Lambda_{N_{\mathrm{meas}}}(\bm{q})]^{\top}.
\end{align}

Let $\bm{A} \in \mathbb{R}^{N_{\mathrm{meas}}\times M}$ be the matrix with elements $A_{k,j}$.
In matrix form, the measurement model can be written compactly as
\[
\bm{A} =
\begin{pmatrix}
  A_{1,1} & \dots & A_{1,M} \\
  \vdots  &       & \vdots  \\
  A_{N_{\mathrm{meas}},1} & \dots & A_{N_{\mathrm{meas}},M}
\end{pmatrix},
\]
and
\begin{align}
    \label{eq:chap3_b_Aq}
        \boldsymbol{\Lambda}(\bm{q}) = \bm{A}\bm{q} .
\end{align}

As discussed in Section~\ref{chap2_radiation}, radiation is emitted by stochastic radioactive decay processes, and the number of counts recorded by the detector follows a Poisson distribution~\cite{ref_Tsoulfanidis1995}.
Let $z_{k}$ denote the observed count at the $k$-th measurement.
The likelihood of observing $z_{k}$ given the source distribution $\bm{q}$ is
\begin{align}
    \label{eq:chap3_equation4}
    p(z_{k} \mid \bm{q})
      = \frac{\Lambda_{k}(\bm{q})^{\,z_{k}} \exp\!\big(-\Lambda_{k}(\bm{q})\big)}{z_{k}!} .
\end{align}

This single-isotope, count-only Poisson model is extended in Section~\ref{subsec:spectrum_isotope_counts} and in Section~\ref{chap3_pf_pf} to isotope-wise counts $z_{k,h}$ and shield-dependent kernels $K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$.

\subsection{Shield Mass and Robot Payload}
\label{chap3_shield_weight}

The densities of candidate shielding materials at \SI{20}{^\circ C} are listed in Table~\ref{table:density}.
For a given attenuation level (e.g., a specified number of half-value or tenth-value layers), the required shield mass is largely governed by the areal density (mass per unit area) and thus does not strongly depend on the material: denser materials require thinner shields, whereas less dense materials require thicker shields.
Accordingly, when mounting a shield on a mobile robot with a limited payload capacity, materials with high density are advantageous because they can achieve the required attenuation with smaller thickness.

In this study, lead and iron are selected as the shielding materials and are used \emph{simultaneously}.
Specifically, both materials are fabricated as lightweight one-eighth spherical shells (1/8-shells) that partially cover a non-directional detector.
This configuration provides sufficient attenuation while keeping the total shield mass within the payload constraints of the mobile robot, making the system feasible for deployment in real environments.

Figure~\ref{fig:chap3_detector_shield} illustrates the detector and shield configuration assumed in this chapter.
The non-directional detector is mounted at the center of the assembly, and two partial 1/8 spherical shells---one made of lead and the other made of iron---surround the detector.
During operation, these shells are actively rotated around the detector so that the attenuation factor in \eqref{eq:pf_shield_attn} varies with time.

% -------------------------------------------------- %
% Payload feasibility check (robot + shield mass)
% -------------------------------------------------- %

\subsubsection*{Representative robot platform and payload constraint}

To make the payload constraint explicit, we consider a representative compact UGV, the Clearpath
\emph{Jackal}. According to the manufacturer specifications, Jackal can carry a maximum payload
of \SI{20}{kg}.\cite{ref_clearpath_jackal}
This value is used as a practical reference when designing the detector--shield module
(detector head, digitizer, shields, rotation mechanism, and mounting structure).

\subsubsection*{Detector envelope and shield geometry}

As a robot-mountable energy-resolving spectrometer, we assume a compact \ce{CeBr3} scintillation detector.
Commercial \ce{CeBr3} detectors are available up to \SI{102}{mm} in diameter and \SI{127}{mm} in length.\cite{ref_scionix_cebr3}
To enclose a cylindrical detector of diameter $D$ and length $L$ inside a spherical shield assembly,
the minimum inner radius satisfies
\begin{align}
  R_{\mathrm{in}} \ge \frac{1}{2}\sqrt{D^{2}+L^{2}} .
\end{align}
Using the above maximum dimensions ($D=\SI{102}{mm}$, $L=\SI{127}{mm}$) gives
$R_{\mathrm{in}} \ge \SI{81.5}{mm}$.
In the following calculation, we conservatively set $R_{\mathrm{in}}=\SI{85}{mm}$ to account for the detector
housing and mechanical clearance.

\subsubsection*{Mass of 1/8 spherical shields for a tenth-value layer of \ce{^{137}Cs}}

For the dominant \SI{662}{keV} $\gamma$ ray of \ce{^{137}Cs}, the tenth-value layer (TVL) thicknesses
are $X_{\mathrm{Pb}}=\SI{22}{mm}$ and $X_{\mathrm{Fe}}=\SI{50}{mm}$ (Table~\ref{table:half_shield}), and the material densities are
$\rho_{\mathrm{Pb}}=\SI{11.36}{g/cm^{3}}$ and $\rho_{\mathrm{Fe}}=\SI{7.87}{g/cm^{3}}$
(Table~\ref{table:density}).
The mass of a one-eighth spherical shell (an octant) of inner radius $R_{\mathrm{in}}$ and thickness $X$ is
\begin{align}
  m(\rho,R_{\mathrm{in}},X)
  = \rho \frac{\pi}{6}\left[(R_{\mathrm{in}}+X)^{3}-R_{\mathrm{in}}^{3}\right],
  \label{eq:octant_shell_mass}
\end{align}
where $\rho$ is in \si{kg/m^{3}} and $R_{\mathrm{in}},X$ are in \si{m}.
Converting the densities in Table~\ref{table:density} to SI units yields
$\rho_{\mathrm{Pb}}=\SI{11360}{kg/m^{3}}$ and $\rho_{\mathrm{Fe}}=\SI{7870}{kg/m^{3}}$.
Substituting $R_{\mathrm{in}}=\SI{85}{mm}$, $X_{\mathrm{Pb}}=\SI{22}{mm}$, and $X_{\mathrm{Fe}}=\SI{50}{mm}$ into
\eqref{eq:octant_shell_mass} gives
\begin{align}
  m_{\mathrm{Pb}} &\approx \SI{3.63}{kg}, &
  m_{\mathrm{Fe}} &\approx \SI{7.61}{kg}, &
  m_{\mathrm{shields}} &\approx \SI{11.24}{kg}.
\end{align}

Therefore, even a conservative TVL design for \ce{^{137}Cs} results in a total shield mass of
approximately \SI{11}{kg}, which is below the \SI{20}{kg} maximum payload of Jackal.\cite{ref_clearpath_jackal}
This leaves roughly \SI{9}{kg} of payload margin for the detector head, a compact digitizer
(e.g., CAEN DT5730, \SI{670}{g}),\cite{ref_caen_dt5730}
the rotation mechanism, and mounting hardware, supporting the feasibility of the proposed
robot-mountable rotating-shield configuration.

\begin{table}[b]
  \caption{Material properties~\cite{ref_NIST_XCOM}}
  \label{table:density}
  \begin{center}
        \begin{tabular}{c|c}
        \hline
        Material & Density \\
        \hline \hline
        Lead & \SI{11.36}{g/cm^{3}} \\
        Iron & \SI{7.87}{g/cm^{3}}\\
        Concrete & \SI{2.1}{g/cm^{3}} \\
        \hline
        \end{tabular}
  \end{center}
\end{table}


\clearpage
\newpage

% ================================================== %
% section
% ================================================== %
\section{Particle Filter Formulation for Multi-Isotope Source-Term Estimation}
\label{chap3_pf_pf}

In this section I formulate the estimation of multiple three-dimensional point sources as Bayesian inference with parallel PFs, one PF for each isotope.
The PFs use the isotope-wise counts defined in Eq.~(\ref{eq:pf_isotope_vector}) of Section~\ref{subsec:spectrum_isotope_counts} and the precomputed geometric and shielding kernels to infer the number, locations, and strengths of sources and the background rate.

\subsection{Precomputed Geometric and Shielding Kernels}
\label{chap3_pf_kernel}

To efficiently evaluate expected counts for different robot poses and shield orientations,
I precompute attenuation kernels that encode geometric spreading and shielding attenuation
for point sources. In contrast to grid--based methods, the PF in this thesis represents the
radiation field directly as a finite set of point sources whose positions are state variables;
no voxelisation of the environment is required.

For isotope $h$, let the $j$-th source position be
\begin{align}
  \bm{s}_{h,j} = [x_{h,j}, y_{h,j}, z_{h,j}]^{\top},
  \qquad j = 1,\dots,r_h ,
\end{align}
where $r_h$ is the (unknown) number of sources of isotope $h$.
The robot measurement poses (detector positions) are denoted by
\begin{align}
  \bm{q}_k = [x^{\mathrm{det}}_k, y^{\mathrm{det}}_k, z^{\mathrm{det}}_k]^{\top},
  \qquad k = 1,\dots,N_{\mathrm{meas}}.
\end{align}

The distance and direction from source $j$ to pose $k$ are
\begin{align}
  d_{k,j} &= \|\bm{q}_k - \bm{s}_{h,j}\|_2, \\
  \hat{\bm{u}}_{k,j} &= \frac{\bm{q}_k - \bm{s}_{h,j}}{d_{k,j}} .
  \label{eq:pf_distance_direction}
\end{align}

The basic geometric contribution from a point source to a non--directional detector is given by the inverse--square law~\cite{ref_Tsoulfanidis1995},
\begin{align}
  G_{k,j} = \frac{1}{4\pi d_{k,j}^2}.
  \label{eq:pf_geometry}
\end{align}

In the absence of shielding, the linear model in Eq.~(\ref{eq:chap3_b_Aq}) reduces to this
geometric term.

The detector is surrounded by lightweight iron and lead shields.
At time $k$, their orientations are represented by rotation matrices
\begin{align}
  \bm{R}^{\mathrm{Fe}}_k, \quad \bm{R}^{\mathrm{Pb}}_k \in SO(3).
\end{align}

Let the shield thicknesses be $X^{\mathrm{Fe}}$ and $X^{\mathrm{Pb}}$, and let
$\mu^{\mathrm{Fe}}(E_h)$ and $\mu^{\mathrm{Pb}}(E_h)$ denote the linear attenuation
coefficients of iron and lead at the representative energy $E_h$ of isotope $h$.
For direction $\hat{\bm{u}}_{k,j}$, the effective path lengths through the shields are
\begin{align}
  T^{\mathrm{Fe}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Fe}}_k),\quad
  T^{\mathrm{Pb}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Pb}}_k),
\end{align}
which can be obtained by ray tracing through the shield geometry~\cite{ref_anderson2022_tase}.
The shielding attenuation factor becomes
\begin{align}
  A^{\mathrm{sh}}_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = \exp\left(
      - \mu^{\mathrm{Fe}}(E_h)
        T^{\mathrm{Fe}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Fe}}_k)
      - \mu^{\mathrm{Pb}}(E_h)
        T^{\mathrm{Pb}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Pb}}_k)
    \right).
  \label{eq:pf_shield_attn}
\end{align}

The combined kernel for isotope $h$ and source $j$ is defined as
\begin{align}
  K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = G_{k,j}\,
    A^{\mathrm{sh}}_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
  \label{eq:pf_kernel}
\end{align}

For later use, I also regard the kernel as a function of a generic source position
\[
  K_{k,h}(\bm{s},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k),
\]
so that
$K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
 = K_{k,h}(\bm{s}_{h,j},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$.
Note that attenuation by environmental obstacles is not considered; only geometric spreading
and attenuation by the lightweight shields are modeled.

Let $q_{h,j} \ge 0$ denote the strength of the $j$-th source of isotope $h$, and let $b_h$
denote the background count rate for isotope $h$. I collect the source strengths in the vector
\begin{align}
  \bm{q}_h = (q_{h,1},\dots,q_{h,r_h})^{\top}.
\end{align}

For a given shield orientation, the expected count rate (per unit time) for isotope $h$ at pose $k$ is
\begin{align}
  \lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = b_h
    + \sum_{j=1}^{r_h}
        K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)\,q_{h,j}.
  \label{eq:pf_lambda_rate}
\end{align}
For acquisition time $T_k$, the expected total count is
\begin{align}
  \Lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = T_k\,\lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
  \label{eq:pf_lambda_total}
\end{align}

In practice, all quantities that depend only on geometry and shield orientation, such as
$G_{k,j}$ and $A^{\mathrm{sh}}_{k,j,h}$, can be precomputed or cached and reused, while the
PF state variables $\bm{s}_{h,j}$ and $q_{h,j}$ remain continuous.

\subsection{PF State Representation and Initialization}
\label{chap_pf_state_init}

I construct an independent PF for each isotope $h\in\mathcal{H}$.
The state vector for isotope $h$ is defined as
\begin{align}
  \bm{\theta}_h
  = \left(
      r_h,\;
      \{\bm{s}_{h,m}\}_{m=1}^{r_h},\;
      \{q_{h,m}\}_{m=1}^{r_h},\;\
      b_h
    \right),
  \label{eq:pf_state}
\end{align}
where $r_h$ is the number of sources of isotope $h$, $\bm{s}_{h,m}$ is the location of the $m$-th source, $q_{h,m}$ is its strength, and $b_h$ is the background rate for isotope $h$.

The posterior distribution $p(\bm{\theta}_h\mid \{\bm{z}_k\})$ is approximated by $N_{\mathrm{p}}$ weighted particles
\begin{align}
  \left\{
    \bm{\theta}_h^{(n)}, w_{h}^{(n)}
  \right\}_{n=1}^{N_{\mathrm{p}}},
  \label{eq:pf_particles}
\end{align}
where $\bm{\theta}_h^{(n)}$ is the $n$-th particle and $w_{h}^{(n)}$ is its normalized weight.

For initialization, I assume a broad prior over the number of sources, their locations, and strengths.
The source locations are sampled from a uniform distribution over the explored volume, while source strengths and background rates are sampled from non--negative distributions (e.g., uniform or log--normal)~\cite{ref_Arulampalam2002}.
The initial weights are uniform,
\begin{align}
  w_{h,0}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
  \label{eq:pf_init_weights}
\end{align}

If the number of sources $r_h$ is unknown and may change during the exploration, birth/death moves can be incorporated into the PF~\cite{ref_khadanga2020_ari,ref_pinkam2020_irc}, so that different particles maintain different values of $r_h$.

\subsection{Prediction and Log--Domain Weight Update for Poisson Observations}
\label{chap_pf_weight_update}

At time step $k$, the robot is at pose $\bm{q}_k$ with shield orientation $(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$.
For isotope $h$ and particle $n$, let $r_h^{(n)}$ be the number of sources in
$\bm{\theta}_h^{(n)}$. Using the kernel in Eq.~(\ref{eq:pf_kernel}), the expected total
count is
\begin{align}
  \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = T_k\left(
      b_h^{(n)} + \sum_{j=1}^{r_h^{(n)}}
        K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
        q_{h,j}^{(n)}
    \right),
  \label{eq:pf_expected_counts_particle}
\end{align}
where $q_{h,j}^{(n)}$ is the strength of the $j$-th source of isotope $h$ in particle $n$.
Here $K_{k,j,h}(\cdot)$ is evaluated at the source position $\bm{s}^{(n)}_{h,j}$ of that particle.

The observed isotope--wise count $z_{k,h}$ is modeled as a Poisson random variable,
\begin{align}
  z_{k,h} \sim \mathrm{Poisson}\left(
    \Lambda_{k,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  \right),
  \label{eq:pf_poisson_model}
\end{align}
consistent with the likelihood in Eq.~(\ref{eq:chap3_equation4}).

The likelihood of observing $z_{k,h}$ for particle $n$ is~\cite{ref_Tsoulfanidis1995}
\begin{align}
  p(z_{k,h} \mid \bm{\theta}_h^{(n)},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = \frac{
      \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)^{z_{k,h}}
      \exp\!\Big(-\Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)\Big)
    }{z_{k,h}!}.
  \label{eq:pf_poisson_likelihood}
\end{align}

To avoid numerical underflow, thie thesis perform the weight update in the logarithmic domain~\cite{ref_kemp2023_tns}.
Let $\log w_{h,k-1}^{(n)}$ be the previous log weight.
The unnormalized new log weight is
\begin{align}
  \log \tilde{w}_{h,k}^{(n)}
  = \log w_{h,k-1}^{(n)}
    + z_{k,h}\log \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
    - \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
  \label{eq:pf_log_weight_raw}
\end{align}

The normalized weight $w_{h,k}^{(n)}$ is obtained by
\begin{align}
  w_{h,k}^{(n)}
  = \frac{
      \exp\Big(\log \tilde{w}_{h,k}^{(n)} - \max_{m} \log \tilde{w}_{h,k}^{(m)}\Big)
    }{
      \sum_{m=1}^{N_{\mathrm{p}}}
        \exp\Big(\log \tilde{w}_{h,k}^{(m)} - \max_{m'} \log \tilde{w}_{h,k}^{(m')}\Big)
    }.
  \label{eq:pf_log_weight_norm}
\end{align}

Subtracting the maximum log weight improves numerical stability without changing the normalized weights.

\subsection{Resampling, Regularization, and Particle Count Adaptation}
\label{chap_pf_resample}

When the weights become highly imbalanced, the effective number of particles decreases and the PF may degenerate.
The effective sample size for isotope $h$ is defined as~\cite{ref_Arulampalam2002}
\begin{align}
  N_{\mathrm{eff},h}
  = \frac{1}{\sum_{n=1}^{N_{\mathrm{p}}} \big(w_{h,k}^{(n)}\big)^2}.
  \label{eq:pf_neff}
\end{align}

If $N_{\mathrm{eff},h}$ drops below a threshold $N_{\mathrm{th}}$, resampling is performed.

I adopt low--variance (systematic) resampling or similar algorithms~\cite{ref_Arulampalam2002} to draw a new set of particles from the discrete distribution defined by $\{w_{h,k}^{(n)}\}$.
After resampling, the weights are reset to the uniform value
\begin{align}
  w_{h,k}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
  \label{eq:pf_weights_after_resample}
\end{align}

To prevent premature convergence to local optima, I regularize the resampled particles by adding small Gaussian perturbations to the source positions and strengths~\cite{ref_khadanga2020_ari,ref_pinkam2020_irc}.
Specifically,
\begin{align}
  \bm{s}_{h,m}^{(n)} &\leftarrow \bm{s}_{h,m}^{(n)} + \bm{\epsilon}_{\mathrm{pos}},\\
  q_{h,m}^{(n)} &\leftarrow q_{h,m}^{(n)} + \epsilon_{\mathrm{int}},
  \label{eq:pf_regularization}
\end{align}
where $\bm{\epsilon}_{\mathrm{pos}} \sim \mathcal{N}(\bm{0},\sigma_{\mathrm{pos}}^2\bm{I})$ and $\epsilon_{\mathrm{int}} \sim \mathcal{N}(0,\sigma_{\mathrm{int}}^2)$ are small zero--mean Gaussian noises.

To balance computational cost and estimation accuracy, the number of particles $N_{\mathrm{p}}$ can be adapted online~\cite{ref_kemp2023_tns,ref_pinkam2020_irc}.
For example, when the predictive log--likelihood variance or posterior entropy is large, $N_{\mathrm{p}}$ is increased to better represent the posterior.
Conversely, when the PF has clearly converged, $N_{\mathrm{p}}$ can be decreased to reduce computation time.

\subsection{Mixing of Parallel PFs and Convergence Criteria}
\label{chap_pf_mixing}

The independent PFs for each isotope yield separate estimates of source locations and strengths.
However, some inferred sources may be spurious.
To obtain a consistent multi--isotope source map, this thesis aggregate the PF outputs and remove spurious sources.

Following the ``best--case measurement'' test proposed in~\cite{ref_kemp2023_tns}, this thesis proceed as follows.
For isotope $h$, let the set of candidate sources obtained from the PF (e.g., using the MMSE estimate) be
\begin{align}
  \hat{\mathcal{S}}_h
  = \left\{
      (\hat{\bm{s}}_{h,m}, \hat{q}_{h,m})
    \right\}_{m=1}^{\hat{r}_h}.
  \label{eq:pf_candidate_sources}
\end{align}

For each candidate source $(\hat{\bm{s}}_{h,m}, \hat{q}_{h,m})$, I compute its predicted
contribution to the count at all measurement poses $k$,
\begin{align}
  \hat{\Lambda}_{k,h,m}
  = T_k\,
    K_{k,h}\big(\hat{\bm{s}}_{h,m},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k\big)\,
    \hat{q}_{h,m},
  \label{eq:pf_predicted_single}
\end{align}

Let $k^{\star}$ denote the measurement pose where the candidate source is expected to be most visible, e.g.,
\begin{align}
  k^{\star}
  = \argmax_k \frac{\hat{\Lambda}_{k,h,m}}{z_{k,h} + \epsilon},
  \label{eq:pf_best_measurement_for_source}
\end{align}
where $\epsilon$ is a small positive constant to avoid division by zero.

This thesis then test whether the candidate source can explain a sufficient fraction of the observed count at $k^{\star}$,
\begin{align}
  \frac{\hat{\Lambda}_{k^{\star},h,m}}{z_{k^{\star},h}}
  \ge \tau_{\mathrm{mix}},
  \label{eq:pf_spurious_test}
\end{align}
where $\tau_{\mathrm{mix}}$ is a threshold (e.g., $\tau_{\mathrm{mix}}=0.9$).
If the inequality is not satisfied, the candidate is considered spurious and removed from $\hat{\mathcal{S}}_h$.

Possible convergence criteria for terminating the online estimation include:
\begin{itemize}
  \item The volume of the $95\%$ credible region of the source locations for each isotope $h$ falls below a predefined threshold.
  \item The change in the estimated source strengths or locations between consecutive time steps is small, e.g.,
        \begin{align}
          \left\|
            \hat{\bm{q}}_{h,k} - \hat{\bm{q}}_{h,k-1}
          \right\|
          < \tau_{\mathrm{conv}},
        \end{align}
        for several successive $k$, where $\hat{\bm{q}}_{h,k}$ denotes a vector of estimated source strengths or positions at time $k$.
  \item The information--gain and Fisher--information--based criteria used in the shield--rotation strategy (Section~\ref{chap3_pf_rotation_section}) become small for all candidate poses, indicating that additional measurements are unlikely to significantly improve the estimate.
\end{itemize}

Once the convergence criteria are satisfied, the exploration is terminated and the final estimates are reported.
For each isotope $h$ and each estimated source index $m$, I can compute the posterior
variance $\mathrm{Var}(q_{h,m})$ of its strength from the particles. I define a global
uncertainty measure as
\begin{align}
  U
  = \sum_{h\in\mathcal{H}}\sum_{m=1}^{\hat{r}_h} \mathrm{Var}(q_{h,m}),
  \label{eq:pf_global_uncertainty}
\end{align}
where $\hat{r}_h$ is the current estimated number of sources of isotope $h$.

\clearpage
\newpage

% ================================================== %
% section
% ================================================== %
\section{Shield Rotation Strategy}
\label{chap3_pf_rotation_section}

In this section I describe the proposed method that actively rotates the lightweight shields to obtain pseudo-directional information from the non--directional detector.
The strategy consists of generating candidate shield orientations, predicting the measurement value of each orientation using the PF particles and kernels, executing short--time measurements while rotating the shields, and selecting the next robot pose.

\subsection{Generation of Candidate Shield Orientations}
\label{chap_pf_rotation_candidates}

While the robot is stopped at pose $\bm{q}_k$, it can rotate the shields and perform several measurements with different orientations.
I discretize the azimuth angles of the iron and lead shields as
\begin{align}
  \Phi^{\mathrm{Fe}} = \{\phi^{\mathrm{Fe}}_1,\dots,\phi^{\mathrm{Fe}}_{N_{\mathrm{Fe}}}\}, \\
  \Phi^{\mathrm{Pb}} = \{\phi^{\mathrm{Pb}}_1,\dots,\phi^{\mathrm{Pb}}_{N_{\mathrm{Pb}}}\},
\end{align}
while keeping elevation and roll fixed.

The set of candidate shield orientations is
\begin{align}
  \mathcal{R}
  = \left\{
      \big(\bm{R}^{\mathrm{Fe}}_u,\bm{R}^{\mathrm{Pb}}_v\big)
      \;\middle|\;
      \phi^{\mathrm{Fe}}_u\in\Phi^{\mathrm{Fe}},\;
      \phi^{\mathrm{Pb}}_v\in\Phi^{\mathrm{Pb}}
    \right\},
  \label{eq:pf_orientation_candidates}
\end{align}
where $\bm{R}^{\mathrm{Fe}}_u$ and $\bm{R}^{\mathrm{Pb}}_v$ are the rotation matrices corresponding to the azimuth angles $\phi^{\mathrm{Fe}}_u$ and $\phi^{\mathrm{Pb}}_v$, respectively.
In general, $|\mathcal{R}| = N_{\mathrm{Fe}}N_{\mathrm{Pb}}$, but symmetry and mechanical constraints can be used to reduce the number of effective patterns to a smaller set $N_{\mathrm{R}}$.

For each candidate orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}$, the kernels $K_{k,j,h}$ defined in~\eqref{eq:pf_kernel} are used to predict expected counts and to evaluate the measurement value as described next.

\subsection{Measurement Value Prediction and Orientation Selection}
\label{chap_pf_pose_value}

For a candidate shield orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}$ and acquisition time $T_k$, the expected total count for isotope $h$ and particle $n$ is given by
\begin{align}
  \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  = T_k\left(
      b_h^{(n)} + \sum_{j=1}^{r_h^{(n)}}
        K_{k,j,h}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
        q_{h,j}^{(n)}
    \right),
  \label{eq:pf_expected_counts_particle_rotation}
\end{align}
consistent with Eq.~(\ref{eq:pf_expected_counts_particle}).

To actively select the orientation, I evaluate the ``measurement value'' of each candidate using the current particle set.
In this study, I consider two representative criteria: expected information gain and Fisher information.

\paragraph{Expected information gain.}

Let $\bm{w}_h = (w_{h}^{(1)},\dots,w_{h}^{(N_{\mathrm{p}})})$ be the current weight vector for isotope $h$.
Its Shannon entropy is
\begin{align}
  H(\bm{w}_h) = -\sum_{n=1}^{N_{\mathrm{p}}} w_{h}^{(n)} \log w_{h}^{(n)}.
  \label{eq:pf_entropy}
\end{align}

Suppose that measurement $z_{k,h}$ is hypothetically obtained under orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$.
After updating the weights (Section~\ref{chap_pf_weight_update}), the new weight vector becomes $\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$.
The expected posterior entropy is
\begin{align}
  \mathbb{E}_{z_{k,h}}
  \big[
    H\big(\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\big)
  \big].
\end{align}

The expected information gain (EIG) for isotope $h$ is defined as
\begin{align}
  \mathrm{IG}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  = H(\bm{w}_h)
    - \mathbb{E}_{z_{k,h}}
      \big[
        H\big(\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\big)
      \big],
  \label{eq:pf_ig_single}
\end{align}
which measures the expected reduction in uncertainty for isotope $h$~\cite{ref_ristic2010,ref_pinkam2020_irc,ref_lazna2025_net}.

To combine multiple isotopes, this thesis uses a weighted sum
\begin{align}
  \mathrm{IG}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  = \sum_{h\in\mathcal{H}} \alpha_h\,
    \mathrm{IG}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}}), \\
  \sum_{h\in\mathcal{H}} \alpha_h = 1,
  \label{eq:pf_ig_total}
\end{align}
where $\alpha_h$ reflects the relative importance of isotope $h$.

\paragraph{Fisher-information-based criteria.}

Alternatively, I can evaluate each orientation using the Fisher information matrix, which characterizes the local sensitivity of the likelihood to parameter changes~\cite{ref_anderson2022_tase}.
For isotope $h$, let $\bm{I}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$ denote the Fisher information matrix of parameters $\bm{\theta}_h$ under orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$.
For a Poisson model, the Fisher information can be written as a sum of outer products of the gradient of the expected counts~\cite{ref_anderson2022_tase}.

Two standard scalar criteria are
\begin{align}
  J_{\mathrm{A}}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  &= \sum_{h\in\mathcal{H}}
     \beta_h\,
     \mathrm{Tr}\!\left(\bm{I}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})^{-1}\right)^{-1},
     \label{eq:pf_Aopt}\\
  J_{\mathrm{D}}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
  &= \sum_{h\in\mathcal{H}}
     \beta_h\,
     \log\det\!\left(\bm{I}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\right),
     \label{eq:pf_Dopt}
\end{align}
where $\beta_h$ are weighting coefficients.
Maximizing $J_{\mathrm{A}}$ or $J_{\mathrm{D}}$ corresponds to A-- or D--optimal design, respectively.

\subsection{Short--Time Measurements with Rotating Shields}
\label{chap_pf_rotation_measurement}

At pose $\bm{q}_k$, the shield orientation for the next measurement is chosen by maximizing the measurement value over all candidates,
\begin{align}
  (\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  = \argmax_{(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}}
    \mathrm{IG}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}}),
  \label{eq:pf_best_orientation}
\end{align}
or similarly by maximizing $J_{\mathrm{A}}$ or $J_{\mathrm{D}}$ defined in~\eqref{eq:pf_Aopt} and~\eqref{eq:pf_Dopt}.

The robot rotates the iron and lead shields to $(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$ and acquires a spectrum for a short time interval $T_k$.
The spectrum is processed according to Section~\ref{subsec:spectrum_isotope_counts} to obtain the isotope--wise counts $z_{k,h}$ in Eq.~(\ref{eq:pf_isotope_vector}).
These counts serve as the observations for updating the PF at time step $k$.

The acquisition time $T_k$ per rotation is selected considering the trade--off between signal--to--noise ratio and total exploration time.

\subsection{Stopping Shield Rotation and Active Selection of the Next Pose}
\label{chap_pf_rotation_stop_nextpose}

At a single robot pose $\bm{q}_k$, performing multiple measurements with different shield orientations increases information but also increases total measurement time.
Let $\Delta \mathrm{IG}_k^{(r)}$ denote the information gain obtained by the $r$-th rotation at pose $\bm{q}_k$,
\begin{align}
  \Delta \mathrm{IG}_k^{(r)}
  = \mathrm{IG}\!\left(
      (\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})^{(r)}
    \right).
  \label{eq:pf_delta_ig}
\end{align}

When $\Delta \mathrm{IG}_k^{(r)}$ falls below a threshold $\tau_{\mathrm{IG}}$, I stop rotating at pose $\bm{q}_k$.
In addition, a maximum dwell time $T_{\max}$ per pose is imposed for safety and scheduling reasons,
\begin{align}
  \sum_{r} T_k^{(r)} \le T_{\max},
  \label{eq:pf_dwell_limit}
\end{align}
where $T_k^{(r)}$ is the acquisition time of the $r$-th rotation at pose $\bm{q}_k$.

The next robot pose is chosen actively based on the current PF state.
Let $\{\bm{q}^{\mathrm{cand}}_1,\dots,\bm{q}^{\mathrm{cand}}_L\}$ be a set of candidate future poses, generated for example by sampling reachable positions while avoiding obstacles.

Using the global uncertainty measure $U$ defined in Eq.~(\ref{eq:pf_global_uncertainty}), I approximate, for each candidate pose $\bm{q}^{\mathrm{cand}}_{\ell}$, the expected uncertainty after one hypothetical measurement as $\mathbb{E}[U\mid \bm{q}^{\mathrm{cand}}_{\ell}]$ using the attenuation kernels and the PF particles~\cite{ref_ristic2010,ref_pinkam2020_irc,ref_lazna2025_net}.

I then choose the next pose by minimizing the expected uncertainty plus a motion cost,
\begin{align}
  \bm{q}_{k+1}
  = \argmin_{\bm{q}^{\mathrm{cand}}_{\ell}}
      \left(
        \mathbb{E}[U\mid \bm{q}^{\mathrm{cand}}_{\ell}]
        + \lambda_{\mathrm{cost}}
          C(\bm{q}_k,\bm{q}^{\mathrm{cand}}_{\ell})
      \right),
  \label{eq:pf_next_pose}
\end{align}
where $C(\bm{q}_k,\bm{q}^{\mathrm{cand}}_{\ell})$ is a cost function that encodes travel distance and obstacle avoidance, and $\lambda_{\mathrm{cost}}$ is a weighting parameter balancing information gain against motion cost~\cite{ref_lazna2025_net}.

The overall online procedure, combining the measurement model of Section~\ref{chap3_pf_model}, the isotope-wise count processing of Section~\ref{subsec:spectrum_isotope_counts}, the particle-filter-based inference of Section~\ref{chap3_pf_pf}, and the shield-rotation  strategy of this section, is summarised in Fig.~\ref{fig:pf_flow_rotshield}.
The procedure for selecting the next measurement pose is summarised in Fig.~\ref{fig:chap3_pose_selection}.


\clearpage
\newpage

% ================================================== %
% section
% ================================================== %
\section{Convergence Criteria and Output}
\label{chap3_pf_conclusion}

Once the shield rotation and robot motion planning described in Section~\ref{chap3_pf_rotation_section}, together with the PF inference in Section~\ref{chap3_pf_pf}, have reduced the uncertainty below the desired thresholds, the exploration is terminated and the final estimates are reported.

For each isotope $h$ and source index $m$, the method outputs the estimated source location $\hat{\bm{s}}_{h,m}$, the estimated strength $\hat{q}_{h,m}$, and the associated covariance matrix $\mathrm{Cov}(\bm{s}_{h,m},q_{h,m})$.
For visualization and practical use, the estimated source distribution can be rendered as a radiation heat map overlaid with the robot trajectory and obstacles, enabling operators to intuitively understand the spatial distribution of radiation in the environment.

\clearpage
\newpage

% ================================================== %
% section
% ================================================== %
\section{Summary}
\label{chap3_summary}

In this chapter, an online radiation source distribution estimation method using a particle filter with rotating shields was described.\par
Section~\ref{chap3_pf_model} presented the mathematical measurement model for non-directional count
observations and the design of lightweight shielding that satisfies the payload constraints of the mobile robot.\par
Section~\ref{chap3_pf_pf} formulated multi-isotope three-dimensional source-term estimation as Bayesian
inference with parallel particle filters, including the construction of precomputed geometric and shielding kernels, state representation, prediction, log-domain weight update, resampling, regularisation, and removal of spurious sources.\par
Section~\ref{chap3_pf_rotation_section} described the shield-rotation strategy that actively selects shield orientations and
subsequent robot poses based on information-gain and Fisher-information criteria, balancing measurement informativeness and motion cost.\par
Section~\ref{chap3_pf_conclusion} discussed convergence criteria and the final outputs of the algorithm, namely the
estimated source locations, strengths, and uncertainties, which can be visualised as radiation maps overlaid on the environment for practical use.\par

\clearpage
\newpage
