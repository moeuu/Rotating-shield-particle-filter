<!--
Chapter 2 (converted from LaTeX to Markdown)
Notes:
- LaTeX labels are kept as Pandoc-style identifiers {#...} where practical.
- Citations are kept as [@key].
- Math is kept in LaTeX math blocks for compatibility with Pandoc / MathJax.
-->

# Gamma-Ray Spectrum Unfolding and Radionuclide Identification {#chap:chap2}

## Introduction {#chap2_intro}

This chapter presents the gamma-ray spectrum unfolding and automated radionuclide identification method used throughout this thesis.  
Given a one-dimensional energy spectrum measured by a mobile scintillation spectrometer, the objective is to decompose the spectrum into contributions from individual radionuclides and to estimate their relative activities.  
The target application is high-dose, cluttered environments in which a small ground robot must infer the three-dimensional distribution of $\gamma$-ray sources from non-directional, energy-resolved measurements.

Following the framework of Kemp *et al.* [@ref_kemp2023_tns], the measured spectrum is modelled as a linear superposition of responses from candidate radionuclides plus background, with Poisson counting statistics governing the observed bin counts.  
The detector response matrix encodes the expected contribution of each radionuclide to each energy bin, including full-energy peaks, energy resolution effects, and background components.  
On top of this model, the chapter adopts a practical, peak-based unfolding strategy similar to that of Anderson *et al.* [@ref_anderson2019_case; @ref_anderson2022_tase], in which individual photopeaks are detected, quantified, associated with library lines, and then mapped to radionuclide activities.

- Section [Overview of Radiation](#chap2_radiation) reviews basic concepts of ionising radiation, radioactive decay, and representative $\gamma$-emitting radionuclides relevant to nuclear power plant accidents and related scenarios.  
- Section [Radiation Detectors](#chap2_rad_detector) summarises typical radiation detectors, contrasts directional and non-directional instruments, and explains the choice of a compact, non-directional scintillation spectrometer mounted on a mobile robot.  
- Section [Spectrum Modeling and Problem Formulation](#chap2_spectrum_problem) formulates the gamma-ray spectrum unfolding and radionuclide-identification problem as a linear inverse problem based on a detector response matrix and Poisson statistics.  
- Section [Spectrum Unfolding Procedure](#chap2_spectrum_procedure) details the practical peak-based unfolding pipeline, including preprocessing, peak detection, baseline estimation, decomposition of overlapping peaks, dead-time correction, radionuclide library matching, construction of isotope-wise count sequences for use in the particle-filter-based source-term estimation of Chapter 3, and estimation of relative activities.  
- Section [Summary](#chap2_summary) summarises the chapter and links these components to the source-term estimation methods developed in subsequent chapters.  

Finally, Section [Summary](#chap2_summary) presents a summary of this chapter.

---

## Overview of Radiation {#chap2_radiation}

### Types of Radiation

Representative types of radiation include particle radiation such as $\alpha$ rays, $\beta$ rays, and neutron radiation, as well as electromagnetic radiation such as $\gamma$ rays and X rays.

Each type of radiation has different energies and characteristics, and therefore different abilities to penetrate matter, as shown in Fig. [Types of radiation and their penetrating power](#penetrating).  
An $\alpha$ ray is a helium nucleus and can be stopped by a single sheet of paper. In air, it can travel only about 3 cm. Therefore, even if a source emitting $\alpha$ rays exists outside the human body, it can be shielded easily.

A $\beta$ ray is a fast electron and exhibits a continuous energy spectrum, so $\beta$ rays emitted from the same nuclide have different energies.  
A $\beta$ ray with a maximum energy of about 1–2 MeV can penetrate approximately 2–4 mm of aluminum.  
For external exposure, the main concern is damage to the skin; $\beta$ rays do not penetrate deeply into the body.  
However, when radioactive materials that emit $\alpha$ or $\beta$ rays are taken into the body, internal exposure becomes an issue.

$\gamma$ rays and X rays are electromagnetic waves with high penetrating power. Shielding them requires tens of centimeters of lead or several meters of concrete. When $\gamma$ rays or X rays enter the body, some interact with tissues and lose their energy, while the rest pass through the body. X rays are widely used for radiographic diagnosis. $\gamma$ rays can affect internal organs, and in external exposure the main concern is usually $\gamma$ rays.

Neutron radiation interacts only weakly with matter and therefore has very high penetrating power. For neutron shielding, materials containing hydrogen atoms such as water, paraffin, or concrete, which are effective in slowing down neutrons, are used.

Because of their high penetrating power and strong adverse effects on the human body, this study focuses on $\gamma$ rays as the target of measurement.

![Types of radiation and their penetrating power](Figures/chap2/Penetrating.png){#penetrating}

### Radioactive Materials

Among the above radiations, $\alpha$ rays, $\beta$ rays, and $\gamma$ rays are emitted in association with the decay of radioactive nuclides. When a nuclide undergoes $\alpha$ decay or $\beta$ decay, it transforms into a different nuclide. In $\gamma$ decay, the nuclide itself does not change, but its nuclear energy state changes. Such decay occurs only once for each nucleus, following probabilistic laws, and each radioactive nuclide has a characteristic decay rate $\lambda$, which is the probability of decay per unit time.

The rate of decrease $\frac{dN}{dt}$ of the number $N$ of undecayed nuclei is proportional to $N$ and can be expressed as:

$$
-\frac{dN}{dt} = \lambda N.
$$

where $\lambda$ is called the decay constant. If the initial number of nuclei is $N_0$, solving the above yields:

$$
N(t) = N_0 e^{-\lambda t},
$$

which shows that the number of radioactive nuclei decreases exponentially with time. The time required for the number of radioactive nuclei to decrease to one half of its initial value is defined as the half-life $T$. By substituting $N = N_0/2$ into the above equation, the half-life is obtained as:

$$
T = \frac{\ln 2}{\lambda}.
$$

Through $\beta$ decay, cesium-137 becomes the metastable nuclide barium-137m, which then emits a $\gamma$ ray and transitions to stable barium-137.

In nuclear power plant accidents and in some medical accidents, various radionuclides can be released into the environment. Typical examples relevant to this thesis include cesium-137, cesium-134, cobalt-60, europium-154, europium-155, and radium-226. These nuclides have different origins and characteristics: for instance, cesium-134 and cesium-137 are fission products, whereas cobalt-60 and europium isotopes are mainly activation products in reactor structures and medical or industrial sources, and radium-226 has historical medical and natural origins. The types of emitted radiation, representative $\gamma$-ray energies, and half-lives of these radionuclides are listed in Table [Representative $\gamma$-ray energies and half-lives](#table:half_time) [@ref_ICRP107].

Among them, cesium-137 is particularly problematic because it has a relatively long half-life of about 30 years and emits a strong 662 keV $\gamma$ ray, so cesium-137 released into the environment remains for decades and often dominates the long-term dose. Cobalt-60 and europium-155 have shorter half-lives (on the order of 5 years) than cesium-137, but they can still persist in the environment for several years after an accident and contribute appreciably to the radiation field, especially inside reactor buildings and around activated components. In realistic post-accident scenarios, these radionuclides may coexist, and appropriate countermeasures such as shielding, decontamination, and waste management depend on the specific isotopes present. Therefore, it is necessary to identify the radionuclides in the environment rather than relying only on total dose.

In this study, we focus on three representative $\gamma$-emitting radionuclides: cesium-137, cobalt-60, and europium-155. The spectrum unfolding and source-term estimation methods developed in the following chapters are designed to identify these isotopes and to estimate their spatial distributions and strengths.

**Table: Representative $\gamma$-ray energies and half-lives of selected radionuclides** [@ref_ICRP107] {#table:half_time}

| Radionuclide | Emitted radiation | Representative $\gamma$ lines [keV] | Half-life |
|---|---|---:|---|
| **Cesium-137** | $\beta^-$, **$\gamma$** | **662** | **30.1 years** |
| Cesium-134 | $\beta^-$, $\gamma$ | 605, 796 | 2.06 years |
| Cobalt-60 | $\beta^-$, $\gamma$ | 1173, 1332 | 5.27 years |
| Europium-154 | $\beta^-$, $\gamma$ | 723, 873, 996, 1275, 1494, 1596 | 8.6 years |
| Europium-155 | $\beta^-$, $\gamma$ | 86.5 | 4.76 years |
| Radium-226 | $\alpha$, $\gamma$ | 186 | 1600 years |
| Iodine-131 | $\beta^-$, $\gamma$ | 364 | 8.0 days |
| Strontium-90 | $\beta^-$ | -- | 29 years |
| Plutonium-239 | $\alpha$, $\gamma$ | 129, 375, 414 | 24,000 years |

---

## Radiation Detectors {#chap2_rad_detector}

Various radiation detectors are used for radiation measurement depending on the purpose of measurement and the type of radiation to be detected. In radiation source–term estimation, two broad classes of detectors are commonly used:

- *Non-directional* detectors, which only measure the number of incoming radiation quanta.
- *Directional* detectors, which provide both count and incident-direction information.

This section summarises representative detectors in each class and explains the reason for the detector choice adopted in this thesis.

### Non-directional Detectors

Non-directional detectors register radiation that enters the sensitive volume without resolving its direction of arrival. Because they do not require heavy collimators or imaging optics, they are typically compact, robust, and capable of operating in high-dose environments. However, the lack of directional information means that more measurement locations and longer integration times are generally required to estimate the spatial distribution of sources.

#### Dose-rate survey meters

A basic example of a non-directional detector is the dose-rate survey meter. Typical survey meters employ either a Geiger–Müller tube, an ionisation chamber, or a scintillation crystal coupled to a photomultiplier tube. They output a scalar quantity such as count rate or ambient dose equivalent rate, integrated over energy. Because the output does not contain spectral information, these instruments cannot distinguish different radionuclides; nevertheless, they are widely used for radiation safety management owing to their simplicity, robustness, and wide dynamic range.

Figure [Principle of a scintillation survey meter](#fig:chap2_scintillation) illustrates the principle of a scintillation survey meter. When a $\gamma$ ray enters the scintillator, electrons in the crystal are excited and the crystal emits scintillation light. The light is converted into electrical pulses by a photomultiplier tube, and the pulse rate is proportional to the deposited energy per unit time, enabling the measurement of radiation dose.

![Principle of a scintillation survey meter. Incident $\gamma$ rays deposit energy in the scintillator, producing scintillation light that is converted into electrical pulses by a photomultiplier tube.](Figures/chap2/Penetrating.png){#fig:chap2_scintillation width=60%}

#### Non-directional spectrometers

For source–term estimation and radionuclide identification, it is desirable to obtain not only the total count rate but also the *energy spectrum* of the detected $\gamma$ rays. Non-directional spectrometers fulfil this requirement by combining a scintillation crystal (e.g., NaI(Tl) or CeBr$_3$) or a semiconductor detector (e.g., HPGe, CdZnTe) with multichannel pulse-height analysis electronics. From the measured spectrum, the characteristic photopeaks of individual radionuclides can be identified, enabling isotopic decomposition as described in this chapter.

In this thesis, the detector mounted on the mobile robot is a compact, energy-resolving, non-directional $\gamma$-ray spectrometer based on a scintillation crystal. It provides count-rate measurements over a wide dose-rate range comparable to that of survey meters, while simultaneously recording short-time energy spectra that are later processed by the spectrum-unfolding pipeline of this chapter. An example of such a compact non-directional spectrometer is shown in Fig. [Example of a compact non-directional spectrometer](#fig:chap2_spectrometer).

![Example of a compact non-directional $\gamma$-ray spectrometer based on a CeBr$_3$ scintillation crystal. The cylindrical detector head can be mounted on a mobile robot while providing energy-resolved spectra.](Figures/chap2/Penetrating.png){#fig:chap2_spectrometer width=45%}

### Directional Detectors

Directional detectors are designed to measure not only the number of incident $\gamma$ rays but also their approximate direction of arrival. By exploiting collimation, scattering kinematics, or coded apertures, these systems can directly form images of the radiation field. Examples include gamma cameras, Compton cameras, and various collimated or coded-aperture spectrometers. In radiation source–distribution estimation, directional information greatly reduces the number of required measurement locations. However, most directional systems either require heavy shielding and collimators or have limited count-rate capability, which complicates their use on small mobile robots in high-dose environments.

#### Gamma camera

Figure [Principle of a gamma camera](#fig:chap2_gamma) shows the principle of a gamma camera, a representative directional detector. A pixelated scintillation or semiconductor detector is placed inside a lead shield with a pinhole or multi-pinhole collimator. Only $\gamma$ rays that pass through the pinhole are detected, and an inverted map of radiation intensity is formed on the detector plane. Because the count in each pixel directly corresponds to the local intensity, subsequent image reconstruction is straightforward.

![Principle of a gamma camera. A pixelated detector placed behind a lead pinhole collimator forms an inverted image of the radiation intensity on the detector plane.](Figures/chap2/Penetrating.png){#fig:chap2_gamma width=60%}

The main drawback of gamma cameras is the need for thick lead collimators to achieve sufficient angular resolution and background rejection. As a result, the total system mass typically reaches several tens of kilograms. For compact ground robots that must traverse cluttered environments with narrow passages and stairs, such payloads exceed the allowable mass, making the deployment of gamma cameras difficult in practice.

#### Compton camera

Figure [Principle of a Compton camera](#fig:chap2_compton) illustrates the principle of a Compton camera, another directional detector. Unlike gamma cameras, Compton cameras do not rely on heavy collimators and can therefore be made relatively lightweight. A typical system consists of two detector layers: a *scatterer* and an *absorber*. An incident $\gamma$ ray first undergoes Compton scattering in the scatterer; the scattered $\gamma$ ray is then absorbed in the absorber. By measuring the energy deposits and interaction positions in both layers, the direction and energy of the incident $\gamma$ ray can be reconstructed. Each detected event constrains the source to lie on a so-called Compton cone, and by superimposing many cones the source distribution can be estimated.

![Principle of a Compton camera. (a) An incident $\gamma$ ray undergoes Compton scattering in the scatterer and is absorbed in the absorber, defining a Compton cone. (b) Superposition of many cones allows the source position to be estimated.](Figures/chap2/Penetrating.png){#fig:chap2_compton width=80%}

However, Compton cameras suffer from an upper limit on the usable count rate. In high-dose environments, multiple $\gamma$ rays may enter the scatterer within a short time window, leading to overlapping interactions. When two or more Compton scattering events occur simultaneously, it becomes impossible to correctly associate the corresponding interactions in the scatterer and absorber, and the incident directions cannot be reconstructed reliably. For example, the Temporal Imaging Compton Camera V3 has an upper dose-rate limit of approximately 1 mSv/h, so it cannot be operated in the high-dose environments considered in this thesis.

### Selection of Detector

In this thesis, the goal is to estimate the three-dimensional distribution of $\gamma$-ray sources in a high-dose environment using a small ground robot. The detector must therefore satisfy the following requirements:

1. It must be mountable on a mobile robot with a limited payload capacity.  
2. It must operate reliably in high-dose environments, up to at least several Sv/h.  
3. It should provide energy-resolved spectra to enable radionuclide identification.  

Directional detectors such as gamma cameras and Compton cameras provide valuable directional information but do not meet all of these requirements simultaneously: gamma cameras are too heavy to be mounted on the robot, whereas Compton cameras suffer from low upper dose-rate limits and cannot be used in the extreme high-dose regions of interest. On the other hand, non-directional spectrometers based on scintillation crystals are lightweight, robust against high count rates, and capable of measuring energy spectra.

For these reasons, this thesis employs a non-directional, energy-resolving scintillation spectrometer mounted on a mobile robot as the primary radiation sensor. A qualitative comparison of the detector types considered is summarised in Table [Comparison of radiation detectors](#table:detector). Because the chosen detector does not provide inherent directional information, the subsequent chapters develop methods that exploit attenuation by lightweight, actively rotated shields and environmental structures to recover pseudo-directional information and to perform three-dimensional source–term estimation.

**Table: Comparison of radiation detectors** {#table:detector}

| Type | Mountable on robot | High-dose environment | Energy spectrum | Directionality |
|---|---:|---:|---:|---|
| **Non-directional spectrometer (this work)** | ○ | ○ | ○ | Non-directional |
| Simple survey meter (GM / ionisation) | ○ | ○ | × | Non-directional |
| Gamma camera | × | ○ | ○ | Directional |
| Compton camera | ○ | × | ○ | Directional |
| Coded-aperture / scanned collimated detector | △ | △ | ○ | Directional |

---

## Spectrum Modeling and Problem Formulation {#chap2_spectrum_problem}

Consider a stationary or mobile gamma-ray detector that measures an energy spectrum in an environment where one or more radionuclides may be present. The detector output is discretised into $K$ energy bins. Let $\tilde{y}_{k}$ denote the observed number of counts in the $k$-th energy bin with central energy $E_{k}$ and width $\Delta E_{k}$ ($k = 1,\dots,K$). The measured spectrum is written as:

$$
\tilde{\bm{y}} = (\tilde{y}_{1}, \tilde{y}_{2}, \dots, \tilde{y}_{K})^{\mathrm{T}} .
$$

Assume that there exist $M \geq 1$ candidate radionuclides. Let $q_{j}$ denote a parameter proportional to the activity of the $j$-th radionuclide (for example, the activity at the detector position or the source strength after geometric attenuation), and define the vector:

$$
\bm{q} = (q_{1}, q_{2}, \dots, q_{M})^{\mathrm{T}} .
\tag{1}\label{eq:spectrum_q}
$$

Following Kemp *et al.* [@ref_kemp2023_tns], the expected number of counts in each energy bin is modelled as a linear combination of contributions from each radionuclide and from background:

$$
\bm{\mu}(\bm{q}) = \bm{R}\bm{q} + \bm{b} ,
\tag{2}\label{eq:spectrum_mu}
$$

where $\bm{R} \in \mathbb{R}^{K \times M}$ is the detector response matrix and

$$
\bm{b} = (b_{1}, b_{2}, \dots, b_{K})^{\mathrm{T}}
$$

is the background spectrum (including environmental and intrinsic backgrounds).

The aim of spectrum unfolding is to estimate $\bm{q}$ from $\tilde{\bm{y}}$ and to determine which radionuclides are present. In the following, the structure of the response matrix $\bm{R}$ and the statistical model of the spectrum are detailed.

### Detector Response Matrix {#subsec:spectrum_response}

For the $j$-th radionuclide, assume that the nuclear data library provides $L_{j}$ discrete gamma lines. The $\ell$-th line has energy $E_{j\ell}$ and emission probability (branching ratio) $\beta_{j\ell}$ per decay. Let $\epsilon(E)$ denote the full-energy peak detection efficiency at energy $E$, and let $\sigma(E)$ denote the energy resolution (standard deviation) of the detector, which is often approximated as [@ref_Tsoulfanidis1995]:

$$
\sigma(E) = a\sqrt{E} + b ,
$$

where $a$ and $b$ are calibration constants.

Assuming a Gaussian full-energy peak shape, the contribution of isotope $j$ to bin $k$ is modelled as:

$$
R_{kj}
  = \sum_{\ell = 1}^{L_{j}} \beta_{j\ell}\,\epsilon(E_{j\ell})\,G\!\left(E_{k}; E_{j\ell}, \sigma(E_{j\ell})\right),
\tag{3}\label{eq:spectrum_Rkj}
$$

where $G(E; \mu, \sigma)$ is the normalised Gaussian function:

$$
G(E; \mu, \sigma)
  = \frac{1}{\sqrt{2\pi}\sigma}
    \exp\!\left(-\frac{(E-\mu)^{2}}{2\sigma^{2}}\right).
$$

In practice, $R_{kj}$ may incorporate not only full-energy peaks but also Compton continua and scattering from the environment, either by Monte Carlo simulation or by experimental calibration [@ref_kemp2023_tns].

Collecting $R_{kj}$ for all bins and isotopes yields the response matrix:

$$
\bm{R} =
\begin{pmatrix}
    R_{11} & \dots & R_{1M} \\
    \vdots &       & \vdots \\
    R_{K1} & \dots & R_{KM}
\end{pmatrix}.
\tag{4}\label{eq:spectrum_R_matrix}
$$

Using Eq. \eqref{eq:spectrum_mu}, the expected count in bin $k$ is then:

$$
\mu_{k}(\bm{q}) = \sum_{j=1}^{M} R_{kj} q_{j} + b_{k}.
$$

### Statistical Model {#subsec:spectrum_stat}

As gamma rays are emitted by stochastic radioactive decay processes, the number of counts in each bin is modelled as a Poisson random variable [@ref_Tsoulfanidis1995]. Thus, for a given $\bm{q}$,

$$
\tilde{y}_{k} \sim \mathrm{Poisson}\!\left( \mu_{k}(\bm{q}) \right) \quad (k=1,\dots,K),
\tag{5}\label{eq:spectrum_poisson}
$$

where $\mu_{k}(\bm{q})$ is the $k$-th element of $\bm{\mu}(\bm{q})$.

Under the Poisson assumption, the likelihood of observing $\tilde{y}_{k}$ counts in bin $k$ given $\bm{q}$ is:

$$
p(\tilde{y}_{k} \mid \bm{q})
  = \frac{\mu_{k}(\bm{q})^{\tilde{y}_{k}} \exp\!\left( -\mu_{k}(\bm{q}) \right)}{\tilde{y}_{k}!} .
$$

Assuming independence between energy bins, the joint likelihood for the whole spectrum is:

$$
p(\tilde{\bm{y}} \mid \bm{q})
  = \prod_{k=1}^{K}
     \frac{\mu_{k}(\bm{q})^{\tilde{y}_{k}} \exp\!\left( -\mu_{k}(\bm{q}) \right)}
          {\tilde{y}_{k}!} .
\tag{6}\label{eq:spectrum_likelihood}
$$

In practice, however, direct optimisation of Eq. \eqref{eq:spectrum_likelihood} over all bins and isotopes can be numerically challenging. Therefore, as in Kemp *et al.* [@ref_kemp2024_tns], a peak-based unfolding approach is adopted, in which the spectrum is first decomposed into individual peaks and then mapped to radionuclides.

---

## Spectrum Unfolding Procedure {#chap2_spectrum_procedure}

This section describes the practical steps used to unfold the measured spectrum and to identify radionuclides. The procedure consists of the following steps:

1. preprocessing (energy calibration and smoothing),  
2. peak detection,  
3. baseline estimation and net peak area computation,  
4. decomposition of overlapping peaks (spectral stripping),  
5. dead-time correction,  
6. radionuclide library matching and identification,  
7. construction of isotope-wise count sequences from successive spectra,  
8. estimation of relative activities (optional).  

These steps closely follow the approach of Kemp *et al.* [@ref_kemp2024_tns] and Anderson *et al.* [@ref_anderson2022_tase].

### Preprocessing: Energy Calibration and Smoothing {#subsec:spectrum_preprocessing}

#### Energy Calibration

The raw detector output is given in channel number $c$. A polynomial calibration function is used to convert channel numbers to energy:

$$
E(c) = a_{0} + a_{1}c + a_{2}c^{2},
\tag{7}\label{eq:spectrum_calib}
$$

where $a_{0}, a_{1}, a_{2}$ are calibration coefficients. These coefficients are estimated by least squares using one or more reference radionuclides with well-known peak energies. Let $(c_{r\ell}, E_{r\ell})$ be the channel location and known energy of the $\ell$-th reference peak. The calibration parameters are obtained by:

$$
\hat{\bm{a}}
  = \underset{\bm{a}}{\operatorname{argmin}}
    \sum_{\ell}
    \left( E_{r\ell} - (a_{0} + a_{1}c_{r\ell} + a_{2}c_{r\ell}^{2}) \right)^{2},
\tag{8}\label{eq:spectrum_calib_ls}
$$

where $\bm{a} = (a_{0}, a_{1}, a_{2})^{\mathrm{T}}$.

#### Smoothing

To suppress high-frequency statistical noise without significantly distorting peak shapes, the spectrum is smoothed by convolution with a Gaussian kernel:

$$
\tilde{y}^{(\mathrm{sm})}_{k}
  = \sum_{m=-M}^{M} h_{m}\,\tilde{y}_{k-m},
\tag{9}\label{eq:spectrum_smoothing}
$$

where the kernel coefficients $h_{m}$ are given by:

$$
h_{m}
  = \frac{1}{\sum_{r=-M}^{M} \exp\!\left(-\frac{r^{2}}{2\sigma_{\mathrm{sm}}^{2}}\right)}
    \exp\!\left(-\frac{m^{2}}{2\sigma_{\mathrm{sm}}^{2}}\right),
$$

and $\sigma_{\mathrm{sm}}$ controls the smoothing strength. Kemp *et al.* [@ref_kemp2023_tns] report that moderate smoothing improves peak detection robustness while preserving resolution.

### Peak Detection {#subsec:spectrum_peaks}

Peaks are detected on the smoothed spectrum $\tilde{y}^{(\mathrm{sm})}_{k}$. One classical approach is the second-derivative method proposed by Mariscotti [@ref_Mariscotti1967]. Define the discrete second difference:

$$
D^{2}\tilde{y}^{(\mathrm{sm})}_{k}
  = \tilde{y}^{(\mathrm{sm})}_{k+1}
    - 2\tilde{y}^{(\mathrm{sm})}_{k}
    + \tilde{y}^{(\mathrm{sm})}_{k-1}.
$$

For a locally linear background $B(E) = \alpha + \beta E$, the second difference satisfies $D^{2}B \approx 0$; therefore, a negative value of $D^{2}\tilde{y}^{(\mathrm{sm})}_{k}$ indicates the presence of a peak. Using a noise estimate $\sigma_{D^{2}}$, a channel $k$ is declared a peak candidate if:

$$
-D^{2}\tilde{y}^{(\mathrm{sm})}_{k} > \lambda_{\mathrm{th}}\,\sigma_{D^{2}},
$$

where $\lambda_{\mathrm{th}}$ is a user-defined signal-to-noise threshold [@ref_Mariscotti1967].

In this thesis, peak detection is implemented using a Gaussian-matched filter. The filter correlates the spectrum with a normalised Gaussian template of width equal to the detector resolution, and local maxima in the filter response above a certain threshold are selected as peak candidates. This approach has been shown by Kemp *et al.* [@ref_kemp2024_tns] to perform robustly in the presence of Compton continua and moderate statistical noise.

### Baseline Estimation and Net Peak Area {#subsec:spectrum_baseline}

Gamma-ray spectra typically exhibit a significant continuous component due to Compton scattering and environmental backgrounds. Accurate radionuclide identification requires subtracting this baseline and computing the net area of each photopeak.

#### Baseline Estimation

The baseline is estimated using an asymmetric least-squares (ALS) smoothing method, which penalises deviations of the estimated baseline above the measured spectrum more strongly than deviations below it. Let $y_{k} = \tilde{y}^{(\mathrm{sm})}_{k}$. The baseline $\hat{b}_{k}$ is obtained by solving:

$$
\hat{\bm{b}} =
\underset{\bm{b}}{\operatorname{argmin}}\;
\sum_{k=1}^{K} w_{k}(y_{k} - b_{k})^{2}
+ \lambda \sum_{k=2}^{K-1} (\Delta^{2} b_{k})^{2},
\tag{10}\label{eq:spectrum_baseline_als}
$$

where $\Delta^{2}b_{k} = b_{k+1} - 2b_{k} + b_{k-1}$ is the discrete second difference and $\lambda$ is a smoothness parameter. The weights $w_{k}$ are updated iteratively according to:

$$
w_{k} =
\begin{cases}
    p    & \text{if } y_{k} > b_{k}, \\
    1-p  & \text{otherwise},
\end{cases}
$$

with $0 < p < 1$. This choice forces the baseline to lie predominantly below the data, thus preserving peaks.

#### Net Peak Area

For each detected peak, a local fitting window $\mathcal{W}_{p}$ centred at energy $E_{p}$ is defined, typically covering $\pm 3\sigma_{p}$ where $\sigma_{p}$ is estimated from the detector resolution. Within this window, the peak shape is modelled as a Gaussian:

$$
g_{p}(E; A_{p}, E_{p}, \sigma_{p})
  = A_{p}\exp\!\left(-\frac{(E-E_{p})^{2}}{2\sigma_{p}^{2}}\right),
$$

where $A_{p}$ is the amplitude. The total model in the window is:

$$
y_{k} \approx \hat{b}_{k} + g_{p}(E_{k}; A_{p}, E_{p}, \sigma_{p}) .
\tag{11}\label{eq:spectrum_peak_fit}
$$

The parameters $(A_{p}, E_{p})$ are obtained by least-squares fitting.

The net peak area $N_{p}$ is then given by:

$$
N_{p}
  = \sqrt{2\pi}\,A_{p}\sigma_{p},
\tag{12}\label{eq:spectrum_net_area_gauss}
$$

or equivalently by discrete summation above the baseline:

$$
N_{p}
  = \sum_{k \in \mathcal{W}_{p}} \left( y_{k} - \hat{b}_{k} \right).
\tag{13}\label{eq:spectrum_net_area_sum}
$$

Assuming Poisson statistics, the variance of $N_{p}$ can be approximated as:

$$
\sigma_{N_{p}}^{2}
  \approx \sum_{k \in \mathcal{W}_{p}} y_{k}.
\tag{14}\label{eq:spectrum_net_area_var}
$$

These net areas and their uncertainties are the primary inputs for radionuclide identification.

### Decomposition of Overlapping Peaks {#subsec:spectrum_stripping}

When peaks from different radionuclides overlap due to limited energy resolution, their contributions must be separated. Kemp *et al.* [@ref_kemp2023_tns] employ a spectral stripping approach based on known intensity ratios from the nuclear data library.

For each radionuclide $j$, one *reference line* (e.g., the most intense or least overlapped peak) with energy $E_{jr}$ is chosen. Let $N_{jr}$ be its net area. For another line $\ell$ of the same radionuclide, the expected net area is:

$$
\hat{N}_{j\ell} = r_{j\ell} N_{jr},
\tag{15}\label{eq:spectrum_ratio}
$$

where

$$
r_{j\ell}
  = \frac{\beta_{j\ell}\,\epsilon(E_{j\ell})}
         {\beta_{jr}\,\epsilon(E_{jr})}
$$

is the intensity ratio corrected for detector efficiency. Suppose that an observed peak at energy $E_{i}$ is the sum of contributions from multiple radionuclides:

$$
N_{i}^{\mathrm{obs}} = \sum_{j} N_{ij}.
$$

If a subset of radionuclides has already been quantified via their reference lines, their contributions $\hat{N}_{ij}$ can be predicted by Eq. \eqref{eq:spectrum_ratio} and subtracted:

$$
N_{i}^{\mathrm{res}}
  = N_{i}^{\mathrm{obs}} - \sum_{j\in\mathcal{J}_{\mathrm{known}}} \hat{N}_{ij},
\tag{16}\label{eq:spectrum_stripping}
$$

where $\mathcal{J}_{\mathrm{known}}$ is the set of radionuclides whose reference peaks are already assigned. The residual area $N_{i}^{\mathrm{res}}$ is then used to estimate the remaining radionuclides. For example, Kemp *et al.* [@ref_kemp2023_tns] strip the contribution of the 609 keV line of the uranium/radium decay chain from the 662 keV region before quantifying $\ce{^{137}Cs}$.

This stripping can be expressed as a linear system. Let $\bm{N}^{\mathrm{obs}}$ be the vector of observed peak areas and $\bm{\theta}$ the vector of reference-line areas for all radionuclides. Then:

$$
\bm{N}^{\mathrm{obs}} \approx \bm{S}\bm{\theta},
$$

where $\bm{S}$ is a matrix of intensity ratios. The least-squares estimate of $\bm{\theta}$ is:

$$
\hat{\bm{\theta}}
  = \underset{\bm{\theta} \ge 0}{\operatorname{argmin}}
    \left\| \bm{N}^{\mathrm{obs}} - \bm{S}\bm{\theta} \right\|_{2}^{2}.
$$

The component $\hat{\theta}_{j}$ corresponding to radionuclide $j$ determines all its peak areas via Eq. \eqref{eq:spectrum_ratio}.

### Dead-Time Correction {#subsec:spectrum_deadtime}

At high count rates, the detector and data acquisition system exhibit a dead time $\tau_{d}$ during which additional pulses are not recorded. For a non-paralysable system, the relationship between the true count rate $n$ and the measured count rate $m$ is [@ref_Tsoulfanidis1995]:

$$
m = \frac{n}{1 + n\tau_{d}} .
\tag{17}\label{eq:spectrum_deadtime}
$$

Solving for $n$ yields:

$$
n = \frac{m}{1 - m\tau_{d}} .
$$

Kemp *et al.* [@ref_kemp2023_tns] apply this correction to the total count rate using a measured dead time of $\tau_{d} = 5.813\times 10^{-9}\,\mathrm{s}$ for their detector.

Let $T$ be the live time of a measurement. The total measured count rate is $m_{\mathrm{tot}} = N_{\mathrm{tot}}/T$, where $N_{\mathrm{tot}}$ is the total number of recorded counts. The corrected total count rate is:

$$
n_{\mathrm{tot}} = \frac{m_{\mathrm{tot}}}{1 - m_{\mathrm{tot}}\tau_{d}} ,
$$

and the corrected total number of counts is $N_{\mathrm{tot}}^{\mathrm{corr}} = n_{\mathrm{tot}}T$. Assuming that dead time affects all energy bins proportionally, each bin is scaled by the factor:

$$
f_{\mathrm{DT}}
  = \frac{N_{\mathrm{tot}}^{\mathrm{corr}}}{N_{\mathrm{tot}}}
  = \frac{1}{1 - m_{\mathrm{tot}}\tau_{d}} .
\tag{18}\label{eq:spectrum_deadtime_factor}
$$

Thus the dead-time-corrected bin counts are:

$$
\tilde{y}^{(\mathrm{corr})}_{k}
  = f_{\mathrm{DT}} \tilde{y}_{k},
$$

and the same factor is applied to peak areas.

### Radionuclide Library Matching and Identification {#subsec:spectrum_matching}

Having obtained a set of peak energies $\{E_{p}\}$, net areas $\{N_{p}\}$, and uncertainties $\{\sigma_{N_{p}}\}$, each peak must be associated with candidate gamma lines from the radionuclide library.

#### Energy Matching

Following Anderson *et al.* [@ref_anderson2022_tase], the difference between a measured peak energy $E_{\alpha}$ and a library line energy $E_{\beta}$ is denoted by:

$$
d = \lvert E_{\alpha} - E_{\beta} \rvert .
$$

Let $\sigma_{\alpha}$ and $\sigma_{\beta}$ denote the standard uncertainties of the measured and library energies, respectively. To account for calibration drift and energy-dependent resolution, an empirical factor $H(E)$ is introduced [@ref_anderson2022_tase]:

$$
H(E) = 1 + \frac{E}{4000},
$$

where $E$ is the average energy of the pair. Define the variance of $d$ as:

$$
\sigma_{d}^{2} = \zeta H(E)\left(\sigma_{\alpha}^{2} + \sigma_{\beta}^{2}\right),
$$

where $\zeta$ is a tuning parameter. Assuming $d$ is normally distributed with mean zero and variance $\sigma_{d}^{2}$, the normalised deviation is:

$$
t = \frac{d}{\sigma_{d}}.
$$

Let $\Phi(\cdot)$ denote the cumulative distribution function of the standard normal distribution. The two-sided tail probability is then:

$$
Z(d) = 2\left[1 - \Phi(t)\right].
\tag{19}\label{eq:spectrum_Z}
$$

A small value of $Z(d)$ indicates a good match between the measured peak and the library line. If $Z(d) < Z_{\mathrm{th}}$ for a predefined threshold $Z_{\mathrm{th}}$, the pair $(\alpha,\beta)$ is considered a possible association.

#### Peak–Isotope Association and Detection Probability

For each radionuclide $j$, let $\mathcal{P}_{j}$ be the set of peaks whose energies are compatible with at least one gamma line of $j$ according to the above criterion. Anderson *et al.* [@ref_anderson2022_tase] employ a Bayesian framework based on the method of Stinnett and Sullivan to compute the probability that radionuclide $j$ is present. Conceptually, for each isotope $j$, a likelihood ratio is constructed between the hypotheses:

- $H_{0}: \; q_{j} = 0$  
- $H_{1}: \; q_{j} > 0$  

using the peak areas in $\mathcal{P}_{j}$ and their expected values under each hypothesis. The posterior probability of presence is then:

$$
P(H_{1} \mid \mathrm{data})
  = \frac{L(\mathrm{data} \mid H_{1})\,P(H_{1})}
         {L(\mathrm{data} \mid H_{0})\,P(H_{0})
          + L(\mathrm{data} \mid H_{1})\,P(H_{1})},
$$

where $P(H_{0})$ and $P(H_{1})$ are prior probabilities and $L(\cdot)$ denotes the likelihood. A radionuclide is declared present if this posterior probability exceeds a threshold (e.g., $0.9$).

### Isotope-wise Count Sequence from Spectra {#subsec:spectrum_isotope_counts}

For source–term estimation in later chapters, it is convenient to summarise each short-time spectrum as a compact vector of isotope-wise counts. This subsection describes how such counts are constructed from the unfolded spectra obtained by the procedures in Sections [Preprocessing](#subsec:spectrum_preprocessing)–[Radionuclide Library Matching](#subsec:spectrum_matching).

We assume that an energy-resolving detector (e.g., a scintillation detector) is used and that the gamma-ray spectrum is recorded at each measurement time along the robot trajectory. Let $\tilde{z}_{k,c}$ denote the number of counts in channel $c\in\{1,\dots,C\}$ at time step $k$. After energy calibration, each channel corresponds to an energy $E_c$.

A library of candidate isotopes $\mathcal{H}$ is prepared. Each isotope $h\in\mathcal{H}$ has a set of characteristic photopeaks with energies $E_{h,p}$ and relative intensities (branching ratios) $r_{h,p}$, where $p=1,\dots,P_h$. Using the peak-finding, baseline-subtraction, and deconvolution procedures described earlier, each spectrum is analysed to identify these photopeaks. For each peak $p$ associated with isotope $h$, an energy window $\mathcal{C}_{h,p}$ (a set of channels) is defined and the corresponding peak count at time $k$ is obtained as:

$$
\tilde{y}_{k,h,p} = \sum_{c\in \mathcal{C}_{h,p}} \tilde{z}_{k,c}.
\tag{20}\label{eq:pf_peak_counts}
$$

Here, $\tilde{z}_{k,c}$ can be regarded as the dead-time-uncorrected, baseline-subtracted counts in channel $c$.

Dead-time correction is applied to the recorded peak counts using the non-paralysable model described in Section [Dead-Time Correction](#subsec:spectrum_deadtime), and the corrected values are again denoted by $y_{k,h,p}$ for simplicity.

The contributions of multiple peaks belonging to the same isotope $h$ are then aggregated using the branching ratios as weights. The total count for isotope $h$ at time $k$ is defined as:

$$
z_{k,h}
= \sum_{p=1}^{P_h} w_{h,p} y_{k,h,p},
\qquad
w_{h,p} = \frac{r_{h,p}}{\sum_{p'=1}^{P_h} r_{h,p'}}.
\tag{21}\label{eq:pf_isotope_counts}
$$

Thus, for each time step $k$, we obtain the isotope-wise count vector:

$$
\bm{z}_k = \{z_{k,h}\}_{h\in\mathcal{H}}.
\tag{22}\label{eq:pf_isotope_vector}
$$

These isotope-wise counts summarise the unfolded spectra in a form that is directly usable as observations for a set of parallel particle filters, one PF per isotope, in the multi-isotope source–term estimation framework of Chapter 3 [@ref_kemp2023_tns; @ref_kemp2024_tns].

---

## Summary {#chap2_summary}

In this chapter, the gamma-ray spectrum unfolding and radionuclide identification method employed in this thesis was described.  
Section [Overview of Radiation](#chap2_radiation) reviewed basic properties of ionising radiation, introduced representative radioactive materials, and highlighted key $\gamma$-emitting radionuclides relevant to post-accident environments.  
Section [Radiation Detectors](#chap2_rad_detector) compared non-directional and directional radiation detectors and justified the use of a compact, non-directional scintillation spectrometer mounted on a mobile robot for high-dose measurements.  
Section [Spectrum Modeling and Problem Formulation](#chap2_spectrum_problem) modelled the measured spectrum using a detector response matrix and Poisson counting statistics, formulating spectrum unfolding as a linear inverse problem over radionuclide activities.  
Section [Spectrum Unfolding Procedure](#chap2_spectrum_procedure) detailed the practical peak-based unfolding procedure, including preprocessing, peak detection, baseline and overlapping-peak treatment, dead-time correction, radionuclide library matching, construction of isotope-wise count sequences, and estimation of relative activities.  
These elements together provide the spectral analysis foundation required for the three-dimensional source-term estimation developed in the following chapters.

<!--
Chapter 3 (converted from LaTeX to Markdown)
Notes:
- LaTeX labels are kept as Pandoc-style identifiers {#...} where practical.
- Citations are kept as [@key].
- Math is kept in LaTeX math blocks for compatibility with Pandoc / MathJax.
-->

# Online Radiation Source Distribution Estimation Using a Particle Filter with Rotating Shields {#chap:chap3}

## Introduction {#chap_pf_introduction}

In this chapter, we propose an online method for estimating the three-dimensional distribution of multiple $\gamma$-ray sources using a particle filter (PF) combined with actively rotating shields.  
A mobile robot equipped with an energy-resolving, non-directional detector and lightweight iron and lead shields moves in a high-dose indoor environment and acquires radiation measurements while continuously changing the shield orientations.  
By exploiting geometric spreading and controlled attenuation due to the shields, the method recovers pseudo-directional information from non-directional measurements and estimates the locations and strengths of multiple $\gamma$-ray sources.

The previous chapter focused on gamma-ray spectrum unfolding and radionuclide identification. In particular, Section `subsec:spectrum_isotope_counts` defined how each short-time spectrum is converted into an isotope-wise count vector $\bm{z}_k$.  
In this chapter, these isotope-wise count sequences are used as PF observations to infer spatial source distributions for each isotope in real time. The radiation transport between sources and detector is modelled using an inverse-square law and shield-dependent attenuation kernels, and the resulting Poisson count model is embedded in a Bayesian PF framework. In addition, an active sensing strategy selects shield orientations and robot poses based on information-theoretic criteria so that measurements are collected where they are most informative for source-term estimation.

- Section [Measurement Model and Shield Design](#chap3_pf_model) introduces the physical and mathematical model of non-directional count measurements with a shielded detector and discusses the design of lightweight lead shielding under robot payload constraints.  
- Section [Particle Filter Formulation for Multi–Isotope Source–Term Estimation](#chap3_pf_pf) formulates multi-isotope source-term estimation as Bayesian inference with parallel PFs, including the definition of precomputed geometric and shielding kernels, state representation, prediction, log-domain weight update for Poisson observations, resampling, regularisation, and spurious-source rejection.  
- Section [Shield Rotation Strategy](#chap3_pf_rotation_section) presents the shield-rotation strategy, which selects informative shield orientations and next robot poses using information-gain and Fisher-information-based criteria together with short-time measurements.  
- Section [Convergence Criteria and Output](#chap3_pf_conclusion) summarises the convergence criteria and the final outputs of the algorithm, which provide radiation maps for subsequent visualisation and decision making.  
- Finally, Section [Summary](#chap3_summary) presents a summary of this chapter.

---

## Measurement Model and Shield Design {#chap3_pf_model}

### Macroscopic Attenuation and Shield Thickness {#chap3_shield}

Although $\gamma$ rays are electromagnetic waves, their high energy also gives them particle-like properties. A macroscopic view of the interaction between $\gamma$ rays and matter considers an incident beam on a flat slab of material. When $\gamma$ rays are incident on the material, some are absorbed and some undergo scattering, changing their direction and energy.

Consider a thin layer of thickness $dX$ inside the material, through which $N$ $\gamma$ rays are passing. Let $N'$ be the number of $\gamma$ rays that pass through this layer without interaction. The number of $\gamma$ rays that interact in the layer, $-dN = N - N'$, is proportional to both the thickness $dX$ of the layer and the number $N$ of incident $\gamma$ rays. Using the proportionality constant $\mu$, this relationship can be written as:

$$
-dN = \mu \, N \, dX ,
\tag{1}\label{eq:chap2_attenuation_differential}
$$

where $\mu$ is called the linear attenuation coefficient and represents the ease with which interactions occur.

If $N_{0}$ is the number of $\gamma$ rays incident on the slab, solving Eq. \eqref{eq:chap2_attenuation_differential} yields:

$$
N = N_{0} e^{-\mu X} .
\tag{2}\label{eq:chap2_attenuation}
$$

The radiation dose measured by a detector obeys the inverse-square law with respect to the distance between a point source and the detector [@ref_Tsoulfanidis1995]. Let the detector position at the $k$-th measurement be:

$$
\bm{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^{\top},
$$

and let a point source of intensity $q_j$ be located at:

$$
\bm{s}_j = [x_j, y_j, z_j]^{\top}.
$$

Defining the distance:

$$
d_{k,j} = \sqrt{(x_k^{\mathrm{det}}-x_j)^{2} + (y_k^{\mathrm{det}}-y_j)^{2} + (z_k^{\mathrm{det}}-z_j)^{2}},
$$

the radiation intensity $I(\bm{q}_k)$ measured at $\bm{q}_k$ is given by:

$$
I(\bm{q}_k) = \frac{S q_{j}}{4\pi d_{k,j}^{2}} e^{-\mu_{\mathrm{air}} d_{k,j}} ,
\tag{3}\label{eq:chap2_inverse_square_law}
$$

where $S$ is the detector area and $\mu_{\mathrm{air}}$ is the linear attenuation coefficient of air. For the $\gamma$ rays emitted by cesium-137, which is one of the main radiation sources considered in this study, the linear attenuation coefficient in air at 20$^\circ$C is approximately $9.7\times10^{-3}$ m$^{-1}$ [@ref_Tsoulfanidis1995]. In this study, measurements are performed relatively close to the radiation sources, and thus we approximate $e^{-\mu_{\mathrm{air}} d_{k,j}} \approx 1$ and neglect attenuation in air. We also neglect attenuation by environmental obstacles, such as walls and equipment, and explicitly model only the attenuation due to the lightweight shield mounted on the robot.

The thickness of a shielding material required to reduce the dose rate by half is called the half-value layer, and the thickness required to reduce the dose rate to one-tenth is called the tenth-value layer. Table [Shield thickness required to reduce dose rate](#table:half_shield) lists the half-value and tenth-value layers of lead, iron, and concrete for $\gamma$ rays emitted by cesium-137 [@ref_ICRP21].

**Table: Shield thickness required to reduce dose rate [mm]** [@ref_ICRP21] {#table:half_shield}

| Material | Half-value layer | Tenth-value layer |
|---|---:|---:|
| Lead | 7 | 22 |
| Iron | 15 | 50 |
| Concrete | 49 | 163 |

### Mathematical Model of Non-directional Count Measurements {#chap3_radation}

We assume that $M \geq 1$ unknown radiation point sources exist in the environment. Let the position of the $j$-th source be $\bm{s}_j = [x_{j}, y_{j}, z_{j}]^{\top}$ and its strength be $q_{j} \ge 0$. The radiation source distribution is represented by the vector:

$$
\bm{q} = (q_{1}, q_{2}, \dots, q_{M})^{\top} .
\tag{4}\label{eq:chap3_thetaj}
$$

A non-directional detector mounted on the robot acquires $N_{\mathrm{meas}}$ measurements at positions:

$$
\bm{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^{\top},
\qquad k = 1,\dots,N_{\mathrm{meas}}.
$$

Using the distance $d_{k,j}$ defined in Eq. \eqref{eq:chap2_inverse_square_law}, and neglecting attenuation in air and environmental obstacles, the inverse-square law implies that the contribution of a unit-strength source at $\bm{s}_j$ to the detector at pose $\bm{q}_k$ is proportional to $1/d_{k,j}^{2}$. We absorb the detector area, acquisition time, and conversion factors between source strength and count rate into a single constant $\Gamma$ and define:

$$
A_{k,j} = \frac{\Gamma}{d_{k,j}^{2}} ,
\tag{5}\label{eq:chap3_bij}
$$

which represents the expected count at pose $k$ from a unit-strength source at source $j$.

The expected total count at pose $k$ due to the full source distribution $\bm{q}$ is then:

$$
\Lambda_{k}(\bm{q}) = \sum_{j=1}^{M} A_{k,j} q_{j} .
\tag{6}\label{eq:chap3_bi}
$$

Collecting all measurement poses, we define the vector of expected counts:

$$
\boldsymbol{\Lambda}(\bm{q})
= [\Lambda_{1}(\bm{q}), \Lambda_{2}(\bm{q}), \dots, \Lambda_{N_{\mathrm{meas}}}(\bm{q})]^{\top}.
\tag{7}\label{eq:chap3_bq}
$$

Let $\bm{A} \in \mathbb{R}^{N_{\mathrm{meas}}\times M}$ be the matrix with elements $A_{k,j}$. In matrix form, the measurement model can be written as:

$$
\bm{A} =
\begin{pmatrix}
  A_{1,1} & \dots & A_{1,M} \\
  \vdots  &       & \vdots  \\
  A_{N_{\mathrm{meas}},1} & \dots & A_{N_{\mathrm{meas}},M}
\end{pmatrix},
$$

and:

$$
\boldsymbol{\Lambda}(\bm{q}) = \bm{A}\bm{q} .
\tag{8}\label{eq:chap3_b_Aq}
$$

When no shield is present, this linear model is consistent with the geometric term $G_{k,j}$ in Eq. \eqref{eq:pf_geometry} and with the kernel $K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$ in Eq. \eqref{eq:pf_kernel} for a single isotope.

As discussed in Chapter 2, radiation is emitted by stochastic radioactive decay processes, and the number of counts recorded by the detector follows a Poisson distribution [@ref_Tsoulfanidis1995]. Let $z_{k}$ denote the observed count at the $k$-th measurement. The likelihood of observing $z_{k}$ given the source distribution $\bm{q}$ is:

$$
p(z_{k} \mid \bm{q})
  = \frac{\Lambda_{k}(\bm{q})^{\,z_{k}} \exp\!\big(-\Lambda_{k}(\bm{q})\big)}{z_{k}!} .
\tag{9}\label{eq:chap3_equation4}
$$

This single-isotope, count-only Poisson model is extended to isotope-wise counts $z_{k,h}$ and shield-dependent kernels $K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$ in later sections.

### Shield Mass and Robot Payload {#chap3_shield_weight}

The densities of various materials at 20$^\circ$C are listed in Table [Material properties](#table:density). For a given attenuation level (e.g., a given number of half-value or tenth-value layers), the required mass is essentially independent of the material: denser materials require thinner shields, and less dense materials require thicker shields. Therefore, when mounting a shield on a mobile robot with a limited payload capacity, it is desirable to use the material with the highest density. In this study, we choose lead as the shielding material to be mounted on the mobile robot.

**Table: Material properties** [@ref_NIST_XCOM] {#table:density}

| Material | Density |
|---|---:|
| Lead | 11.36 g/cm$^{3}$ |
| Iron | 7.87 g/cm$^{3}$ |
| Concrete | 2.1 g/cm$^{3}$ |

In this study, a lightweight shield is used to partially cover the non-directional detector. Therefore, the proposed shield design satisfies the payload constraints of the mobile robot and is feasible for deployment in real environments.

![Configuration of the non-directional detector and lightweight shields used in this thesis. A compact non-directional $\gamma$-ray detector is placed at the center, and lightweight lead and iron shields partially surround the detector. By rotating these shields during measurement, the incident $\gamma$-ray flux from each direction is modulated, providing pseudo-directional information while keeping the total payload within the robot limits.](Figures/chap3/Detector.eps){#fig:chap3_detector_shield width=90%}

Figure [Configuration of the detector and shields](#fig:chap3_detector_shield) illustrates the detector and shield configuration assumed in this chapter. The non-directional detector is mounted at the center of the assembly, while a partial lead shell and a partial iron shell surround the detector over an angular range $\alpha$. During operation, these shields are actively rotated around the detector so that the attenuation factor in Eq. \eqref{eq:pf_shield_attn} changes with time. The particle filter exploits this controlled modulation of the count rate to recover pseudo-directional information from the non-directional detector.

---

## Particle Filter Formulation for Multi–Isotope Source–Term Estimation {#chap3_pf_pf}

In this section we formulate the estimation of multiple three-dimensional point sources as Bayesian inference with parallel PFs, one PF for each isotope. The PFs use the isotope-wise counts defined in Eq. `eq:pf_isotope_vector` (Chapter 2) and the precomputed geometric and shielding kernels to infer the number, locations, and strengths of sources and the background rate.

### Precomputed Geometric and Shielding Kernels {#chap3_pf_kernel}

To efficiently evaluate expected counts for different robot poses and shield orientations, we precompute attenuation kernels that encode geometric spreading and shielding attenuation for point sources. In contrast to grid-based methods, the PF in this thesis represents the radiation field directly as a finite set of point sources whose positions are state variables; no voxelisation of the environment is required.

For isotope $h$, let the $j$-th source position be:

$$
\bm{s}_{h,j} = [x_{h,j}, y_{h,j}, z_{h,j}]^{\top},
\qquad j = 1,\dots,r_h ,
$$

where $r_h$ is the (unknown) number of sources of isotope $h$. The robot measurement poses (detector positions) are denoted by:

$$
\bm{q}_k = [x^{\mathrm{det}}_k, y^{\mathrm{det}}_k, z^{\mathrm{det}}_k]^{\top},
\qquad k = 1,\dots,N_{\mathrm{meas}}.
$$

The distance and direction from source $j$ to pose $k$ are:

$$
d_{k,j} = \|\bm{q}_k - \bm{s}_{h,j}\|_2,
\qquad
\hat{\bm{u}}_{k,j} = \frac{\bm{q}_k - \bm{s}_{h,j}}{d_{k,j}} .
\tag{10}\label{eq:pf_distance_direction}
$$

The basic geometric contribution from a point source to a non-directional detector is given by the inverse-square law [@ref_Tsoulfanidis1995]:

$$
G_{k,j} = \frac{1}{4\pi d_{k,j}^2}.
\tag{11}\label{eq:pf_geometry}
$$

The detector is surrounded by lightweight iron and lead shields. At time $k$, their orientations are represented by rotation matrices:

$$
\bm{R}^{\mathrm{Fe}}_k, \quad \bm{R}^{\mathrm{Pb}}_k \in SO(3).
$$

Let the shield thicknesses be $X^{\mathrm{Fe}}$ and $X^{\mathrm{Pb}}$, and let $\mu^{\mathrm{Fe}}(E_h)$ and $\mu^{\mathrm{Pb}}(E_h)$ denote the linear attenuation coefficients of iron and lead at the representative energy $E_h$ of isotope $h$. For direction $\hat{\bm{u}}_{k,j}$, the effective path lengths through the shields are:

$$
T^{\mathrm{Fe}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Fe}}_k),\quad
T^{\mathrm{Pb}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Pb}}_k),
$$

which can be obtained by ray tracing through the shield geometry [@ref_anderson2022_tase]. The shielding attenuation factor becomes:

$$
A^{\mathrm{sh}}_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= \exp\left(
    - \mu^{\mathrm{Fe}}(E_h)\,
      T^{\mathrm{Fe}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Fe}}_k)
    - \mu^{\mathrm{Pb}}(E_h)\,
      T^{\mathrm{Pb}}(\hat{\bm{u}}_{k,j},\bm{R}^{\mathrm{Pb}}_k)
  \right).
\tag{12}\label{eq:pf_shield_attn}
$$

The combined kernel for isotope $h$ and source $j$ is defined as:

$$
K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= G_{k,j}\,
  A^{\mathrm{sh}}_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
\tag{13}\label{eq:pf_kernel}
$$

For later use, we also regard the kernel as a function of a generic source position:

$$
K_{k,h}(\bm{s},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k),
$$

so that:

$$
K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= K_{k,h}(\bm{s}_{h,j},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
$$

Note that attenuation by environmental obstacles is not considered; only geometric spreading and attenuation by the lightweight shields are modeled.

Let $q_{h,j} \ge 0$ denote the strength of the $j$-th source of isotope $h$, and let $b_h$ denote the background count rate for isotope $h$. We collect the source strengths in the vector:

$$
\bm{q}_h = (q_{h,1},\dots,q_{h,r_h})^{\top}.
$$

For a given shield orientation, the expected count rate (per unit time) for isotope $h$ at pose $k$ is:

$$
\lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= b_h
  + \sum_{j=1}^{r_h}
      K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)\,q_{h,j}.
\tag{14}\label{eq:pf_lambda_rate}
$$

For acquisition time $T_k$, the expected total count is:

$$
\Lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= T_k\,\lambda_{k,h}(\bm{q}_h,\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
\tag{15}\label{eq:pf_lambda_total}
$$

In practice, all quantities that depend only on geometry and shield orientation, such as $G_{k,j}$ and $A^{\mathrm{sh}}_{k,j,h}$, can be precomputed or cached and reused, while the PF state variables $\bm{s}_{h,j}$ and $q_{h,j}$ remain continuous.

### PF State Representation and Initialization {#chap_pf_state_init}

We construct an independent PF for each isotope $h\in\mathcal{H}$. The state vector for isotope $h$ is defined as:

$$
\bm{\theta}_h
= \left(
    r_h,\;
    \{\bm{s}_{h,m}\}_{m=1}^{r_h},\;
    \{q_{h,m}\}_{m=1}^{r_h},\;
    b_h
  \right),
\tag{16}\label{eq:pf_state}
$$

where $r_h$ is the number of sources of isotope $h$, $\bm{s}_{h,m}$ is the location of the $m$-th source, $q_{h,m}$ is its strength, and $b_h$ is the background rate for isotope $h$.

The posterior distribution $p(\bm{\theta}_h\mid \{\bm{z}_k\})$ is approximated by $N_{\mathrm{p}}$ weighted particles:

$$
\left\{
  \bm{\theta}_h^{(n)}, w_{h}^{(n)}
\right\}_{n=1}^{N_{\mathrm{p}}},
\tag{17}\label{eq:pf_particles}
$$

where $\bm{\theta}_h^{(n)}$ is the $n$-th particle and $w_{h}^{(n)}$ is its normalized weight.

For initialization, we assume a broad prior over the number of sources, their locations, and strengths. The source locations are sampled from a uniform distribution over the explored volume, while source strengths and background rates are sampled from non-negative distributions (e.g., uniform or log-normal) [@ref_Arulampalam2002]. The initial weights are uniform:

$$
w_{h,0}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
\tag{18}\label{eq:pf_init_weights}
$$

If the number of sources $r_h$ is unknown and may change during the exploration, birth/death moves can be incorporated into the PF [@ref_khadanga2020_ari; @ref_pinkam2020_irc], so that different particles maintain different values of $r_h$.

### Prediction and Log-Domain Weight Update for Poisson Observations {#chap_pf_weight_update}

At time step $k$, the robot is at pose $\bm{q}_k$ with shield orientation $(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$. For isotope $h$ and particle $n$, let $r_h^{(n)}$ be the number of sources in $\bm{\theta}_h^{(n)}$. Using the kernel in Eq. \eqref{eq:pf_kernel}, the expected total count is:

$$
\Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= T_k\left(
    b_h^{(n)} + \sum_{j=1}^{r_h^{(n)}}
      K_{k,j,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)\,
      q_{h,j}^{(n)}
  \right),
\tag{19}\label{eq:pf_expected_counts_particle}
$$

where $q_{h,j}^{(n)}$ is the strength of the $j$-th source of isotope $h$ in particle $n$ and $K_{k,j,h}(\cdot)$ is evaluated at the source position $\bm{s}^{(n)}_{h,j}$ of that particle.

The observed isotope-wise count $z_{k,h}$ is modeled as a Poisson random variable:

$$
z_{k,h} \sim \mathrm{Poisson}\left(
  \Lambda_{k,h}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
\right),
\tag{20}\label{eq:pf_poisson_model}
$$

consistent with Eq. \eqref{eq:chap3_equation4}.

The likelihood of observing $z_{k,h}$ for particle $n$ is [@ref_Tsoulfanidis1995]:

$$
p(z_{k,h} \mid \bm{\theta}_h^{(n)},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= \frac{
    \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)^{z_{k,h}}
    \exp\!\Big(-\Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)\Big)
  }{z_{k,h}!}.
\tag{21}\label{eq:pf_poisson_likelihood}
$$

To avoid numerical underflow, we perform the weight update in the logarithmic domain [@ref_kemp2023_tns]. Let $\log w_{h,k-1}^{(n)}$ be the previous log weight. The unnormalized new log weight is:

$$
\log \tilde{w}_{h,k}^{(n)}
= \log w_{h,k-1}^{(n)}
  + z_{k,h}\log \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
  - \Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k).
\tag{22}\label{eq:pf_log_weight_raw}
$$

The normalized weight $w_{h,k}^{(n)}$ is obtained by:

$$
w_{h,k}^{(n)}
= \frac{
    \exp\Big(\log \tilde{w}_{h,k}^{(n)} - \max_{m} \log \tilde{w}_{h,k}^{(m)}\Big)
  }{
    \sum_{m=1}^{N_{\mathrm{p}}}
      \exp\Big(\log \tilde{w}_{h,k}^{(m)} - \max_{m'} \log \tilde{w}_{h,k}^{(m')}\Big)
  }.
\tag{23}\label{eq:pf_log_weight_norm}
$$

Subtracting the maximum log weight improves numerical stability without changing the normalized weights.

### Resampling, Regularization, and Particle Count Adaptation {#chap_pf_resample}

When the weights become highly imbalanced, the effective number of particles decreases and the PF may degenerate. The effective sample size for isotope $h$ is defined as [@ref_Arulampalam2002]:

$$
N_{\mathrm{eff},h}
= \frac{1}{\sum_{n=1}^{N_{\mathrm{p}}} \big(w_{h,k}^{(n)}\big)^2}.
\tag{24}\label{eq:pf_neff}
$$

If $N_{\mathrm{eff},h}$ drops below a threshold $N_{\mathrm{th}}$, resampling is performed.

We adopt low-variance (systematic) resampling or similar algorithms [@ref_Arulampalam2002] to draw a new set of particles from the discrete distribution defined by $\{w_{h,k}^{(n)}\}$. After resampling, the weights are reset to the uniform value:

$$
w_{h,k}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
\tag{25}\label{eq:pf_weights_after_resample}
$$

To prevent premature convergence to local optima, we regularize the resampled particles by adding small Gaussian perturbations to the source positions and strengths [@ref_khadanga2020_ari; @ref_pinkam2020_irc]. Specifically:

$$
\bm{s}_{h,m}^{(n)} \leftarrow \bm{s}_{h,m}^{(n)} + \bm{\epsilon}_{\mathrm{pos}},\qquad
q_{h,m}^{(n)} \leftarrow q_{h,m}^{(n)} + \epsilon_{\mathrm{int}},
\tag{26}\label{eq:pf_regularization}
$$

where $\bm{\epsilon}_{\mathrm{pos}} \sim \mathcal{N}(\bm{0},\sigma_{\mathrm{pos}}^2\bm{I})$ and $\epsilon_{\mathrm{int}} \sim \mathcal{N}(0,\sigma_{\mathrm{int}}^2)$ are small zero-mean Gaussian noises.

To balance computational cost and estimation accuracy, the number of particles $N_{\mathrm{p}}$ can be adapted online [@ref_kemp2023_tns; @ref_pinkam2020_irc]. For example, when the predictive log-likelihood variance or posterior entropy is large, $N_{\mathrm{p}}$ is increased to better represent the posterior. Conversely, when the PF has clearly converged, $N_{\mathrm{p}}$ can be decreased to reduce computation time.

### Mixing of Parallel PFs and Convergence Criteria {#chap_pf_mixing}

The independent PFs for each isotope yield separate estimates of source locations and strengths. However, some inferred sources may be spurious. To obtain a consistent multi-isotope source map, we aggregate the PF outputs and remove spurious sources.

Following the “best-case measurement” test proposed in [@ref_kemp2023_tns], we proceed as follows. For isotope $h$, let the set of candidate sources obtained from the PF (e.g., using the MMSE estimate) be:

$$
\hat{\mathcal{S}}_h
= \left\{
    (\hat{\bm{s}}_{h,m}, \hat{q}_{h,m})
  \right\}_{m=1}^{\hat{r}_h}.
\tag{27}\label{eq:pf_candidate_sources}
$$

For each candidate source $(\hat{\bm{s}}_{h,m}, \hat{q}_{h,m})$, we compute its predicted contribution to the count at all measurement poses $k$:

$$
\hat{\Lambda}_{k,h,m}
= T_k\,
  K_{k,h}\big(\hat{\bm{s}}_{h,m},\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k\big)\,
  \hat{q}_{h,m}.
\tag{28}\label{eq:pf_predicted_single}
$$

Let $k^{\star}$ denote the measurement pose where the candidate source is expected to be most visible, e.g.:

$$
k^{\star}
= \argmax_k \frac{\hat{\Lambda}_{k,h,m}}{z_{k,h} + \epsilon},
\tag{29}\label{eq:pf_best_measurement_for_source}
$$

where $\epsilon$ is a small positive constant to avoid division by zero.

We then test whether the candidate source can explain a sufficient fraction of the observed count at $k^{\star}$:

$$
\frac{\hat{\Lambda}_{k^{\star},h,m}}{z_{k^{\star},h}}
\ge \tau_{\mathrm{mix}},
\tag{30}\label{eq:pf_spurious_test}
$$

where $\tau_{\mathrm{mix}}$ is a threshold (e.g., $\tau_{\mathrm{mix}}=0.9$). If the inequality is not satisfied, the candidate is considered spurious and removed from $\hat{\mathcal{S}}_h$.

Possible convergence criteria for terminating the online estimation include:

- The volume of the 95% credible region of the source locations for each isotope $h$ falls below a predefined threshold.
- The change in the estimated source strengths or locations between consecutive time steps is small, e.g.

  $$
  \left\|
    \hat{\bm{q}}_{h,k} - \hat{\bm{q}}_{h,k-1}
  \right\|
  < \tau_{\mathrm{conv}}
  $$

  for several successive $k$, where $\hat{\bm{q}}_{h,k}$ denotes a vector of estimated source strengths or positions at time $k$.
- The information-gain and Fisher-information-based criteria used in the shield-rotation strategy (Section [Shield Rotation Strategy](#chap3_pf_rotation_section)) become small for all candidate poses, indicating that additional measurements are unlikely to significantly improve the estimate.

Once the convergence criteria are satisfied, the exploration is terminated and the final estimates are reported.

For each isotope $h$ and each estimated source index $m$ we can compute the posterior variance $\mathrm{Var}(q_{h,m})$ of its strength from the particles. We define a global uncertainty measure as:

$$
U
= \sum_{h\in\mathcal{H}}\sum_{m=1}^{\hat{r}_h} \mathrm{Var}(q_{h,m}),
\tag{31}\label{eq:pf_global_uncertainty}
$$

where $\hat{r}_h$ is the current estimated number of sources of isotope $h$.

---

## Shield Rotation Strategy {#chap3_pf_rotation_section}

In this section we describe the proposed method that actively rotates the lightweight shields to obtain pseudo-directional information from the non-directional detector. The strategy consists of generating candidate shield orientations, predicting the measurement value of each orientation using the PF particles and kernels, executing short-time measurements while rotating the shields, and selecting the next robot pose.

### Generation of Candidate Shield Orientations {#chap_pf_rotation_candidates}

While the robot is stopped at pose $\bm{q}_k$, it can rotate the shields and perform several measurements with different orientations. We discretize the azimuth angles of the iron and lead shields as:

$$
\Phi^{\mathrm{Fe}} = \{\phi^{\mathrm{Fe}}_1,\dots,\phi^{\mathrm{Fe}}_{N_{\mathrm{Fe}}}\},
\qquad
\Phi^{\mathrm{Pb}} = \{\phi^{\mathrm{Pb}}_1,\dots,\phi^{\mathrm{Pb}}_{N_{\mathrm{Pb}}}\},
$$

while keeping elevation and roll fixed.

The set of candidate shield orientations is:

$$
\mathcal{R}
= \left\{
    \big(\bm{R}^{\mathrm{Fe}}_u,\bm{R}^{\mathrm{Pb}}_v\big)
    \;\middle|\;
    \phi^{\mathrm{Fe}}_u\in\Phi^{\mathrm{Fe}},\;
    \phi^{\mathrm{Pb}}_v\in\Phi^{\mathrm{Pb}}
  \right\},
\tag{32}\label{eq:pf_orientation_candidates}
$$

where $\bm{R}^{\mathrm{Fe}}_u$ and $\bm{R}^{\mathrm{Pb}}_v$ are the rotation matrices corresponding to the azimuth angles $\phi^{\mathrm{Fe}}_u$ and $\phi^{\mathrm{Pb}}_v$, respectively. In general, $|\mathcal{R}| = N_{\mathrm{Fe}}N_{\mathrm{Pb}}$, but symmetry and mechanical constraints can be used to reduce the number of effective patterns to a smaller set $N_{\mathrm{R}}$.

For each candidate orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}$, the kernels $K_{k,j,h}$ defined in Eq. \eqref{eq:pf_kernel} are used to predict expected counts and to evaluate the measurement value as described next.

### Measurement Value Prediction and Orientation Selection {#chap_pf_pose_value}

For a candidate shield orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}$ and acquisition time $T_k$, the expected total count for isotope $h$ and particle $n$ is given by:

$$
\Lambda_{k,h}^{(n)}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
= T_k\left(
    b_h^{(n)} + \sum_{j=1}^{r_h^{(n)}}
      K_{k,j,h}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\,
      q_{h,j}^{(n)}
  \right),
\tag{33}\label{eq:pf_expected_counts_particle_rotation}
$$

consistent with Eq. \eqref{eq:pf_expected_counts_particle}.

To actively select the orientation, we evaluate the “measurement value” of each candidate using the current particle set. In this study, we consider two representative criteria: expected information gain and Fisher information.

#### Expected information gain

Let $\bm{w}_h = (w_{h}^{(1)},\dots,w_{h}^{(N_{\mathrm{p}})})$ be the current weight vector for isotope $h$. Its Shannon entropy is:

$$
H(\bm{w}_h) = -\sum_{n=1}^{N_{\mathrm{p}}} w_{h}^{(n)} \log w_{h}^{(n)}.
\tag{34}\label{eq:pf_entropy}
$$

Suppose that measurement $z_{k,h}$ is hypothetically obtained under orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$. After updating the weights, the new weight vector becomes $\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$. The expected posterior entropy is:

$$
\mathbb{E}_{z_{k,h}}
\big[
  H\big(\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\big)
\big].
$$

The expected information gain (EIG) for isotope $h$ is defined as:

$$
\mathrm{IG}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
= H(\bm{w}_h)
  - \mathbb{E}_{z_{k,h}}
    \big[
      H\big(\bm{w}'_h(z_{k,h};\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\big)
    \big],
\tag{35}\label{eq:pf_ig_single}
$$

which measures the expected reduction in uncertainty for isotope $h$ [@ref_ristic2010; @ref_pinkam2020_irc; @ref_lazna2025_net].

To combine multiple isotopes, we use a weighted sum:

$$
\mathrm{IG}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
= \sum_{h\in\mathcal{H}} \alpha_h\,
  \mathrm{IG}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}}),
\qquad
\sum_{h\in\mathcal{H}} \alpha_h = 1,
\tag{36}\label{eq:pf_ig_total}
$$

where $\alpha_h$ reflects the relative importance of isotope $h$.

#### Fisher-information-based criteria

Alternatively, we can evaluate each orientation using the Fisher information matrix, which characterizes the local sensitivity of the likelihood to parameter changes [@ref_anderson2022_tase]. For isotope $h$, let $\bm{I}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$ denote the Fisher information matrix of parameters $\bm{\theta}_h$ under orientation $(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})$. For a Poisson model, the Fisher information can be written as a sum of outer products of the gradient of the expected counts [@ref_anderson2022_tase].

Two standard scalar criteria are:

$$
J_{\mathrm{A}}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
= \sum_{h\in\mathcal{H}}
   \beta_h\,
   \mathrm{Tr}\!\left(\bm{I}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})^{-1}\right)^{-1},
\tag{37}\label{eq:pf_Aopt}
$$

$$
J_{\mathrm{D}}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})
= \sum_{h\in\mathcal{H}}
   \beta_h\,
   \log\det\!\left(\bm{I}_h(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\right),
\tag{38}\label{eq:pf_Dopt}
$$

where $\beta_h$ are weighting coefficients. Maximizing $J_{\mathrm{A}}$ or $J_{\mathrm{D}}$ corresponds to A- or D-optimal design, respectively.

### Short-Time Measurements with Rotating Shields {#chap_pf_rotation_measurement}

At pose $\bm{q}_k$, the shield orientation for the next measurement is chosen by maximizing the measurement value over all candidates:

$$
(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)
= \argmax_{(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})\in\mathcal{R}}
  \mathrm{IG}(\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}}),
\tag{39}\label{eq:pf_best_orientation}
$$

or similarly by maximizing $J_{\mathrm{A}}$ or $J_{\mathrm{D}}$ defined in Eq. \eqref{eq:pf_Aopt} and Eq. \eqref{eq:pf_Dopt}.

The robot rotates the iron and lead shields to $(\bm{R}^{\mathrm{Fe}}_k,\bm{R}^{\mathrm{Pb}}_k)$ and acquires a spectrum for a short time interval $T_k$. The spectrum is processed according to Section `subsec:spectrum_isotope_counts` (Chapter 2) to obtain the isotope-wise counts $z_{k,h}$ in Eq. `eq:pf_isotope_vector`. These counts serve as the observations for updating the PF at time step $k$.

The acquisition time $T_k$ per rotation is selected considering the trade-off between signal-to-noise ratio and total exploration time.

### Stopping Shield Rotation and Active Selection of the Next Pose {#chap_pf_rotation_stop_nextpose}

At a single robot pose $\bm{q}_k$, performing multiple measurements with different shield orientations increases information but also increases total measurement time. Let $\Delta \mathrm{IG}_k^{(r)}$ denote the information gain obtained by the $r$-th rotation at pose $\bm{q}_k$:

$$
\Delta \mathrm{IG}_k^{(r)}
= \mathrm{IG}\!\left(
    (\bm{R}^{\mathrm{Fe}},\bm{R}^{\mathrm{Pb}})^{(r)}
  \right).
\tag{40}\label{eq:pf_delta_ig}
$$

When $\Delta \mathrm{IG}_k^{(r)}$ falls below a threshold $\tau_{\mathrm{IG}}$, we stop rotating at pose $\bm{q}_k$. In addition, a maximum dwell time $T_{\max}$ per pose is imposed for safety and scheduling reasons:

$$
\sum_{r} T_k^{(r)} \le T_{\max},
\tag{41}\label{eq:pf_dwell_limit}
$$

where $T_k^{(r)}$ is the acquisition time of the $r$-th rotation at pose $\bm{q}_k$.

The next robot pose is chosen actively based on the current PF state. Let $\{\bm{q}^{\mathrm{cand}}_1,\dots,\bm{q}^{\mathrm{cand}}_L\}$ be a set of candidate future poses, generated for example by sampling reachable positions while avoiding obstacles.

Using the global uncertainty measure $U$ defined in Eq. \eqref{eq:pf_global_uncertainty}, we approximate, for each candidate pose $\bm{q}^{\mathrm{cand}}_{\ell}$, the expected uncertainty after one hypothetical measurement as $\mathbb{E}[U\mid \bm{q}^{\mathrm{cand}}_{\ell}]$ using the attenuation kernels and the PF particles [@ref_ristic2010; @ref_pinkam2020_irc; @ref_lazna2025_net].

We then choose the next pose by minimizing the expected uncertainty plus a motion cost:

$$
\bm{q}_{k+1}
= \argmin_{\bm{q}^{\mathrm{cand}}_{\ell}}
    \left(
      \mathbb{E}[U\mid \bm{q}^{\mathrm{cand}}_{\ell}]
      + \lambda_{\mathrm{cost}}
        C(\bm{q}_k,\bm{q}^{\mathrm{cand}}_{\ell})
    \right),
\tag{42}\label{eq:pf_next_pose}
$$

where $C(\bm{q}_k,\bm{q}^{\mathrm{cand}}_{\ell})$ is a cost function that encodes travel distance and obstacle avoidance, and $\lambda_{\mathrm{cost}}$ is a weighting parameter balancing information gain against motion cost [@ref_lazna2025_net].

![Concept of the proposed shield–rotation strategy and active pose selection. At each robot pose, multiple shield orientations are evaluated using information–theoretic criteria, short–time measurements are acquired with the most informative orientations, and the next robot pose is chosen to maximally reduce the remaining uncertainty.](Figures/chap2/Penetrating.png){#fig:chap3_shield_strategy}

The overall online procedure, combining the measurement model of Section [Measurement Model and Shield Design](#chap3_pf_model), the isotope-wise count processing of Section `subsec:spectrum_isotope_counts` (Chapter 2), the particle-filter-based inference of Section [Particle Filter Formulation](#chap3_pf_pf), and the shield-rotation and pose-selection strategy of this section, is summarised in Fig. [Flowchart of the proposed PF method](#fig:chap3_pf_flowchart).

![Flowchart of the proposed particle–filter–based radiation source distribution estimation with rotating shields.](Figures/chap2/Penetrating.png){#fig:chap3_pf_flowchart width=80%}

---

## Convergence Criteria and Output {#chap3_pf_conclusion}

Once the shield rotation and robot motion planning described in Section [Shield Rotation Strategy](#chap3_pf_rotation_section), together with the PF inference in Section [Particle Filter Formulation](#chap3_pf_pf), have reduced the uncertainty below the desired thresholds, the exploration is terminated and the final estimates are reported.

For each isotope $h$ and source index $m$, the method outputs the estimated source location $\hat{\bm{s}}_{h,m}$, the estimated strength $\hat{q}_{h,m}$, and the associated covariance matrix $\mathrm{Cov}(\bm{s}_{h,m},q_{h,m})$. For visualization and practical use, the estimated source distribution can be rendered as a radiation heat map overlaid with the robot trajectory and obstacles, enabling operators to intuitively understand the spatial distribution of radiation in the environment.

---

## Summary {#chap3_summary}

In this chapter, an online radiation source distribution estimation method using a particle filter with rotating shields was described.  
Section [Measurement Model and Shield Design](#chap3_pf_model) presented the mathematical measurement model for non-directional count observations and the design of lightweight shielding that satisfies the payload constraints of the mobile robot.  
Section [Particle Filter Formulation for Multi–Isotope Source–Term Estimation](#chap3_pf_pf) formulated multi-isotope three-dimensional source-term estimation as Bayesian inference with parallel particle filters, including the construction of precomputed geometric and shielding kernels, state representation, prediction, log-domain weight update, resampling, regularisation, and removal of spurious sources.  
Section [Shield Rotation Strategy](#chap3_pf_rotation_section) described the shield-rotation strategy that actively selects shield orientations and subsequent robot poses based on information-gain and Fisher-information criteria, balancing measurement informativeness and motion cost.  
Section [Convergence Criteria and Output](#chap3_pf_conclusion) discussed convergence criteria and the final outputs of the algorithm, namely the estimated source locations, strengths, and uncertainties, which can be visualised as radiation maps overlaid on the environment for practical use.  
Together with the spectrum-unfolding and isotope-wise count construction procedure of Chapter 2, these components constitute a complete framework for three-dimensional, multi-isotope source-term estimation using a mobile robot with a non-directional detector and lightweight rotating shields.
