# Chapter 2 Gamma-Ray Spectrum Unfolding and Radionuclide Identification

## 2.1 Introduction

This chapter presents the gamma-ray spectrum unfolding and automated radionuclide identification method used throughout this thesis.  
Given a one-dimensional energy spectrum measured by a mobile scintillation spectrometer, the objective is to decompose the spectrum into contributions from individual radionuclides and to estimate their relative activities.  
The target application is high-dose, cluttered environments in which a small ground robot must infer the three-dimensional distribution of $\gamma$-ray sources from non-directional, energy-resolved measurements.

Following the framework of Kemp \textit{et al.}~\cite{ref_kemp2023_tns}, the measured spectrum is modelled as a linear superposition of responses from candidate radionuclides plus background, with Poisson counting statistics governing the observed bin counts.  
The detector response matrix encodes the expected contribution of each radionuclide to each energy bin, including full-energy peaks, energy resolution effects, and background components.  
On top of this model, the chapter adopts a practical, peak-based unfolding strategy similar to that of Anderson \textit{et al.}~\cite{ref_anderson2019_case,ref_anderson2022_tase}, in which individual photopeaks are detected, quantified, associated with library lines, and then mapped to radionuclide activities.

Section~\ref{chap2_radiation} reviews basic concepts of ionising radiation, radioactive decay, and representative $\gamma$-emitting radionuclides relevant to nuclear power plant accidents and related scenarios.  
Section~\ref{chap2_rad_detector} summarises typical radiation detectors, contrasts directional and non-directional instruments, and explains the choice of a compact, non-directional scintillation spectrometer mounted on a mobile robot.  
Section~\ref{chap2_spectrum_problem} formulates the gamma-ray spectrum unfolding and radionuclide-identification problem as a linear inverse problem based on a detector response matrix and Poisson statistics.  
Section~\ref{chap2_spectrum_procedure} details the practical peak-based unfolding pipeline, including preprocessing, peak detection, baseline estimation, decomposition of overlapping peaks, dead-time correction, radionuclide library matching, construction of isotope-wise count sequences for use in the particle-filter-based source-term estimation of Chapter~3, and estimation of relative activities.  
Section~\ref{chap2_summary} summarises the chapter and links these components to the source-term estimation methods developed in subsequent chapters.  
Finally, Section~\ref{chap2_summary} presents a summary of this chapter.

---

## 2.2 Overview of Radiation
\label{chap2_radiation}

### 2.2.1 Types of Radiation

Representative types of radiation include particle radiation such as $\alpha$ rays, $\beta$ rays, and neutron radiation, as well as electromagnetic radiation such as $\gamma$ rays and X rays.

Each type of radiation has different energies and characteristics, and therefore different abilities to penetrate matter, as shown in Fig.~\ref{penetrating}.  
An $\alpha$ ray is a helium nucleus and can be stopped by a single sheet of paper.  
In air, it can travel only about 3~cm.  
Therefore, even if a source emitting $\alpha$ rays exists outside the human body, it can be shielded easily.

A $\beta$ ray is a fast electron and exhibits a continuous energy spectrum, so $\beta$ rays emitted from the same nuclide have different energies.  
A $\beta$ ray with a maximum energy of about 1--2~MeV can penetrate approximately 2--4~mm of aluminum.  
For external exposure, the main concern is damage to the skin; $\beta$ rays do not penetrate deeply into the body.  
However, when radioactive materials that emit $\alpha$ or $\beta$ rays are taken into the body, internal exposure becomes an issue.

$\gamma$ rays and X rays are electromagnetic waves with high penetrating power.  
Shielding them requires tens of centimeters of lead or several meters of concrete.  
When $\gamma$ rays or X rays enter the body, some interact with tissues and lose their energy, while the rest pass through the body.  
X rays are widely used for radiographic diagnosis.  
$\gamma$ rays can affect internal organs, and in external exposure the main concern is usually $\gamma$ rays.

Neutron radiation interacts only weakly with matter and therefore has very high penetrating power.  
For neutron shielding, materials containing hydrogen atoms such as water, paraffin, or concrete, which are effective in slowing down neutrons, are used.

Because of their high penetrating power and strong adverse effects on the human body, this study focuses on $\gamma$ rays as the target of measurement.

\begin{figure}[htb]
    \centering
        \includegraphics[width=120mm]{Figures/chap2/Penetrating.png}
        \caption{Types of radiation and their penetrating power}
        \label{penetrating}
\end{figure}

### 2.2.2 Radioactive Materials

Among the above radiations, $\alpha$ rays, $\beta$ rays, and $\gamma$ rays are emitted in association with the decay of radioactive nuclides.  
When a nuclide undergoes $\alpha$ decay or $\beta$ decay, it transforms into a different nuclide.  
In $\gamma$ decay, the nuclide itself does not change, but its nuclear energy state changes.  
Such decay occurs only once for each nucleus, following probabilistic laws, and each radioactive nuclide has a characteristic decay rate $\lambda$, which is the probability of decay per unit time.  
The rate of decrease $\frac{dN}{dt}$ of the number $N$ of undecayed nuclei is proportional to $N$ and can be expressed as

\begin{align}
  -\frac{dN}{dt} &= \lambda N,
  \label{eq:decay_equation}
\end{align}

where $\lambda$ is called the decay constant.  
If the initial number of nuclei is $N_0$, solving \eqref{eq:decay_equation} yields

\begin{align}
  N(t) &= N_0 e^{-\lambda t},
\end{align}

which shows that the number of radioactive nuclei decreases exponentially with time.  
The time required for the number of radioactive nuclei to decrease to one half of its initial value is defined as the half-life $T$.  
By substituting $N = N_0/2$ into the above equation, the half-life is obtained as

\begin{align}
  T &= \frac{\ln 2}{\lambda}.
\end{align}

Through $\beta$ decay, cesium-137 becomes the metastable nuclide barium-137m, which then emits a $\gamma$ ray and transitions to stable barium-137.

In nuclear power plant accidents and in some medical accidents, various radionuclides can be released into the environment.  
Typical examples relevant to this thesis include cesium-137, cesium-134, cobalt-60, europium-154, europium-155, and radium-226.  
These nuclides have different origins and characteristics: for instance, cesium-134 and cesium-137 are fission products, whereas cobalt-60 and europium isotopes are mainly activation products in reactor structures and medical or industrial sources, and radium-226 has historical medical and natural origins.  
The types of emitted radiation, representative $\gamma$-ray energies, and half-lives of these radionuclides are listed in Table~\ref{table:half_time}~\cite{ref_ICRP107}.

Among them, cesium-137 is particularly problematic because it has a relatively long half-life of about 30~years and emits a strong 662~keV $\gamma$ ray, so cesium-137 released into the environment remains for decades and often dominates the long-term dose.  
Cobalt-60 and europium-155 have shorter half-lives (on the order of 5~years) than cesium-137, but they can still persist in the environment for several years after an accident and contribute appreciably to the radiation field, especially inside reactor buildings and around activated components.  
In realistic post-accident scenarios, these radionuclides may coexist, and appropriate countermeasures such as shielding, decontamination, and waste management depend on the specific isotopes present.  
Therefore, it is necessary to identify the radionuclides in the environment rather than relying only on total dose.

In this study, we focus on three representative $\gamma$-emitting radionuclides: cesium-137, cobalt-60, and europium-155.  
The spectrum unfolding and source-term estimation methods developed in the following chapters are designed to identify these isotopes and to estimate their spatial distributions and strengths.

\begin{table}[b]
  \caption{Representative $\gamma$-ray energies and half-lives of selected radionuclides~\cite{ref_ICRP107}}
  \label{table:half_time}
  \begin{center}
    \begin{tabular}{lccc}
      \hline
      Radionuclide  & Emitted radiation
                    & Representative $\gamma$ lines [keV]
                    & Half-life \\
      \hline \hline
      \textbf{Cesium-137}  & $\beta^-$, \textbf{$\gamma$}
                           & \textbf{662}
                           & \textbf{30.1 years} \\
      Cesium-134           & $\beta^-$, $\gamma$
                           & 605, 796
                           & 2.06 years \\
      Cobalt-60            & $\beta^-$, $\gamma$
                           & 1173, 1332
                           & 5.27 years \\
      Europium-154         & $\beta^-$, $\gamma$
                           & 723, 873, 996, 1275, 1494, 1596
                           & 8.6 years \\
      Europium-155         & $\beta^-$, $\gamma$
                           & 86.5
                           & 4.76 years \\
      Radium-226           & $\alpha$, $\gamma$
                           & 186
                           & 1600 years \\
      Iodine-131           & $\beta^-$, $\gamma$
                           & 364
                           & 8.0 days \\
      Strontium-90         & $\beta^-$
                           & -- % practically no primary $\gamma$ emission
                           & 29 years \\
      Plutonium-239        & $\alpha$, $\gamma$
                           & 129, 375, 414
                           & 24{,}000 years \\
      \hline
    \end{tabular}
  \end{center}
\end{table}

---

## 2.3 Radiation Detectors
\label{chap2_rad_detector}

Various radiation detectors are used for radiation measurement depending on the purpose of measurement and the type of radiation to be detected.  
In radiation source--term estimation, two broad classes of detectors are commonly used:  
\emph{non-directional} detectors, which only measure the number of incoming radiation quanta, and  
\emph{directional} detectors, which provide both count and incident-direction information.  
This section summarises representative detectors in each class and explains the reason for the detector choice adopted in this thesis.

### 2.3.1 Non-directional Detectors

Non-directional detectors register radiation that enters the sensitive volume without resolving its direction of arrival.  
Because they do not require heavy collimators or imaging optics, they are typically compact, robust, and capable of operating in high-dose environments.  
However, the lack of directional information means that more measurement locations and longer integration times are generally required to estimate the spatial distribution of sources.

#### 2.3.1.1 Dose-rate survey meters

A basic example of a non-directional detector is the dose-rate survey meter.  
Typical survey meters employ either a Geiger--M\"uller tube, an ionisation chamber, or a scintillation crystal coupled to a photomultiplier tube.  
They output a scalar quantity such as count rate or ambient dose equivalent rate, integrated over energy.  
Because the output does not contain spectral information, these instruments cannot distinguish different radionuclides; nevertheless, they are widely used for radiation safety management owing to their simplicity, robustness, and wide dynamic range.

Figure~\ref{fig:chap2_scintillation} illustrates the principle of a scintillation survey meter.  
When a $\gamma$ ray enters the scintillator, electrons in the crystal are excited and the crystal emits scintillation light.  
The light is converted into electrical pulses by a photomultiplier tube, and the pulse rate is proportional to the deposited energy per unit time, enabling the measurement of radiation dose.

\begin{figure}[tb]
  \centering
  \includegraphics[width=0.6\linewidth]{Figures/chap2/Penetrating.png}
  \caption{Principle of a scintillation survey meter. Incident $\gamma$ rays deposit
  energy in the scintillator, producing scintillation light that is converted into
  electrical pulses by a photomultiplier tube.}
  \label{fig:chap2_scintillation}
\end{figure}

#### 2.3.1.2 Non-directional spectrometers

For source--term estimation and radionuclide identification, it is desirable to obtain not only the total count rate but also the \emph{energy spectrum} of the detected $\gamma$ rays.  
Non-directional spectrometers fulfil this requirement by combining a scintillation crystal (e.g., NaI(Tl) or CeBr$_3$) or a semiconductor detector (e.g., HPGe, CdZnTe) with multichannel pulse-height analysis electronics.  
From the measured spectrum, the characteristic photopeaks of individual radionuclides can be identified, enabling isotopic decomposition as described in Chapter~\ref{chap:chap2}.

In this thesis, the detector mounted on the mobile robot is a compact, energy-resolving, non-directional $\gamma$-ray spectrometer based on a scintillation crystal.  
It provides count-rate measurements over a wide dose-rate range comparable to that of survey meters, while simultaneously recording short-time energy spectra that are later processed by the spectrum-unfolding pipeline of Chapter~\ref{chap:chap2}.  
An example of such a compact non-directional spectrometer is shown in Fig.~\ref{fig:chap2_spectrometer}.

\begin{figure}[tb]
  \centering
  \includegraphics[width=0.45\linewidth]{Figures/chap2/Penetrating.png}
  \caption{Example of a compact non-directional $\gamma$-ray spectrometer based on
  a CeBr$_3$ scintillation crystal. The cylindrical detector head can be mounted on
  a mobile robot while providing energy-resolved spectra.}
  \label{fig:chap2_spectrometer}
\end{figure}

### 2.3.2 Directional Detectors

Directional detectors are designed to measure not only the number of incident $\gamma$ rays but also their approximate direction of arrival.  
By exploiting collimation, scattering kinematics, or coded apertures, these systems can directly form images of the radiation field.  
Examples include gamma cameras, Compton cameras, and various collimated or coded-aperture spectrometers.  
In radiation source--distribution estimation, directional information greatly reduces the number of required measurement locations.  
However, most directional systems either require heavy shielding and collimators or have limited count-rate capability, which complicates their use on small mobile robots in high-dose environments.

#### 2.3.2.1 Gamma camera

Figure~\ref{fig:chap2_gamma} shows the principle of a gamma camera, a representative directional detector.  
A pixelated scintillation or semiconductor detector is placed inside a lead shield with a pinhole or multi-pinhole collimator.  
Only $\gamma$ rays that pass through the pinhole are detected, and an inverted map of radiation intensity is formed on the detector plane.  
Because the count in each pixel directly corresponds to the local intensity, subsequent image reconstruction is straightforward.

\begin{figure}[tb]
  \centering
  \includegraphics[width=0.6\linewidth]{Figures/chap2/Penetrating.png}
  \caption{Principle of a gamma camera. A pixelated detector placed behind a lead
  pinhole collimator forms an inverted image of the radiation intensity on the
  detector plane.}
  \label{fig:chap2_gamma}
\end{figure}

The main drawback of gamma cameras is the need for thick lead collimators to achieve sufficient angular resolution and background rejection.  
As a result, the total system mass typically reaches several tens of kilograms.  
For compact ground robots that must traverse cluttered environments with narrow passages and stairs, such payloads exceed the allowable mass, making the deployment of gamma cameras difficult in practice.

#### 2.3.2.2 Compton camera

Figure~\ref{fig:chap2_compton} illustrates the principle of a Compton camera, another directional detector.  
Unlike gamma cameras, Compton cameras do not rely on heavy collimators and can therefore be made relatively lightweight.  
A typical system consists of two detector layers: a \emph{scatterer} and an \emph{absorber}.  
An incident $\gamma$ ray first undergoes Compton scattering in the scatterer; the scattered $\gamma$ ray is then absorbed in the absorber.  
By measuring the energy deposits and interaction positions in both layers, the direction and energy of the incident $\gamma$ ray can be reconstructed.  
Each detected event constrains the source to lie on a so-called Compton cone, and by superimposing many cones the source distribution can be estimated.

\begin{figure}[tb]
  \centering
  \includegraphics[width=0.8\linewidth]{Figures/chap2/Penetrating.png}
  \caption{Principle of a Compton camera.
  (a) An incident $\gamma$ ray undergoes Compton scattering in the scatterer and is
  absorbed in the absorber, defining a Compton cone.
  (b) Superposition of many cones allows the source position to be estimated.}
  \label{fig:chap2_compton}
\end{figure}

However, Compton cameras suffer from an upper limit on the usable count rate.  
In high-dose environments, multiple $\gamma$ rays may enter the scatterer within a short time window, leading to overlapping interactions.  
When two or more Compton scattering events occur simultaneously, it becomes impossible to correctly associate the corresponding interactions in the scatterer and absorber, and the incident directions cannot be reconstructed reliably.  
For example, the Temporal Imaging Compton Camera~V3 has an upper dose-rate limit of approximately 1~mSv/h, so it cannot be operated in the high-dose environments considered in this thesis.

### 2.3.3 Selection of Detector

In this thesis, the goal is to estimate the three-dimensional distribution of $\gamma$-ray sources in a high-dose environment using a small ground robot.  
The detector must therefore satisfy the following requirements:

1. It must be mountable on a mobile robot with a limited payload capacity.  
2. It must operate reliably in high-dose environments, up to at least several Sv/h.  
3. It should provide energy-resolved spectra to enable radionuclide identification.

Directional detectors such as gamma cameras and Compton cameras provide valuable directional information but do not meet all of these requirements simultaneously: gamma cameras are too heavy to be mounted on the robot, whereas Compton cameras suffer from low upper dose-rate limits and cannot be used in the extreme high-dose regions of interest.  
On the other hand, non-directional spectrometers based on scintillation crystals are lightweight, robust against high count rates, and capable of measuring energy spectra.

For these reasons, this thesis employs a non-directional, energy-resolving scintillation spectrometer mounted on a mobile robot as the primary radiation sensor.  
A qualitative comparison of the detector types considered is summarised in Table~\ref{table:detector}.  
Because the chosen detector does not provide inherent directional information, the subsequent chapters develop methods that exploit attenuation by lightweight, actively rotated shields and environmental structures to recover pseudo-directional information and to perform three-dimensional source--term estimation.

\begin{table}[hb]
  \caption{Comparison of radiation detectors}
  \label{table:detector}
  \begin{center}
    \begin{tabular}{ccccc}
      \hline
      Type & Mountable on robot & High-dose environment
           & Energy spectrum & Directionality \\
      \hline \hline
      \textbf{Non-directional spectrometer (this work)} & ○ & ○ & ○ & Non-directional \\
      Simple survey meter (GM / ionisation)            & ○ & ○ & × & Non-directional \\
      Gamma camera                                     & × & ○ & ○ & Directional \\
      Compton camera                                   & ○ & × & ○ & Directional \\
      Coded-aperture / scanned collimated detector     & △ & △ & ○ & Directional \\
      \hline
    \end{tabular}
  \end{center}
\end{table}

---

## 2.4 Spectrum Modeling and Problem Formulation
\label{chap2_spectrum_problem}

Consider a stationary or mobile gamma-ray detector that measures an energy spectrum in an environment where one or more radionuclides may be present.  
The detector output is discretised into $K$ energy bins.  
Let $\tilde{y}_{k}$ denote the observed number of counts in the $k$-th energy bin with central energy $E_{k}$ and width $\Delta E_{k}$ ($k = 1,\dots,K$).  
The measured spectrum is written as

\begin{align}
    \tilde{\bm{y}} = (\tilde{y}_{1}, \tilde{y}_{2}, \dots, \tilde{y}_{K})^{\mathrm{T}} .
\end{align}

Assume that there exist $M \geq 1$ candidate radionuclides.  
Let $q_{j}$ denote a parameter proportional to the activity of the $j$-th radionuclide (for example, the activity at the detector position or the source strength after geometric attenuation), and define the vector

\begin{align}
    \label{eq:spectrum_q}
    \bm{q} = (q_{1}, q_{2}, \dots, q_{M})^{\mathrm{T}} .
\end{align}

Following Kemp \textit{et al.}~\cite{ref_kemp2023_tns}, the expected number of counts in each energy bin is modelled as a linear combination of contributions from each radionuclide and from background:

\begin{align}
    \label{eq:spectrum_mu}
    \bm{\mu}(\bm{q}) = \bm{R}\bm{q} + \bm{b} ,
\end{align}

where $\bm{R} \in \mathbb{R}^{K \times M}$ is the detector response matrix and

\begin{align}
    \bm{b} = (b_{1}, b_{2}, \dots, b_{K})^{\mathrm{T}}
\end{align}

is the background spectrum (including environmental and intrinsic backgrounds).

The aim of spectrum unfolding is to estimate $\bm{q}$ from $\tilde{\bm{y}}$ and to determine which radionuclides are present.  
In the following, the structure of the response matrix $\bm{R}$ and the statistical model of the spectrum are detailed.

### 2.4.1 Detector Response Matrix
\label{subsec:spectrum_response}

For the $j$-th radionuclide, assume that the nuclear data library provides $L_{j}$ discrete gamma lines.  
The $\ell$-th line has energy $E_{j\ell}$ and emission probability (branching ratio) $\beta_{j\ell}$ per decay.  
Let $\epsilon(E)$ denote the full-energy peak detection efficiency at energy $E$, and let $\sigma(E)$ denote the energy resolution (standard deviation) of the detector, which is often approximated as

\begin{align}
    \sigma(E) = a\sqrt{E} + b ,
\end{align}

where $a$ and $b$ are calibration constants~\cite{ref_Tsoulfanidis1995}.

Assuming a Gaussian full-energy peak shape, the contribution of isotope $j$ to bin $k$ is modelled as

\begin{align}
    \label{eq:spectrum_Rkj}
    R_{kj}
      = \sum_{\ell = 1}^{L_{j}} \beta_{j\ell}\,\epsilon(E_{j\ell})\,G\!\left(E_{k}; E_{j\ell}, \sigma(E_{j\ell})\right),
\end{align}

where $G(E; \mu, \sigma)$ is the normalised Gaussian function

\begin{align}
    G(E; \mu, \sigma)
      = \frac{1}{\sqrt{2\pi}\sigma}
        \exp\!\left(-\frac{(E-\mu)^{2}}{2\sigma^{2}}\right).
\end{align}

In practice, $R_{kj}$ may incorporate not only full-energy peaks but also Compton continua and scattering from the environment, either by Monte Carlo simulation or by experimental calibration~\cite{ref_kemp2023_tns}.

Collecting $R_{kj}$ for all bins and isotopes yields the response matrix

\begin{align}
    \label{eq:spectrum_R_matrix}
    \bm{R} =
    \begin{pmatrix}
        R_{11} & \dots & R_{1M} \\
        \vdots &       & \vdots \\
        R_{K1} & \dots & R_{KM}
    \end{pmatrix}.
\end{align}

Using~\eqref{eq:spectrum_mu}, the expected count in bin $k$ is then

\begin{align}
    \mu_{k}(\bm{q}) = \sum_{j=1}^{M} R_{kj} q_{j} + b_{k}.
\end{align}

### 2.4.2 Statistical Model
\label{subsec:spectrum_stat}

As gamma rays are emitted by stochastic radioactive decay processes, the number of counts in each bin is modelled as a Poisson random variable~\cite{ref_Tsoulfanidis1995}.  
Thus, for a given $\bm{q}$,

\begin{align}
    \label{eq:spectrum_poisson}
    \tilde{y}_{k} \sim \mathrm{Poisson}\!\left( \mu_{k}(\bm{q}) \right) \quad (k=1,\dots,K),
\end{align}

where $\mu_{k}(\bm{q})$ is the $k$-th element of $\bm{\mu}(\bm{q})$.

Under the Poisson assumption~\eqref{eq:spectrum_poisson}, the likelihood of observing $\tilde{y}_{k}$ counts in bin $k$ given $\bm{q}$ is

\begin{align}
    p(\tilde{y}_{k} \mid \bm{q})
      = \frac{\mu_{k}(\bm{q})^{\tilde{y}_{k}} \exp\!\left( -\mu_{k}(\bm{q}) \right)}{\tilde{y}_{k}!} .
\end{align}

Assuming independence between energy bins, the joint likelihood for the whole spectrum is

\begin{align}
    \label{eq:spectrum_likelihood}
    p(\tilde{\bm{y}} \mid \bm{q})
      &= \prod_{k=1}^{K} p(\tilde{y}_{k} \mid \bm{q}) \nonumber \\
      &= \prod_{k=1}^{K}
         \frac{\mu_{k}(\bm{q})^{\tilde{y}_{k}} \exp\!\left( -\mu_{k}(\bm{q}) \right)}
              {\tilde{y}_{k}!} .
\end{align}

In practice, however, direct optimisation of~\eqref{eq:spectrum_likelihood} over all bins and isotopes can be numerically challenging.  
Therefore, as in Kemp \textit{et al.}~\cite{ref_kemp2024_tns}, a peak-based unfolding approach is adopted, in which the spectrum is first decomposed into individual peaks and then mapped to radionuclides.

---

## 2.5 Spectrum Unfolding Procedure
\label{chap2_spectrum_procedure}

This section describes the practical steps used to unfold the measured spectrum and to identify radionuclides.  
The procedure consists of the following steps:

1. preprocessing (energy calibration and smoothing),  
2. peak detection,  
3. baseline estimation and net peak area computation,  
4. decomposition of overlapping peaks (spectral stripping),  
5. dead-time correction,  
6. radionuclide library matching and identification,  
7. construction of isotope-wise count sequences from successive spectra,  
8. estimation of relative activities (optional).

These steps closely follow the approach of Kemp \textit{et al.}~\cite{ref_kemp2024_tns} and Anderson \textit{et al.}~\cite{ref_anderson2022_tase}.

### 2.5.1 Preprocessing: Energy Calibration and Smoothing
\label{subsec:spectrum_preprocessing}

#### Energy Calibration

The raw detector output is given in channel number $c$.  
A polynomial calibration function is used to convert channel numbers to energy:

\begin{align}
    \label{eq:spectrum_calib}
    E(c) = a_{0} + a_{1}c + a_{2}c^{2},
\end{align}

where $a_{0}, a_{1}, a_{2}$ are calibration coefficients.  
These coefficients are estimated by least squares using one or more reference radionuclides with well-known peak energies.  
Let $(c_{r\ell}, E_{r\ell})$ be the channel location and known energy of the $\ell$-th reference peak.  
The calibration parameters are obtained by

\begin{align}
    \label{eq:spectrum_calib_ls}
    \hat{\bm{a}}
      = \underset{\bm{a}}{\operatorname{argmin}}
        \sum_{\ell}
        \left( E_{r\ell} - (a_{0} + a_{1}c_{r\ell} + a_{2}c_{r\ell}^{2}) \right)^{2},
\end{align}

where $\bm{a} = (a_{0}, a_{1}, a_{2})^{\mathrm{T}}$.

#### Smoothing

To suppress high-frequency statistical noise without significantly distorting peak shapes, the spectrum is smoothed by convolution with a Gaussian kernel:

\begin{align}
    \label{eq:spectrum_smoothing}
    \tilde{y}^{(\mathrm{sm})}_{k}
      = \sum_{m=-M}^{M} h_{m}\,\tilde{y}_{k-m},
\end{align}

where the kernel coefficients $h_{m}$ are given by

\begin{align}
    h_{m}
      = \frac{1}{\sum_{r=-M}^{M} \exp\!\left(-\frac{r^{2}}{2\sigma_{\mathrm{sm}}^{2}}\right)}
        \exp\!\left(-\frac{m^{2}}{2\sigma_{\mathrm{sm}}^{2}}\right),
\end{align}

and $\sigma_{\mathrm{sm}}$ controls the smoothing strength.  
Kemp \textit{et al.}~\cite{ref_kemp2023_tns} report that moderate smoothing improves peak detection robustness while preserving resolution.

### 2.5.2 Peak Detection
\label{subsec:spectrum_peaks}

Peaks are detected on the smoothed spectrum $\tilde{y}^{(\mathrm{sm})}_{k}$.  
One classical approach is the second-derivative method proposed by Mariscotti~\cite{ref_Mariscotti1967}.  
Define the discrete second difference

\begin{align}
    D^{2}\tilde{y}^{(\mathrm{sm})}_{k}
      = \tilde{y}^{(\mathrm{sm})}_{k+1}
        - 2\tilde{y}^{(\mathrm{sm})}_{k}
        + \tilde{y}^{(\mathrm{sm})}_{k-1}.
\end{align}

For a locally linear background $B(E) = \alpha + \beta E$, the second difference satisfies $D^{2}B \approx 0$; therefore, a negative value of $D^{2}\tilde{y}^{(\mathrm{sm})}_{k}$ indicates the presence of a peak.  
Using a noise estimate $\sigma_{D^{2}}$, a channel $k$ is declared a peak candidate if

\begin{align}
    -D^{2}\tilde{y}^{(\mathrm{sm})}_{k} > \lambda_{\mathrm{th}}\,\sigma_{D^{2}},
\end{align}

where $\lambda_{\mathrm{th}}$ is a user-defined signal-to-noise threshold~\cite{ref_Mariscotti1967}.

In this thesis, peak detection is implemented using a Gaussian-matched filter.  
The filter correlates the spectrum with a normalised Gaussian template of width equal to the detector resolution, and local maxima in the filter response above a certain threshold are selected as peak candidates.  
This approach has been shown by Kemp \textit{et al.}~\cite{ref_kemp2024_tns} to perform robustly in the presence of Compton continua and moderate statistical noise.

### 2.5.3 Baseline Estimation and Net Peak Area
\label{subsec:spectrum_baseline}

Gamma-ray spectra typically exhibit a significant continuous component due to Compton scattering and environmental backgrounds.  
Accurate radionuclide identification requires subtracting this baseline and computing the net area of each photopeak.

#### Baseline Estimation

The baseline is estimated using an asymmetric least-squares (ALS) smoothing method, which penalises deviations of the estimated baseline above the measured spectrum more strongly than deviations below it.  
Let $y_{k} = \tilde{y}^{(\mathrm{sm})}_{k}$.  
The baseline $\hat{b}_{k}$ is obtained by solving

\begin{align}
    \label{eq:spectrum_baseline_als}
    \hat{\bm{b}} =
    \underset{\bm{b}}{\operatorname{argmin}}\;
    \sum_{k=1}^{K} w_{k}(y_{k} - b_{k})^{2}
    + \lambda \sum_{k=2}^{K-1} (\Delta^{2} b_{k})^{2},
\end{align}

where $\Delta^{2}b_{k} = b_{k+1} - 2b_{k} + b_{k-1}$ is the discrete second difference and $\lambda$ is a smoothness parameter.  
The weights $w_{k}$ are updated iteratively according to

\begin{align}
    w_{k} =
    \begin{cases}
        p    & \text{if } y_{k} > b_{k}, \\
        1-p  & \text{otherwise},
    \end{cases}
\end{align}

with $0 < p < 1$.  
This choice forces the baseline to lie predominantly below the data, thus preserving peaks.

#### Net Peak Area

For each detected peak, a local fitting window $\mathcal{W}_{p}$ centred at energy $E_{p}$ is defined, typically covering $\pm 3\sigma_{p}$ where $\sigma_{p}$ is estimated from the detector resolution.  
Within this window, the peak shape is modelled as a Gaussian:

\begin{align}
    g_{p}(E; A_{p}, E_{p}, \sigma_{p})
      = A_{p}\exp\!\left(-\frac{(E-E_{p})^{2}}{2\sigma_{p}^{2}}\right),
\end{align}

where $A_{p}$ is the amplitude.  
The total model in the window is

\begin{align}
    \label{eq:spectrum_peak_fit}
    y_{k} \approx \hat{b}_{k} + g_{p}(E_{k}; A_{p}, E_{p}, \sigma_{p}) .
\end{align}

The parameters $(A_{p}, E_{p})$ are obtained by least-squares fitting.

The net peak area $N_{p}$ is then given by

\begin{align}
    \label{eq:spectrum_net_area_gauss}
    N_{p}
      = \sqrt{2\pi}\,A_{p}\sigma_{p},
\end{align}

or equivalently by discrete summation above the baseline,

\begin{align}
    \label{eq:spectrum_net_area_sum}
    N_{p}
      = \sum_{k \in \mathcal{W}_{p}} \left( y_{k} - \hat{b}_{k} \right).
\end{align}

Assuming Poisson statistics, the variance of $N_{p}$ can be approximated as

\begin{align}
    \label{eq:spectrum_net_area_var}
    \sigma_{N_{p}}^{2}
      \approx \sum_{k \in \mathcal{W}_{p}} y_{k}.
\end{align}

These net areas and their uncertainties are the primary inputs for radionuclide identification.

### 2.5.4 Decomposition of Overlapping Peaks
\label{subsec:spectrum_stripping}

When peaks from different radionuclides overlap due to limited energy resolution, their contributions must be separated.  
Kemp \textit{et al.}~\cite{ref_kemp2023_tns} employ a spectral stripping approach based on known intensity ratios from the nuclear data library.

For each radionuclide $j$, one \emph{reference line} (e.g.\ the most intense or least overlapped peak) with energy $E_{jr}$ is chosen.  
Let $N_{jr}$ be its net area.  
For another line $\ell$ of the same radionuclide, the expected net area is

\begin{align}
    \label{eq:spectrum_ratio}
    \hat{N}_{j\ell} = r_{j\ell} N_{jr},
\end{align}

where

\begin{align}
    r_{j\ell}
      = \frac{\beta_{j\ell}\,\epsilon(E_{j\ell})}
             {\beta_{jr}\,\epsilon(E_{jr})}
\end{align}

is the intensity ratio corrected for detector efficiency.  
Suppose that an observed peak at energy $E_{i}$ is the sum of contributions from multiple radionuclides:

\begin{align}
    N_{i}^{\mathrm{obs}} = \sum_{j} N_{ij}.
\end{align}

If a subset of radionuclides has already been quantified via their reference lines, their contributions $\hat{N}_{ij}$ can be predicted by~\eqref{eq:spectrum_ratio} and subtracted:

\begin{align}
    \label{eq:spectrum_stripping}
    N_{i}^{\mathrm{res}}
      = N_{i}^{\mathrm{obs}} - \sum_{j\in\mathcal{J}_{\mathrm{known}}} \hat{N}_{ij},
\end{align}

where $\mathcal{J}_{\mathrm{known}}$ is the set of radionuclides whose reference peaks are already assigned.  
The residual area $N_{i}^{\mathrm{res}}$ is then used to estimate the remaining radionuclides.  
For example, Kemp \textit{et al.}~\cite{ref_kemp2023_tns} strip the contribution of the \SI{609}{keV} line of the uranium/radium decay chain from the \SI{662}{keV} region before quantifying \ce{^{137}Cs}.

This stripping can be expressed as a linear system.  
Let $\bm{N}^{\mathrm{obs}}$ be the vector of observed peak areas and $\bm{\theta}$ the vector of reference-line areas for all radionuclides.  
Then

\begin{align}
    \bm{N}^{\mathrm{obs}} \approx \bm{S}\bm{\theta},
\end{align}

where $\bm{S}$ is a matrix of intensity ratios.  
The least-squares estimate of $\bm{\theta}$ is

\begin{align}
    \hat{\bm{\theta}}
      = \underset{\bm{\theta} \ge 0}{\operatorname{argmin}}
        \left\| \bm{N}^{\mathrm{obs}} - \bm{S}\bm{\theta} \right\|_{2}^{2}.
\end{align}

The component $\hat{\theta}_{j}$ corresponding to radionuclide $j$ determines all its peak areas via~\eqref{eq:spectrum_ratio}.

### 2.5.5 Dead-Time Correction
\label{subsec:spectrum_deadtime}

At high count rates, the detector and data acquisition system exhibit a dead time $\tau_{d}$ during which additional pulses are not recorded.  
For a non-paralysable system, the relationship between the true count rate $n$ and the measured count rate $m$ is~\cite{ref_Tsoulfanidis1995}

\begin{align}
    \label{eq:spectrum_deadtime}
    m = \frac{n}{1 + n\tau_{d}} .
\end{align}

Solving for $n$ yields

\begin{align}
    n = \frac{m}{1 - m\tau_{d}} .
\end{align}

Kemp \textit{et al.}~\cite{ref_kemp2023_tns} apply this correction to the total count rate using a measured dead time of $\tau_{d} = 5.813\times 10^{-9}\,\mathrm{s}$ for their detector.

Let $T$ be the live time of a measurement.  
The total measured count rate is $m_{\mathrm{tot}} = N_{\mathrm{tot}}/T$, where $N_{\mathrm{tot}}$ is the total number of recorded counts.  
The corrected total count rate is

\begin{align}
    n_{\mathrm{tot}} = \frac{m_{\mathrm{tot}}}{1 - m_{\mathrm{tot}}\tau_{d}} ,
\end{align}

and the corrected total number of counts is $N_{\mathrm{tot}}^{\mathrm{corr}} = n_{\mathrm{tot}}T$.  
Assuming that dead time affects all energy bins proportionally, each bin is scaled by the factor

\begin{align}
    \label{eq:spectrum_deadtime_factor}
    f_{\mathrm{DT}}
      = \frac{N_{\mathrm{tot}}^{\mathrm{corr}}}{N_{\mathrm{tot}}}
      = \frac{1}{1 - m_{\mathrm{tot}}\tau_{d}} .
\end{align}

Thus the dead-time-corrected bin counts are

\begin{align}
    \tilde{y}^{(\mathrm{corr})}_{k}
      = f_{\mathrm{DT}} \tilde{y}_{k},
\end{align}

and the same factor is applied to peak areas.

### 2.5.6 Radionuclide Library Matching and Identification
\label{subsec:spectrum_matching}

Having obtained a set of peak energies $\{E_{p}\}$, net areas $\{N_{p}\}$, and uncertainties $\{\sigma_{N_{p}}\}$, each peak must be associated with candidate gamma lines from the radionuclide library.

#### Energy Matching

Following Anderson \textit{et al.}~\cite{ref_anderson2022_tase}, the difference between a measured peak energy $E_{\alpha}$ and a library line energy $E_{\beta}$ is denoted by

\begin{align}
    d = \lvert E_{\alpha} - E_{\beta} \rvert .
\end{align}

Let $\sigma_{\alpha}$ and $\sigma_{\beta}$ denote the standard uncertainties of the measured and library energies, respectively.  
To account for calibration drift and energy-dependent resolution, an empirical factor $H(E)$ is introduced~\cite{ref_anderson2022_tase}:

\begin{align}
    H(E) = 1 + \frac{E}{4000},
\end{align}

where $E$ is the average energy of the pair.  
Define the variance of $d$ as

\begin{align}
    \sigma_{d}^{2} = \zeta H(E)\left(\sigma_{\alpha}^{2} + \sigma_{\beta}^{2}\right),
\end{align}

where $\zeta$ is a tuning parameter.  
Assuming $d$ is normally distributed with mean zero and variance $\sigma_{d}^{2}$, the normalised deviation is

\begin{align}
    t = \frac{d}{\sigma_{d}}.
\end{align}

Let $\Phi(\cdot)$ denote the cumulative distribution function of the standard normal distribution.  
The two-sided tail probability is then

\begin{align}
    \label{eq:spectrum_Z}
    Z(d) = 2\left[1 - \Phi(t)\right].
\end{align}

A small value of $Z(d)$ indicates a good match between the measured peak and the library line.  
If $Z(d) < Z_{\mathrm{th}}$ for a predefined threshold $Z_{\mathrm{th}}$, the pair $(\alpha,\beta)$ is considered a possible association.

#### Peak–Isotope Association and Detection Probability

For each radionuclide $j$, let $\mathcal{P}_{j}$ be the set of peaks whose energies are compatible with at least one gamma line of $j$ according to the above criterion.  
Anderson \textit{et al.}~\cite{ref_anderson2022_tase} employ a Bayesian framework based on the method of Stinnett and Sullivan to compute the probability that radionuclide $j$ is present.  
Conceptually, for each isotope $j$, a likelihood ratio is constructed between the hypotheses

\begin{align*}
    H_{0}: &\quad q_{j} = 0, \\
    H_{1}: &\quad q_{j} > 0,
\end{align*}

using the peak areas in $\mathcal{P}_{j}$ and their expected values under each hypothesis.  
The posterior probability of presence is then

\begin{align}
    P(H_{1} \mid \mathrm{data})
      = \frac{L(\mathrm{data} \mid H_{1})\,P(H_{1})}
             {L(\mathrm{data} \mid H_{0})\,P(H_{0})
              + L(\mathrm{data} \mid H_{1})\,P(H_{1})},
\end{align}

where $P(H_{0})$ and $P(H_{1})$ are prior probabilities and $L(\cdot)$ denotes the likelihood.  
A radionuclide is declared present if this posterior probability exceeds a threshold (e.g.\ $0.9$).

### 2.5.7 Isotope–wise Count Sequence from Spectra
\label{subsec:spectrum_isotope_counts}

For source--term estimation in later chapters, it is convenient to summarise each short-time spectrum as a compact vector of isotope--wise counts.  
This subsection describes how such counts are constructed from the unfolded spectra obtained by the procedures in Sections~\ref{subsec:spectrum_preprocessing}--\ref{subsec:spectrum_matching}.

We assume that an energy--resolving detector (e.g., a scintillation detector) is used and that the gamma--ray spectrum is recorded at each measurement time along the robot trajectory.  
Let $\tilde{z}_{k,c}$ denote the number of counts in channel $c\in\{1,\dots,C\}$ at time step $k$.  
After energy calibration, each channel corresponds to an energy $E_c$.

A library of candidate isotopes $\mathcal{H}$ is prepared.  
Each isotope $h\in\mathcal{H}$ has a set of characteristic photopeaks with energies $E_{h,p}$ and relative intensities (branching ratios) $r_{h,p}$, where $p=1,\dots,P_h$.  
Using the peak-finding, baseline-subtraction, and deconvolution procedures described in Sections~\ref{subsec:spectrum_preprocessing}--\ref{subsec:spectrum_baseline}, each spectrum is analysed to identify these photopeaks.  
For each peak $p$ associated with isotope $h$, an energy window $\mathcal{C}_{h,p}$ (a set of channels) is defined and the corresponding peak count at time $k$ is obtained as

\begin{align}
  \tilde{y}_{k,h,p} = \sum_{c\in \mathcal{C}_{h,p}} \tilde{z}_{k,c}.
  \label{eq:pf_peak_counts}
\end{align}

Here, $\tilde{z}_{k,c}$ can be regarded as the dead-time-uncorrected, baseline-subtracted counts in channel $c$.

Dead-time correction is applied to the recorded peak counts using the non-paralysable model described in Section~\ref{subsec:spectrum_deadtime}, and the corrected values are again denoted by $y_{k,h,p}$ for simplicity.

The contributions of multiple peaks belonging to the same isotope $h$ are then aggregated using the branching ratios as weights.  
The total count for isotope $h$ at time $k$ is defined as

\begin{align}
  z_{k,h}
  = \sum_{p=1}^{P_h} w_{h,p} y_{k,h,p},
  \qquad
  w_{h,p} = \frac{r_{h,p}}{\sum_{p'=1}^{P_h} r_{h,p'}}.
  \label{eq:pf_isotope_counts}
\end{align}

Thus, for each time step $k$, we obtain the isotope--wise count vector

\begin{align}
  \bm{z}_k = \{z_{k,h}\}_{h\in\mathcal{H}}.
  \label{eq:pf_isotope_vector}
\end{align}

These isotope--wise counts summarise the unfolded spectra in a form that is directly usable as observations for a set of parallel particle filters, one PF per isotope, in the multi--isotope source--term estimation framework of Chapter~3~\cite{ref_kemp2023_tns,ref_kemp2024_tns}.

---

## 2.6 Summary
\label{chap2_summary}

In this chapter, the gamma-ray spectrum unfolding and radionuclide identification method employed in this thesis was described.  
Section~\ref{chap2_radiation} reviewed basic properties of ionising radiation, introduced representative radioactive materials, and highlighted key $\gamma$-emitting radionuclides relevant to post-accident environments.  
Section~\ref{chap2_rad_detector} compared non-directional and directional radiation detectors and justified the use of a compact, non-directional scintillation spectrometer mounted on a mobile robot for high-dose measurements.  
Section~\ref{chap2_spectrum_problem} modelled the measured spectrum using a detector response matrix and Poisson counting statistics, formulating spectrum unfolding as a linear inverse problem over radionuclide activities.  
Section~\ref{chap2_spectrum_procedure} detailed the practical peak-based unfolding procedure, including preprocessing, peak detection, baseline and overlapping-peak treatment, dead-time correction, radionuclide library matching, construction of isotope-wise count sequences, and estimation of relative activities.  
These elements together provide the spectral analysis foundation required for the three-dimensional source-term estimation developed in the following chapters.


# Chapter 3 Online Radiation Source Distribution Estimation Using a Particle Filter with Rotating Shields

## 3.1 Introduction

In this chapter, we propose an online method for estimating the three-dimensional distribution of multiple γ-ray sources using a particle filter (PF) combined with actively rotating shields. A mobile robot equipped with an energy-resolving, non-directional detector and lightweight iron and lead shields moves in a high-dose indoor environment and acquires radiation measurements while continuously changing the shield orientations. By exploiting geometric spreading and controlled attenuation due to the shields, the method recovers pseudo-directional information from non-directional measurements and estimates the locations and strengths of multiple γ-ray sources.

The previous chapter focused on gamma-ray spectrum unfolding and radionuclide identification. In particular, it defined how each short-time spectrum is converted into an isotope-wise count vector \(\boldsymbol{z}_k\). In this chapter, these isotope-wise count sequences are used as PF observations to infer spatial source distributions for each isotope in real time. The radiation transport between sources and detector is modelled using an inverse-square law and shield-dependent attenuation kernels, and the resulting Poisson count model is embedded in a Bayesian PF framework. In addition, an active sensing strategy selects shield orientations and robot poses based on information-theoretic criteria so that measurements are collected where they are most informative for source-term estimation.

Section 3.2 introduces the physical and mathematical model of non-directional count measurements with a shielded detector and discusses the design of lightweight lead shielding under robot payload constraints.  
Section 3.3 formulates multi-isotope source-term estimation as Bayesian inference with parallel PFs, including the definition of precomputed geometric and shielding kernels, state representation, prediction, log-domain weight update for Poisson observations, resampling, regularisation, and spurious-source rejection.  
Section 3.4 presents the shield-rotation strategy, which selects informative shield orientations and next robot poses using information-gain and Fisher-information-based criteria together with short-time measurements.  
Section 3.5 summarises the convergence criteria and the final outputs of the algorithm, which provide radiation maps for subsequent visualisation and decision making.  
Finally, Section 3.6 presents a summary of this chapter.
## 3.2 Measurement Model and Shield Design

### 3.2.1 Macroscopic Attenuation and Shield Thickness

Although γ rays are electromagnetic waves, their high energy also gives them particle-like properties. A macroscopic view of the interaction between γ rays and matter considers an incident beam on a flat slab of material. When γ rays are incident on the material, some are absorbed and some undergo scattering, changing their direction and energy.

Consider a thin layer of thickness \(dX\) inside the material, through which \(N\) γ rays are passing. Let \(N'\) be the number of γ rays that pass through this layer without interaction. The number of γ rays that interact in the layer, \(-dN = N - N'\), is proportional to both the thickness \(dX\) of the layer and the number \(N\) of incident γ rays. Using the proportionality constant \(\mu\), this relationship can be written as

$$
-dN = \mu N dX,
$$

where \(\mu\) is called the *linear attenuation coefficient* and represents the ease with which interactions occur.

If \(N_0\) is the number of γ rays incident on the slab, the solution is

$$
N = N_0 e^{-\mu X}.
$$

The radiation dose measured by a detector obeys the inverse-square law with respect to the distance between a point source and the detector. Let the detector position at the \(k\)-th measurement be

$$
\boldsymbol{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^\top,
$$

and let a point source of intensity \(q_j\) be located at

$$
\boldsymbol{s}_j = [x_j, y_j, z_j]^\top.
$$

Defining the distance

$$
d_{k,j}
  = \sqrt{
      (x_k^{\mathrm{det}} - x_j)^2
      + (y_k^{\mathrm{det}} - y_j)^2
      + (z_k^{\mathrm{det}} - z_j)^2
    },
$$

the radiation intensity \(I(\boldsymbol{q}_k)\) measured at \(\boldsymbol{q}_k\) is given by

$$
I(\boldsymbol{q}_k)
  = \frac{S q_j}{4\pi d_{k,j}^2}
    e^{-\mu_{\mathrm{air}} d_{k,j}},
$$

where \(S\) is the detector area and \(\mu_{\mathrm{air}}\) is the linear attenuation coefficient of air.

For the γ rays emitted by cesium-137, which is one of the main radiation sources considered in this study, the linear attenuation coefficient in air at 20 °C is approximately \(9.7 \times 10^{-3}\,\mathrm{m}^{-1}\). In this study, measurements are performed relatively close to the radiation sources, and thus we approximate \(e^{-\mu_{\mathrm{air}} d_{k,j}} \approx 1\) and neglect attenuation in air. We also neglect attenuation by environmental obstacles, such as walls and equipment, and explicitly model only the attenuation due to the lightweight shield mounted on the robot.

The thickness of a shielding material required to reduce the dose rate by half is called the *half-value layer*, and the thickness required to reduce the dose rate to one-tenth is called the *tenth-value layer*. Table 3.1 lists the half-value and tenth-value layers of lead, iron, and concrete for γ rays emitted by cesium-137.

**Table 3.1: Shield thickness required to reduce dose rate [mm]**

| Material | Half-value layer | Tenth-value layer |
|----------|------------------|-------------------|
| Lead     | 7                | 22                |
| Iron     | 15               | 50                |
| Concrete | 49               | 163               |

### 3.2.2 Mathematical Model of Non-directional Count Measurements

We assume that \(M \ge 1\) unknown radiation point sources exist in the environment. Let the position of the \(j\)-th source be \(\boldsymbol{s}_j = [x_j, y_j, z_j]^\top\) and its strength be \(q_j \ge 0\). The radiation source distribution is represented by the vector

$$
\boldsymbol{q} = (q_1, q_2, \dots, q_M)^\top.
$$

A non-directional detector mounted on the robot acquires \(N_{\mathrm{meas}}\) measurements at positions

$$
\boldsymbol{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^\top,
\quad k = 1,\dots,N_{\mathrm{meas}}.
$$

Using the distance \(d_{k,j}\) defined above, and neglecting attenuation in air and environmental obstacles, the inverse-square law implies that the contribution of a unit-strength source at \(\boldsymbol{s}_j\) to the detector at pose \(\boldsymbol{q}_k\) is proportional to \(1/d_{k,j}^2\). We absorb the detector area, acquisition time, and conversion factors between source strength and count rate into a single constant \(\Gamma\) and define

$$
A_{k,j} = \frac{\Gamma}{d_{k,j}^2},
$$

which represents the expected count at pose \(k\) from a unit-strength source at source \(j\).

The expected total count at pose \(k\) due to the full source distribution \(\boldsymbol{q}\) is then

$$
\Lambda_k(\boldsymbol{q}) = \sum_{j=1}^M A_{k,j} q_j.
$$

Collecting all measurement poses, we define the vector of expected counts

$$
\boldsymbol{\Lambda}(\boldsymbol{q})
  = [\Lambda_1(\boldsymbol{q}), \Lambda_2(\boldsymbol{q}), \dots,
     \Lambda_{N_{\mathrm{meas}}}(\boldsymbol{q})]^\top.
$$

Let \(\mathbf{A} \in \mathbb{R}^{N_{\mathrm{meas}}\times M}\) be the matrix with elements \(A_{k,j}\). In matrix form, the measurement model can be written as

$$
\boldsymbol{\Lambda}(\boldsymbol{q}) = \mathbf{A}\boldsymbol{q}.
$$

When no shield is present, this linear model is consistent with the geometric term \(G_{k,j}\) and with the kernel \(K_{k,j,h}\) introduced later for a single isotope.

As discussed in Chapter 2, radiation is emitted by stochastic radioactive decay processes, and the number of counts recorded by the detector follows a Poisson distribution. Let \(z_k\) denote the observed count at the \(k\)-th measurement. The likelihood of observing \(z_k\) given the source distribution \(\boldsymbol{q}\) is

$$
p(z_k \mid \boldsymbol{q})
  = \frac{
      \Lambda_k(\boldsymbol{q})^{z_k}
      \exp\!\left(-\Lambda_k(\boldsymbol{q})\right)
    }{
      z_k!
    }.
$$

This single-isotope, count-only Poisson model is extended in later sections to isotope-wise counts \(z_{k,h}\) and shield-dependent kernels \(K_{k,j,h}\).

### 3.2.3 Shield Mass and Robot Payload

The densities of various materials at 20 °C are listed in Table 3.2. For a given attenuation level (for example a given number of half-value or tenth-value layers), the required mass is essentially independent of the material: denser materials require thinner shields, and less dense materials require thicker shields. Therefore, when mounting a shield on a mobile robot with a limited payload capacity, it is desirable to use the material with the highest density. In this study, we choose lead as the shielding material to be mounted on the mobile robot.

**Table 3.2: Material properties**

| Material | Density           |
|----------|-------------------|
| Lead     | 11.36 g/cm³       |
| Iron     | 7.87 g/cm³        |
| Concrete | 2.1 g/cm³         |

In this study, a lightweight shield is used to partially cover the non-directional detector. Therefore, the proposed shield design satisfies the payload constraints of the mobile robot and is feasible for deployment in real environments.

![Configuration of the detector and lightweight shields](Figures/chap3/Detector.eps)

*Figure 3.1: Configuration of the non-directional detector and lightweight shields used in this thesis. A compact non-directional γ-ray detector is placed at the center, and lightweight lead and iron shields partially surround the detector. By rotating these shields during measurement, the incident γ-ray flux from each direction is modulated, providing pseudo-directional information while keeping the total payload within the robot limits.*

Figure 3.1 illustrates the detector and shield configuration assumed in this chapter. The non-directional detector is mounted at the center of the assembly, while a partial lead shell and a partial iron shell surround the detector over an angular range \(\alpha\). During operation, these shields are actively rotated around the detector so that the attenuation factor changes with time. The particle filter exploits this controlled modulation of the count rate to recover pseudo-directional information from the non-directional detector.

---

## 3.4 Particle Filter Formulation for Multi–Isotope Source–Term Estimation

In this section we formulate the estimation of multiple three-dimensional point sources as Bayesian inference with parallel PFs, one PF for each isotope. The PFs use the isotope–wise counts \(\boldsymbol{z}_k\) and the precomputed geometric and shielding kernels to infer the number, locations, and strengths of sources and the background rate.

### 3.4.1 Precomputed Geometric and Shielding Kernels

To efficiently evaluate expected counts for different robot poses and shield orientations, we precompute attenuation kernels that encode geometric spreading and shielding attenuation for point sources. In contrast to grid–based methods, the PF in this thesis represents the radiation field directly as a finite set of point sources whose positions are state variables; no voxelisation of the environment is required.

For isotope \(h\), let the \(j\)-th source position be

$$
\boldsymbol{s}_{h,j} = [x_{h,j}, y_{h,j}, z_{h,j}]^\top,
\qquad j = 1,\dots,r_h,
$$

where \(r_h\) is the (unknown) number of sources of isotope \(h\). The robot measurement poses (detector positions) are denoted by

$$
\boldsymbol{q}_k = [x_k^{\mathrm{det}}, y_k^{\mathrm{det}}, z_k^{\mathrm{det}}]^\top,
\qquad k = 1,\dots,N_{\mathrm{meas}}.
$$

The distance and direction from source \(j\) to pose \(k\) are

$$
d_{k,j} = \|\boldsymbol{q}_k - \boldsymbol{s}_{h,j}\|_2,
\qquad
\hat{\boldsymbol{u}}_{k,j} = \frac{\boldsymbol{q}_k - \boldsymbol{s}_{h,j}}{d_{k,j}}.
$$

The basic geometric contribution from a point source to a non–directional detector is given by the inverse–square law,

$$
G_{k,j} = \frac{1}{4\pi d_{k,j}^2}.
$$

In the absence of shielding, the linear model above reduces to this geometric term.

The detector is surrounded by lightweight iron and lead shields. At time \(k\), their orientations are represented by rotation matrices

$$
\mathbf{R}^{\mathrm{Fe}}_k,\; \mathbf{R}^{\mathrm{Pb}}_k \in SO(3).
$$

Let the shield thicknesses be \(X^{\mathrm{Fe}}\) and \(X^{\mathrm{Pb}}\), and let \(\mu^{\mathrm{Fe}}(E_h)\) and \(\mu^{\mathrm{Pb}}(E_h)\) denote the linear attenuation coefficients of iron and lead at the representative energy \(E_h\) of isotope \(h\). For direction \(\hat{\boldsymbol{u}}_{k,j}\), the effective path lengths through the shields are

$$
T^{\mathrm{Fe}}(\hat{\boldsymbol{u}}_{k,j}, \mathbf{R}^{\mathrm{Fe}}_k),
\qquad
T^{\mathrm{Pb}}(\hat{\boldsymbol{u}}_{k,j}, \mathbf{R}^{\mathrm{Pb}}_k),
$$

which can be obtained by ray tracing through the shield geometry. The shielding attenuation factor becomes

$$
A^{\mathrm{sh}}_{k,j,h}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
  = \exp\left(
      - \mu^{\mathrm{Fe}}(E_h)
          T^{\mathrm{Fe}}(\hat{\boldsymbol{u}}_{k,j}, \mathbf{R}^{\mathrm{Fe}}_k)
      - \mu^{\mathrm{Pb}}(E_h)
          T^{\mathrm{Pb}}(\hat{\boldsymbol{u}}_{k,j}, \mathbf{R}^{\mathrm{Pb}}_k)
    \right).
$$

The combined kernel for isotope \(h\) and source \(j\) is defined as

$$
K_{k,j,h}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
  = G_{k,j}\,
    A^{\mathrm{sh}}_{k,j,h}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k).
$$

For later use, we also regard the kernel as a function of a generic source position

$$
K_{k,h}(\boldsymbol{s}, \mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k),
$$

so that \(K_{k,j,h}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k) = K_{k,h}(\boldsymbol{s}_{h,j}, \mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)\). Note that attenuation by environmental obstacles is not considered; only geometric spreading and attenuation by the lightweight shields are modelled.

Let \(q_{h,j} \ge 0\) denote the strength of the \(j\)-th source of isotope \(h\), and let \(b_h\) denote the background count rate for isotope \(h\). We collect the source strengths in the vector

$$
\boldsymbol{q}_h = (q_{h,1}, \dots, q_{h,r_h})^\top.
$$

For a given shield orientation, the expected count rate (per unit time) for isotope \(h\) at pose \(k\) is

$$
\lambda_{k,h}(\boldsymbol{q}_h, \mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
  = b_h
    + \sum_{j=1}^{r_h}
        K_{k,j,h}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)\,
        q_{h,j}.
$$

For acquisition time \(T_k\), the expected total count is

$$
\Lambda_{k,h}(\boldsymbol{q}_h, \mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
  = T_k\,\lambda_{k,h}(\boldsymbol{q}_h, \mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k).
$$

In practice, all quantities that depend only on geometry and shield orientation, such as \(G_{k,j}\) and \(A^{\mathrm{sh}}_{k,j,h}\), can be precomputed or cached and reused, while the PF state variables \(\boldsymbol{s}_{h,j}\) and \(q_{h,j}\) remain continuous.

### 3.4.2 PF State Representation and Initialization

We construct an independent PF for each isotope \(h \in \mathcal{H}\). The state vector for isotope \(h\) is defined as

$$
\boldsymbol{\theta}_h
  = \left(
      r_h,\;
      \{\boldsymbol{s}_{h,m}\}_{m=1}^{r_h},\;
      \{q_{h,m}\}_{m=1}^{r_h},\;
      b_h
    \right),
$$

where \(r_h\) is the number of sources of isotope \(h\), \(\boldsymbol{s}_{h,m}\) is the location of the \(m\)-th source, \(q_{h,m}\) is its strength, and \(b_h\) is the background rate for isotope \(h\).

The posterior distribution \(p(\boldsymbol{\theta}_h \mid \{\boldsymbol{z}_k\})\) is approximated by \(N_{\mathrm{p}}\) weighted particles

$$
\left\{
  \boldsymbol{\theta}_h^{(n)},\; w_h^{(n)}
\right\}_{n=1}^{N_{\mathrm{p}}},
$$

where \(\boldsymbol{\theta}_h^{(n)}\) is the \(n\)-th particle and \(w_h^{(n)}\) is its normalised weight.

For initialization, we assume a broad prior over the number of sources, their locations, and strengths. The source locations are sampled from a uniform distribution over the explored volume, while source strengths and background rates are sampled from non–negative distributions (e.g. uniform or log–normal). The initial weights are uniform,

$$
w_{h,0}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
$$

If the number of sources \(r_h\) is unknown and may change during the exploration, birth/death moves can be incorporated into the PF so that different particles maintain different values of \(r_h\).

### 3.4.3 Prediction and Log–Domain Weight Update for Poisson Observations

At time step \(k\), the robot is at pose \(\boldsymbol{q}_k\) with shield orientation \((\mathbf{R}^{\mathrm{Fe}}_k,\mathbf{R}^{\mathrm{Pb}}_k)\). For isotope \(h\) and particle \(n\), let \(r_h^{(n)}\) be the number of sources in \(\boldsymbol{\theta}_h^{(n)}\). Using the kernel defined above, the expected total count is

$$
\Lambda_{k,h}^{(n)}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
  = T_k
    \left(
      b_h^{(n)}
      + \sum_{j=1}^{r_h^{(n)}}
          K_{k,j,h}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)\,
          q_{h,j}^{(n)}
    \right),
$$

where \(q_{h,j}^{(n)}\) is the strength of the \(j\)-th source of isotope \(h\) in particle \(n\). Here \(K_{k,j,h}(\cdot)\) is evaluated at the source position \(\boldsymbol{s}_{h,j}^{(n)}\) of that particle.

The observed isotope–wise count \(z_{k,h}\) is modelled as a Poisson random variable,

$$
z_{k,h}
  \sim \mathrm{Poisson}\!\left(
    \Lambda_{k,h}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
  \right),
$$

consistent with the likelihood defined earlier.

The likelihood of observing \(z_{k,h}\) for particle \(n\) is

$$
p(z_{k,h} \mid \boldsymbol{\theta}_h^{(n)}, \mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
  = \frac{
      \Lambda_{k,h}^{(n)}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)^{z_{k,h}}
      \exp\!\left(-\Lambda_{k,h}^{(n)}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)\right)
    }{
      z_{k,h}!
    }.
$$

To avoid numerical underflow, we perform the weight update in the logarithmic domain. Let \(\log w_{h,k-1}^{(n)}\) be the previous log weight. The unnormalised new log weight is

$$
\log \tilde{w}_{h,k}^{(n)}
  = \log w_{h,k-1}^{(n)}
    + z_{k,h}\log \Lambda_{k,h}^{(n)}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)
    - \Lambda_{k,h}^{(n)}(\mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k).
$$

The normalised weight \(w_{h,k}^{(n)}\) is obtained by

$$
w_{h,k}^{(n)}
  = \frac{
      \exp\left(\log \tilde{w}_{h,k}^{(n)} - \max_m \log \tilde{w}_{h,k}^{(m)}\right)
    }{
      \sum_{m=1}^{N_{\mathrm{p}}}
        \exp\left(\log \tilde{w}_{h,k}^{(m)} - \max_{m'} \log \tilde{w}_{h,k}^{(m')}\right)
    }.
$$

Subtracting the maximum log weight improves numerical stability without changing the normalised weights.

### 3.4.4 Resampling, Regularization, and Particle Count Adaptation

When the weights become highly imbalanced, the effective number of particles decreases and the PF may degenerate. The effective sample size for isotope \(h\) is defined as

$$
N_{\mathrm{eff},h}
  = \frac{1}{\sum_{n=1}^{N_{\mathrm{p}}} (w_{h,k}^{(n)})^2}.
$$

If \(N_{\mathrm{eff},h}\) drops below a threshold \(N_{\mathrm{th}}\), resampling is performed.

We adopt low–variance (systematic) resampling or similar algorithms to draw a new set of particles from the discrete distribution defined by \(\{w_{h,k}^{(n)}\}\). After resampling, the weights are reset to the uniform value

$$
w_{h,k}^{(n)} = \frac{1}{N_{\mathrm{p}}}.
$$

To prevent premature convergence to local optima, we regularise the resampled particles by adding small Gaussian perturbations to the source positions and strengths:

$$
\boldsymbol{s}_{h,m}^{(n)} \leftarrow \boldsymbol{s}_{h,m}^{(n)} + \boldsymbol{\epsilon}_{\mathrm{pos}},
\qquad
q_{h,m}^{(n)} \leftarrow q_{h,m}^{(n)} + \epsilon_{\mathrm{int}},
$$

where \(\boldsymbol{\epsilon}_{\mathrm{pos}} \sim \mathcal{N}(\boldsymbol{0}, \sigma_{\mathrm{pos}}^2 \mathbf{I})\) and \(\epsilon_{\mathrm{int}} \sim \mathcal{N}(0, \sigma_{\mathrm{int}}^2)\) are small zero–mean Gaussian noises.

To balance computational cost and estimation accuracy, the number of particles \(N_{\mathrm{p}}\) can be adapted online. For example, when the predictive log–likelihood variance or posterior entropy is large, \(N_{\mathrm{p}}\) is increased to better represent the posterior. Conversely, when the PF has clearly converged, \(N_{\mathrm{p}}\) can be decreased to reduce computation time.

### 3.4.5 Mixing of Parallel PFs and Convergence Criteria

The independent PFs for each isotope yield separate estimates of source locations and strengths. However, some inferred sources may be spurious. To obtain a consistent multi–isotope source map, we aggregate the PF outputs and remove spurious sources.

Following the “best–case measurement” test proposed in previous work, we proceed as follows. For isotope \(h\), let the set of candidate sources obtained from the PF (for example using the MMSE estimate) be

$$
\hat{\mathcal{S}}_h
  = \left\{
      (\hat{\boldsymbol{s}}_{h,m}, \hat{q}_{h,m})
    \right\}_{m=1}^{\hat{r}_h},
$$

where \(\hat{r}_h\) is the estimated number of sources of isotope \(h\).

For each candidate source \((\hat{\boldsymbol{s}}_{h,m}, \hat{q}_{h,m})\), we compute its predicted contribution to the count at all measurement poses \(k\),

$$
\hat{\Lambda}_{k,h,m}
  = T_k\,
    K_{k,h}(\hat{\boldsymbol{s}}_{h,m},
            \mathbf{R}^{\mathrm{Fe}}_k, \mathbf{R}^{\mathrm{Pb}}_k)\,
    \hat{q}_{h,m}.
$$

Let \(k^\star\) denote the measurement pose where the candidate source is expected to be most visible, e.g.,

$$
k^\star
  = \operatorname*{arg\,max}_k
      \frac{\hat{\Lambda}_{k,h,m}}{z_{k,h} + \epsilon},
$$

where \(\epsilon\) is a small positive constant to avoid division by zero.

We then test whether the candidate source can explain a sufficient fraction of the observed count at \(k^\star\),

$$
\frac{\hat{\Lambda}_{k^\star,h,m}}{z_{k^\star,h}}
  \ge \tau_{\mathrm{mix}},
$$

where \(\tau_{\mathrm{mix}}\) is a threshold (e.g. \(\tau_{\mathrm{mix}} = 0.9\)). If the inequality is not satisfied, the candidate is considered spurious and removed from \(\hat{\mathcal{S}}_h\).

Possible convergence criteria for terminating the online estimation include:

- The volume of the 95% credible region of the source locations for each isotope \(h\) falls below a predefined threshold.  
- The change in the estimated source strengths or locations between consecutive time steps is small, e.g.
  \[
  \|\hat{\boldsymbol{q}}_{h,k} - \hat{\boldsymbol{q}}_{h,k-1}\|
    < \tau_{\mathrm{conv}}
  \]
  for several successive \(k\), where \(\hat{\boldsymbol{q}}_{h,k}\) denotes a vector of estimated source strengths or positions at time \(k\).  
- The information–gain and Fisher–information–based criteria used in the shield–rotation strategy (Section 3.5) become small for all candidate poses, indicating that additional measurements are unlikely to significantly improve the estimate.

Once the convergence criteria are satisfied, the exploration is terminated and the final estimates are reported. For each isotope \(h\) and each estimated source index \(m\) we can compute the posterior variance \(\mathrm{Var}(q_{h,m})\) of its strength from the particles. We define a global uncertainty measure as

$$
U = \sum_{h\in\mathcal{H}}\sum_{m=1}^{\hat{r}_h} \mathrm{Var}(q_{h,m}),
$$

where \(\hat{r}_h\) is the current estimated number of sources of isotope \(h\).

---

## 3.4 Shield Rotation Strategy

In this section we describe the proposed method that actively rotates the lightweight shields to obtain pseudo-directional information from the non–directional detector. The strategy consists of generating candidate shield orientations, predicting the measurement value of each orientation using the PF particles and kernels, executing short–time measurements while rotating the shields, and selecting the next robot pose.

### 3.5.1 Generation of Candidate Shield Orientations

While the robot is stopped at pose \(\boldsymbol{q}_k\), it can rotate the shields and perform several measurements with different orientations. We discretise the azimuth angles of the iron and lead shields as

$$
\Phi^{\mathrm{Fe}} = \{\phi^{\mathrm{Fe}}_1,\dots,\phi^{\mathrm{Fe}}_{N_{\mathrm{Fe}}}\},
\qquad
\Phi^{\mathrm{Pb}} = \{\phi^{\mathrm{Pb}}_1,\dots,\phi^{\mathrm{Pb}}_{N_{\mathrm{Pb}}}\},
$$

while keeping elevation and roll fixed.

The set of candidate shield orientations is

$$
\mathcal{R}
  = \left\{
      (\mathbf{R}^{\mathrm{Fe}}_u, \mathbf{R}^{\mathrm{Pb}}_v)
      \,\middle|\,
      \phi^{\mathrm{Fe}}_u \in \Phi^{\mathrm{Fe}},\;
      \phi^{\mathrm{Pb}}_v \in \Phi^{\mathrm{Pb}}
    \right\},
$$

where \(\mathbf{R}^{\mathrm{Fe}}_u\) and \(\mathbf{R}^{\mathrm{Pb}}_v\) are the rotation matrices corresponding to the azimuth angles \(\phi^{\mathrm{Fe}}_u\) and \(\phi^{\mathrm{Pb}}_v\), respectively. In general, \(|\mathcal{R}| = N_{\mathrm{Fe}}N_{\mathrm{Pb}}\), but symmetry and mechanical constraints can be used to reduce the number of effective patterns to a smaller set \(N_{\mathrm{R}}\).

For each candidate orientation \((\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\in\mathcal{R}\), the kernels \(K_{k,j,h}\) defined in Section 3.4.1 are used to predict expected counts and to evaluate the measurement value as described next.

### 3.5.2 Measurement Value Prediction and Orientation Selection

For a candidate shield orientation \((\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\in\mathcal{R}\) and acquisition time \(T_k\), the expected total count for isotope \(h\) and particle \(n\) is given by

$$
\Lambda_{k,h}^{(n)}(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})
  = T_k
    \left(
      b_h^{(n)}
      + \sum_{j=1}^{r_h^{(n)}}
          K_{k,j,h}(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\,
          q_{h,j}^{(n)}
    \right),
$$

consistent with the expression used in the PF update.

To actively select the orientation, we evaluate the “measurement value” of each candidate using the current particle set. In this study, we consider two representative criteria: expected information gain and Fisher information.

#### Expected information gain

Let \(\boldsymbol{w}_h = (w_h^{(1)},\dots,w_h^{(N_{\mathrm{p}})})\) be the current weight vector for isotope \(h\). Its Shannon entropy is

$$
H(\boldsymbol{w}_h)
  = -\sum_{n=1}^{N_{\mathrm{p}}} w_h^{(n)} \log w_h^{(n)}.
$$

Suppose that measurement \(z_{k,h}\) is hypothetically obtained under orientation \((\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\). After updating the weights, the new weight vector becomes \(\boldsymbol{w}'_h(z_{k,h};\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\). The expected posterior entropy is

$$
\mathbb{E}_{z_{k,h}}
  \left[
    H\left(\boldsymbol{w}'_h(z_{k,h};\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\right)
  \right].
$$

The expected information gain (EIG) for isotope \(h\) is defined as

$$
\mathrm{IG}_h(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})
  = H(\boldsymbol{w}_h)
    - \mathbb{E}_{z_{k,h}}
      \left[
        H\left(\boldsymbol{w}'_h(z_{k,h};\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\right)
      \right],
$$

which measures the expected reduction in uncertainty for isotope \(h\).

To combine multiple isotopes, we use a weighted sum

$$
\mathrm{IG}(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})
  = \sum_{h\in\mathcal{H}} \alpha_h\,
    \mathrm{IG}_h(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}}),
\qquad
\sum_{h\in\mathcal{H}} \alpha_h = 1,
$$

where \(\alpha_h\) reflects the relative importance of isotope \(h\).

#### Fisher-information-based criteria

Alternatively, we can evaluate each orientation using the Fisher information matrix, which characterises the local sensitivity of the likelihood to parameter changes. For isotope \(h\), let \(\mathbf{I}_h(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\) denote the Fisher information matrix of parameters \(\boldsymbol{\theta}_h\) under orientation \((\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\). For a Poisson model, the Fisher information can be written as a sum of outer products of the gradient of the expected counts.

Two standard scalar criteria are

$$
J_{\mathrm{A}}(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})
  = \sum_{h\in\mathcal{H}}
      \beta_h
      \left[
        \mathrm{Tr}\!\left(
          \mathbf{I}_h(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})^{-1}
        \right)
      \right]^{-1},
$$

$$
J_{\mathrm{D}}(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})
  = \sum_{h\in\mathcal{H}}
      \beta_h
      \log\det\!\left(
        \mathbf{I}_h(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})
      \right),
$$

where \(\beta_h\) are weighting coefficients. Maximising \(J_{\mathrm{A}}\) or \(J_{\mathrm{D}}\) corresponds to A– or D–optimal design, respectively.

### 3.5.3 Short–Time Measurements with Rotating Shields

At pose \(\boldsymbol{q}_k\), the shield orientation for the next measurement is chosen by maximising the measurement value over all candidates, for example

$$
(\mathbf{R}^{\mathrm{Fe}}_k,\mathbf{R}^{\mathrm{Pb}}_k)
  = \operatorname*{arg\,max}_{(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})\in\mathcal{R}}
      \mathrm{IG}(\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}}),
$$

or similarly by maximising \(J_{\mathrm{A}}\) or \(J_{\mathrm{D}}\).

The robot rotates the iron and lead shields to \((\mathbf{R}^{\mathrm{Fe}}_k,\mathbf{R}^{\mathrm{Pb}}_k)\) and acquires a spectrum for a short time interval \(T_k\). The spectrum is processed according to Section 3.3 to obtain the isotope–wise counts \(z_{k,h}\). These counts serve as the observations for updating the PF at time step \(k\).

The acquisition time \(T_k\) per rotation is selected considering the trade–off between signal–to–noise ratio and total exploration time.

### 3.5.4 Stopping Shield Rotation and Active Selection of the Next Pose

At a single robot pose \(\boldsymbol{q}_k\), performing multiple measurements with different shield orientations increases information but also increases total measurement time. Let \(\Delta \mathrm{IG}_k^{(r)}\) denote the information gain obtained by the \(r\)-th rotation at pose \(\boldsymbol{q}_k\),

$$
\Delta \mathrm{IG}_k^{(r)}
  = \mathrm{IG}\left((\mathbf{R}^{\mathrm{Fe}},\mathbf{R}^{\mathrm{Pb}})^{(r)}\right).
$$

When \(\Delta \mathrm{IG}_k^{(r)}\) falls below a threshold \(\tau_{\mathrm{IG}}\), we stop rotating at pose \(\boldsymbol{q}_k\). In addition, a maximum dwell time \(T_{\max}\) per pose is imposed for safety and scheduling reasons,

$$
\sum_r T_k^{(r)} \le T_{\max},
$$

where \(T_k^{(r)}\) is the acquisition time of the \(r\)-th rotation at pose \(\boldsymbol{q}_k\).

The next robot pose is chosen actively based on the current PF state. Let \(\{\boldsymbol{q}^{\mathrm{cand}}_1,\dots,\boldsymbol{q}^{\mathrm{cand}}_L\}\) be a set of candidate future poses, generated for example by sampling reachable positions while avoiding obstacles.

Using the global uncertainty measure \(U\) defined earlier, we approximate, for each candidate pose \(\boldsymbol{q}^{\mathrm{cand}}_\ell\), the expected uncertainty after one hypothetical measurement as \(\mathbb{E}[U\mid \boldsymbol{q}^{\mathrm{cand}}_\ell]\) using the attenuation kernels and the PF particles.

We then choose the next pose by minimising the expected uncertainty plus a motion cost,

$$
\boldsymbol{q}_{k+1}
  = \operatorname*{arg\,min}_{\boldsymbol{q}^{\mathrm{cand}}_\ell}
      \left(
        \mathbb{E}[U\mid \boldsymbol{q}^{\mathrm{cand}}_\ell]
        + \lambda_{\mathrm{cost}}
          C(\boldsymbol{q}_k,\boldsymbol{q}^{\mathrm{cand}}_\ell)
      \right),
$$

where \(C(\boldsymbol{q}_k,\boldsymbol{q}^{\mathrm{cand}}_\ell)\) is a cost function that encodes travel distance and obstacle avoidance, and \(\lambda_{\mathrm{cost}}\) is a weighting parameter balancing information gain against motion cost.

![Concept of shield–rotation strategy and active pose selection](Figures/chap2/Penetrating.png)

*Figure 3.2: Concept of the proposed shield–rotation strategy and active pose selection. At each robot pose, multiple shield orientations are evaluated using information–theoretic criteria, short–time measurements are acquired with the most informative orientations, and the next robot pose is chosen to maximally reduce the remaining uncertainty.*

The overall online procedure, combining the measurement model of Section 3.2, the isotope-wise count processing of Section 3.3, the particle-filter-based inference of Section 3.4, and the shield-rotation and pose-selection strategy of this section, is summarised in Fig. 3.3.

![Flowchart of the PF-based source distribution estimation with rotating shields](Figures/chap2/Penetrating.png)

*Figure 3.3: Flowchart of the proposed particle–filter–based radiation source distribution estimation with rotating shields.*

---

## 3.5 Convergence Criteria and Output

Once the shield rotation and robot motion planning described in Section 3.5, together with the PF inference in Section 3.4, have reduced the uncertainty below the desired thresholds, the exploration is terminated and the final estimates are reported.

For each isotope \(h\) and source index \(m\), the method outputs the estimated source location \(\hat{\boldsymbol{s}}_{h,m}\), the estimated strength \(\hat{q}_{h,m}\), and the associated covariance matrix \(\mathrm{Cov}(\boldsymbol{s}_{h,m}, q_{h,m})\). For visualisation and practical use, the estimated source distribution can be rendered as a radiation heat map overlaid with the robot trajectory and obstacles, enabling operators to intuitively understand the spatial distribution of radiation in the environment.

---

## 3.6 Summary

In this chapter, an online radiation source distribution estimation method using a particle filter with rotating shields was described.

- Section 3.2 presented the mathematical measurement model for non-directional count observations and the design of lightweight shielding that satisfies the payload constraints of the mobile robot.  
- Section 3.3 explained how energy-resolved gamma-ray spectra are processed to obtain isotope-wise count sequences that serve as observations for the particle filters.  
- Section 3.4 formulated multi-isotope three-dimensional source-term estimation as Bayesian inference with parallel particle filters, including the construction of precomputed geometric and shielding kernels, state representation, prediction, log-domain weight update, resampling, regularisation, and removal of spurious sources.  
- Section 3.5 described the shield-rotation strategy that actively selects shield orientations and subsequent robot poses based on information-gain and Fisher-information criteria, balancing measurement informativeness and motion cost.  
- Section 3.6 discussed convergence criteria and the final outputs of the algorithm, namely the estimated source locations, strengths, and uncertainties, which can be visualised as radiation maps overlaid on the environment for practical use.