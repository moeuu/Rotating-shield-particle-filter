\chapter{Gamma-Ray Spectrum Unfolding and Radionuclide Identification}
\thispagestyle{empty}
\label{chap:chap2}
\graphicspath{{Figures/chap2/}}
\lhead[Chapter 2]{}
\minitoc

\newpage
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Introduction}
\label{chap2_intro}
\hspace{9.5pt}

This chapter describes the gamma-ray spectrum unfolding and automated radionuclide identification method used throughout this thesis.
A mobile robot equipped with a compact, energy-resolving scintillation spectrometer measures one-dimensional pulse-height spectra in high-dose, cluttered environments (e.g., reactor interiors).
The objective is to decompose each measured spectrum into contributions from individual radionuclides and to estimate isotope-wise count indicators that can be used as observations for the three-dimensional STE framework developed in Chapter~3.

Section~\ref{chap2_radiation} reviews fundamental concepts of ionising radiation and radioactive decay, and summarises representative radionuclides and their $\gamma$-ray lines and half-lives relevant to post-accident environments. \par
Section~\ref{chap2_rad_detector} introduces representative non-directional and directional radiation detectors, and justifies the choice of a compact non-directional spectrometer for high-dose robotic measurements. \par
Section~\ref{chap2_spectrum_problem} formulates spectrum unfolding as a linear inverse problem using a detector response matrix and a Poisson statistical model. \par
Section~\ref{chap2_spectrum_procedure} details the unfolding procedure, including calibration and smoothing, peak detection, baseline estimation and net peak-area computation, decomposition of overlapping peaks, dead-time correction, radionuclide library matching, and construction of isotope-wise count vectors for use in Chapter~3. \par
Finally, Section~\ref{chap2_summary} summarises this chapter.


\clearpage
\newpage


% ================================================== %
% section
% ================================================== %
\section{Overview of Radiation}
\label{chap2_radiation}
\hspace{9.5pt}

\subsection{Types of Radiation}

Representative types of radiation include particle radiation such as $\alpha$ rays, $\beta$ rays, and neutron radiation, as well as electromagnetic radiation such as $\gamma$ rays and X rays.

Each type of radiation has different energies and characteristics, and therefore different abilities to penetrate matter, as shown in Fig.~\ref{fig:chap2_penetration}.
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


\clearpage
\newpage

\subsection{Radioactive Materials}
Among the above radiations, $\alpha$ rays, $\beta$ rays, and $\gamma$ rays are emitted in
association with the decay of radioactive nuclides.
When a nuclide undergoes $\alpha$ decay or $\beta$ decay, it transforms into a different
nuclide.
In $\gamma$ decay, the nuclide itself does not change, but its nuclear energy state changes.
Such decay occurs only once for each nucleus, following probabilistic laws, and each
radioactive nuclide has a characteristic decay rate $\lambda$, which is the probability of
decay per unit time.
The rate of decrease $\frac{dN}{dt}$ of the number $N$ of undecayed nuclei is
proportional to $N$ and can be expressed as
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
The time required for the number of radioactive nuclei to decrease to one half of its
initial value is defined as the half-life $T$.
By substituting $N = N_0/2$ into the above equation, the half-life is obtained as
\begin{align}
  T &= \frac{\ln 2}{\lambda}.
\end{align}

As illustrated in Fig.~\ref{fig:chap2_Cs}, \ce{^{137}Cs} undergoes $\beta^-$ decay to the metastable nuclide \ce{^{137m}Ba}, which subsequently emits a $\gamma$ ray and transitions to stable \ce{^{137}Ba}.

In nuclear power plant accidents and in some medical accidents, various radionuclides
can be released into the environment.
Typical examples relevant to this thesis include \ce{^{137}Cs}, \ce{^{134}Cs}, \ce{^{60}Co},
\ce{^{154}Eu}, \ce{^{155}Eu}, \ce{^{226}Ra}, \ce{^{131}I}, \ce{^{90}Sr}, and \ce{^{239}Pu}.
These nuclides have different origins and characteristics.
For instance, \ce{^{134}Cs}, \ce{^{137}Cs}, \ce{^{131}I}, and \ce{^{90}Sr} are representative fission products.
In particular, \ce{^{131}I} is an important short-lived nuclide in early-phase releases because it has a relatively
short half-life and emits characteristic $\gamma$ rays, whereas \ce{^{90}Sr} is a long-lived nuclide that predominantly emits
$\beta^-$ radiation and can contribute to long-term contamination even though it has no prominent $\gamma$ line.
In contrast, \ce{^{60}Co} and europium isotopes are mainly activation products originating from reactor structures or
from medical and industrial sources.
Moreover, \ce{^{226}Ra} is a naturally occurring radionuclide that has also been used historically in medical applications.
\ce{^{239}Pu} is a long-lived actinide that mainly emits $\alpha$ radiation; although it is less prominent in external
dose-rate fields compared with strong $\gamma$ emitters, it is relevant from the perspective of long-term radiological
hazards and contamination management.
The types of emitted radiation, representative $\gamma$-ray energies, and half-lives of these radionuclides are listed
in Table~\ref{table:half_time}~\cite{ref_ICRP107}.

Among them, \ce{^{137}Cs} is particularly problematic because it has a relatively long
half-life of about 30~years and emits a strong 662~keV $\gamma$ ray, so \ce{^{137}Cs}
released into the environment remains for decades and often dominates the long-term
dose.
\ce{^{60}Co} and \ce{^{154}Eu} have shorter half-lives than \ce{^{137}Cs}, but they can still persist in the environment for several years after an
accident and contribute appreciably to the radiation field, especially inside reactor
buildings and around activated components.
In realistic post-accident scenarios, these radionuclides may coexist, and appropriate
countermeasures such as shielding, decontamination, and waste management depend on
the specific isotopes present.
Therefore, it is necessary to identify the radionuclides in the environment rather than
relying only on total dose.

In this study, I focus on three representative $\gamma$-emitting radionuclides:
\ce{^{137}Cs}, \ce{^{60}Co}, and \ce{^{154}Eu}.
The spectrum unfolding and STE methods developed in the following
chapters are designed to identify these isotopes and to estimate their spatial distributions and strengths.

\begin{table}[hb]
    \centering
    \caption{Representative ~\cite{ref_ICRP107}.}
    \label{table:half_time}
    \begin{center}
        \begin{tabular}{lccc}
          \hline
          Radionuclide  & Emitted radiation
                        & Representative $\gamma$ lines [keV]
                        & Half-life \\
          \hline \hline
          \textbf{\ce{^{137}Cs}} & $\beta^-$, \textbf{$\gamma$}
                                 & \textbf{662}
                                 & \textbf{30.1 years} \\
          \ce{^{134}Cs}          & $\beta^-$, $\gamma$
                                 & 605, 796
                                 & 2.06 years \\
          \textbf{\ce{^{60}Co}}  & $\beta^-$, $\gamma$
                                 & 1173, 1332
                                 & 5.27 years \\
          \textbf{\ce{^{154}Eu}} & $\beta^-$, $\gamma$
                                 & 723, 873, 996, 1275, 1494, 1596
                                 & 8.6 years \\
          \ce{^{155}Eu}          & $\beta^-$, $\gamma$
                                 & 86.5
                                 & 4.76 years \\
          \ce{^{226}Ra}          & $\alpha$, $\gamma$
                                 & 186
                                 & 1600 years \\
          \ce{^{131}I}           & $\beta^-$, $\gamma$
                                 & 364
                                 & 8.0 days \\
          \ce{^{90}Sr}           & $\beta^-$
                                 & --
                                 & 29 years \\
          \ce{^{239}Pu}          & $\alpha$, $\gamma$
                                 & 129, 375, 414
                                 & 24{,}000 years \\
          \hline
        \end{tabular}
      \end{center}
\end{table}


\clearpage
\newpage

% ================================================== %
% section
% ================================================== %
\section{Radiation Detectors}
\label{chap2_rad_detector}
\hspace{9.5pt}

Various radiation detectors are used for radiation measurement depending on the purpose
of measurement and the type of radiation to be detected.
In radiation source--term estimation, two broad classes of detectors are commonly used:
\emph{non-directional} detectors, which only measure the number of incoming radiation
quanta, and \emph{directional} detectors, which provide both count and incident-direction
information.
This section summarises representative detectors in each class and explains the reason
for the detector choice adopted in this thesis.

\subsection{Non-directional Detectors}

Non-directional detectors register radiation entering the sensitive volume without resolving
its direction of arrival.
Because they do not rely on heavy collimators or imaging optics, they are typically compact,
robust, and well suited to deployment on small mobile robots operating in high-dose environments.
However, the lack of inherent directional information generally requires more measurement
poses and/or longer integration times to estimate the spatial distribution of sources.

In this thesis, non-directional detectors are classified based on the measurement output:
(i) \emph{energy-integrating} dose-rate meters that provide a scalar count-rate or dose-rate value,
and (ii) \emph{energy-resolving} spectrometers that record an energy spectrum.
Note that the same sensing medium (e.g., a scintillation crystal) can be used in either mode,
depending on the signal-processing electronics.

% -------------------------------------------------- %
\subsubsection{Energy-integrating dose-rate meters}

Energy-integrating dose-rate meters (survey meters) output a single scalar quantity such as
count rate or ambient dose equivalent rate, integrated over energy.
Typical instruments employ a Geiger--M\"uller tube, an ionisation chamber, or a scintillation
detector, and they are widely used for radiation-safety management owing to their simplicity,
robustness, and wide dynamic range.
Because the output does not contain spectral information, however, these instruments cannot
distinguish radionuclides and are therefore insufficient when isotope identification is required.

Figure~\ref{fig:chap2_survey_meter_principle} illustrates the operating principle and signal chain of a
GM-tube survey meter as a representative energy-integrating instrument.
Incident radiation produces ion pairs in the fill gas, and the resulting Townsend avalanche yields
a large current pulse.
These pulses are counted and, via calibration, converted into a scalar count-rate or dose-rate estimate,
without performing pulse-height analysis or constructing an energy spectrum.


% -------------------------------------------------- %
\subsubsection{Energy-resolving non-directional spectrometers}

For source-term estimation and radionuclide identification, it is desirable to obtain not only
the total count rate but also the \emph{energy spectrum} of detected $\gamma$ rays.
Energy-resolving non-directional spectrometers fulfill this requirement by combining a detector
(e.g., a scintillation crystal such as NaI(Tl), CeBr$_3$, or LaBr$_3$:Ce, or a semiconductor detector such as
HPGe or CdZnTe) with multichannel pulse-height analysis electronics.
The recorded spectrum enables identification of characteristic photopeaks and subsequent isotopic
decomposition, as described in Sections~\ref{chap2_spectrum_problem}--\ref{chap2_spectrum_procedure}.

Figure~\ref{fig:chap2_spectrometer} summarises a typical spectrometer signal-processing chain,
highlighting the key difference from dose-rate meters: pulse-height analysis and histogramming to produce
an energy spectrum.

\clearpage
\newpage

\subsection{Directional Detectors}

Directional detectors are designed to measure not only the number of incident $\gamma$
rays but also their approximate direction of arrival.
By exploiting collimation, scattering kinematics, or coded apertures, these systems can
directly form images of the radiation field.
Examples include gamma cameras, Compton cameras, and various collimated or coded-aperture
spectrometers.
In radiation source-distribution estimation, directional information greatly reduces the
number of required measurement locations.
However, most directional systems either require heavy shielding and collimators or
have limited count-rate capability, which complicates their use on small mobile robots
in high-dose environments.

\subsubsection{Gamma camera}

Figure~\ref{fig:chap2_gamma} shows the principle of a gamma camera, which is a representative directional detector.
A pixelated scintillation or semiconductor detector is placed inside a lead shield with a pinhole or multi-pinhole collimator.
Only $\gamma$ rays that pass through the pinhole are detected, and an inverted map of
radiation intensity is formed on the detector plane.
Because the count in each pixel directly corresponds to the local intensity, subsequent
image reconstruction is straightforward.

The main drawback of gamma cameras is the need for thick lead collimators to achieve
sufficient angular resolution and background rejection.
As a result, the total system mass typically reaches several tens of kilograms.
For compact ground robots that must traverse cluttered environments with narrow
passages and stairs, such payloads exceed the allowable mass, making the deployment of
gamma cameras difficult in practice.

\subsubsection{Compton camera}

Figure~\ref{fig:chap2_compton} illustrates the principle of a Compton camera, another
directional detector.
Unlike gamma cameras, Compton cameras do not rely on heavy collimators and can therefore be made relatively lightweight.
A typical system consists of two detector layers: a \emph{scatterer} and an \emph{absorber}.
An incident $\gamma$ ray first undergoes Compton scattering in the scatterer; the scattered $\gamma$ ray is then absorbed in the absorber.
By measuring the energy deposits and interaction positions in both layers, the direction and energy of the incident $\gamma$ ray can be reconstructed.
Each detected event constrains the source to lie on a so-called Compton cone, and by superimposing many cones, the source distribution can be estimated.

However, Compton cameras suffer from an upper limit on the usable count rate.
In high-dose environments, multiple $\gamma$ rays may enter the scatterer within a short time window, leading to overlapping interactions.
When two or more Compton scattering events occur simultaneously, it becomes
impossible to correctly associate the corresponding interactions in the scatterer and absorber, and the incident directions cannot be reconstructed reliably.

\subsubsection{Coded-aperture and scanned collimated detectors}

Coded-aperture and scanned collimated detectors constitute another class of directional
$\gamma$-ray imaging systems.
A coded-aperture system replaces a single pinhole with a patterned mask (typically a
high-$Z$ material such as tungsten or lead) placed in front of a position-sensitive detector.
Incoming $\gamma$ rays cast a characteristic shadow pattern on the detector, and an image of the
radiation field is reconstructed by decoding this pattern (e.g., correlation or maximum-likelihood
methods).
Because many mask openings contribute simultaneously, coded apertures can achieve higher
sensitivity than a single-pinhole gamma camera at comparable angular resolution, provided that
the mask pattern and detector geometry are well calibrated.

Scanned collimated detectors, in contrast, form directional measurements by mechanically
steering a collimator (or by sweeping the detector/collimator assembly) to sample multiple
viewing directions.
For example, a slit or parallel-hole collimator can be rotated to obtain a set of projection
measurements, from which the source distribution is reconstructed similarly to tomographic
imaging.
This approach is attractive when a high dynamic range or strong background rejection is required,
because the collimator defines a well-controlled field of view.

Despite their potential, both coded-aperture and scanned collimated systems face practical
limitations in high-dose, cluttered environments.
First, achieving adequate angular resolution and suppressing off-axis background generally requires
thick, high-density masks or collimators, which increases mass and volume and often exceeds the
payload constraints of compact mobile robots.
Second, under very high count rates, detector saturation, dead-time losses, and pile-up degrade
the recorded shadow patterns or directional projections, reducing reconstruction fidelity.
In addition, scattered radiation from surrounding structures and strong near-field sources can
violate the assumptions underlying simple decoding models (e.g., far-field geometry and limited
multipath scattering), leading to artifacts and false hot spots.
Finally, scanned systems require multiple orientations or dwell times to form an image, which
increases measurement time at each pose and is difficult to reconcile with the strict time and
dose constraints typical of reactor interiors.


\subsection{Selection of Detector}

In this thesis, the goal is to estimate the three-dimensional distribution of $\gamma$-ray
sources in a high-dose environment using a small ground robot.
The detector must therefore satisfy the following requirements:
\begin{enumerate}
  \item It must be mountable on a mobile robot with a limited payload capacity.
  \item It must operate reliably in high-dose environments, up to at least several Sv/h.
  \item It should provide energy-resolved spectra to enable radionuclide identification.
\end{enumerate}

Directional detectors such as gamma cameras and Compton cameras provide valuable
directional information but do not meet all of these requirements simultaneously:
gamma cameras are too heavy to be mounted on the robot, whereas Compton cameras
suffer from low upper dose-rate limits and cannot be used in the extreme high-dose
regions of interest.
On the other hand, non-directional spectrometers based on scintillation crystals are
lightweight, robust against high count rates, and capable of measuring energy spectra.

For these reasons, this thesis employs a non-directional, energy-resolving scintillation
spectrometer mounted on a mobile robot as the primary radiation sensor.
A qualitative comparison of the detector types considered is summarised in
Table~\ref{table:detector}.
Because the chosen detector does not provide inherent directional information, the
subsequent chapters develop methods that exploit attenuation by lightweight, actively
rotated shields to recover pseudo-directional information and to perform three-dimensional STE.

\begin{table}[tb]
  \caption{Comparison of radiation detectors.}
  \label{table:detector}
  \centering
  \begin{tabular}{ccccc}
    \hline
    Type
    & \makecell[c]{Mountable\\on robot}
    & \makecell[c]{High-dose\\environment}
    & \makecell[c]{Energy\\spectrum}
    & Directionality \\
    \hline\hline
    \textbf{\makecell[c]{Energy-resolving\\non-directional spectrometer\\(this work)}} & $\bigcirc$ & $\bigcirc$ & $\bigcirc$ & Non-directional \\ \hline
    \makecell[c]{Energy-integrating\\dose-rate meter\\(survey meter)}                 & $\bigcirc$ & $\bigcirc$ & $\times$   & Non-directional \\ \hline
    Gamma camera                                                                      & $\times$   & $\bigcirc$ & $\bigcirc$ & Directional     \\ \hline
    Compton camera                                                                    & $\bigcirc$ & $\times$   & $\bigcirc$ & Directional     \\ \hline
    \makecell[c]{Coded-aperture / scanned\\collimated detector}                       & $\triangle$& $\triangle$& $\bigcirc$ & Directional     \\ \hline
  \end{tabular}
\end{table}



\clearpage
\newpage
% ================================================== %
% section
% ================================================== %
\section{Spectrum Modeling and Problem Formulation}
\label{chap2_spectrum_problem}
\hspace{9.5pt}

Figure~\ref{fig:chap2_raw_spectra_example} shows representative raw $\gamma$-ray spectra acquired by the compact, non-directional scintillation spectrometer used in this thesis.
The unshielded measurement exhibits prominent photopeaks superimposed on a strong Compton continuum, whereas the shielded measurement (all sources blocked) yields substantially reduced counts over the entire energy range.
Such spectra are the primary measurement modality available on a small mobile robot in high-dose, cluttered environments; however, the particle-filter-based source-term estimation (STE) in Chapter~3 requires a compact observation vector rather than the full raw spectrum at each time step.
Therefore, each short-time spectrum must be converted into isotope-wise count observations that summarise the contributions of candidate radionuclides while accounting for baseline components, overlapping peaks, and dead-time effects.
To establish this conversion, Section~\ref{chap2_spectrum_problem} first formulates the spectrum as a Poisson linear model using a detector response matrix, and Section~\ref{chap2_spectrum_procedure} then describes the practical peak-based unfolding procedure that maps raw spectra to isotope-wise count sequences used as PF observations in Chapter~3.

At each robot pose, the spectrometer records a short-time raw spectrum
$\tilde{\bm{y}}\in\mathbb{N}^{K}$ (counts per energy bin), as exemplified in Fig.~\ref{fig:chap2_raw_spectra_example}.
The objective of this chapter is to convert each raw spectrum into
(i) a radionuclide identification result (which candidates are present) and
(ii) a compact isotope-wise observation vector that can be used by the particle-filter-based STE in Chapter~3.
This conversion is challenging because short acquisition times lead to high Poisson noise,
photopeaks can overlap due to limited energy resolution, and weak peaks may be masked by a strong Compton continuum and environmental/intrinsic background.
Moreover, high total count rates can introduce dead-time losses, which must be corrected before constructing consistent isotope-wise count sequences.

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
    \bm{b} = (b_{1}, b_{2}, \dots, b_{K})^{\mathrm{T}},
\end{align}
is the background spectrum (including environmental and intrinsic backgrounds).

The aim of spectrum unfolding is to estimate $\bm{q}$ from $\tilde{\bm{y}}$ and to determine which radionuclides are present.
In the following, the structure of the response matrix $\bm{R}$ and the statistical model of the spectrum are detailed.

% -------------------------------------------------- %
\subsection{Detector Response Matrix}
\label{subsec:spectrum_response}

For the $j$-th radionuclide, assume that the nuclear data library provides $L_{j}$ discrete gamma lines.
The $\ell$-th line has energy $E_{j\ell}$ and emission probability (branching ratio) $\beta_{j\ell}$ per decay.
Let $\epsilon(E)$ denote the full-energy peak detection efficiency at energy $E$, and let $\sigma(E)$ denote the energy resolution (standard deviation) of the detector, which is often approximated as
\begin{align}
    \sigma(E) = a\sqrt{E} + b ,
\end{align}
where $a$ and $b$ are calibration constants~\cite{ref_Tsoulfanidis1995}.

Assuming a Gaussian full-energy peak shape, the contribution of isotope $j$ to bin $k$ is modeled as
\begin{align}
    \label{eq:spectrum_Rkj}
    R_{kj}
      = \sum_{\ell = 1}^{L_{j}} \beta_{j\ell}\,\epsilon(E_{j\ell})\,G\!\left(E_{k}; E_{j\ell}, \sigma(E_{j\ell})\right),
\end{align}
where $G(E; \mu, \sigma)$ is the normalized Gaussian function
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


% -------------------------------------------------- %
\subsection{Statistical Model}
\label{subsec:spectrum_stat}

As gamma rays are emitted by stochastic radioactive decay processes, the number of counts in each bin is modeled as a Poisson random variable~\cite{ref_Tsoulfanidis1995}.
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

In practice, however, direct optimization of~\eqref{eq:spectrum_likelihood} over all bins and isotopes can be numerically challenging.
Therefore, as in Kemp \textit{et al.}~\cite{ref_kemp2024_tns}, a peak-based unfolding approach is adopted, in which the spectrum is first decomposed into individual peaks and then mapped to radionuclides.

\clearpage
\newpage

% ================================================== %
% section
% ================================================== %

\section{Spectrum Unfolding Procedure}
\label{chap2_spectrum_procedure}

This section describes the practical steps used to unfold the measured spectrum and to identify radionuclides.
As illustrated in Fig.~\ref{fig:chap2_raw_spectra_example}, each short-time spectrum contains discrete photopeaks superimposed on a broad continuum.
The unfolding pipeline below takes such a raw spectrum as input and outputs a set of corrected peak parameters (energies and net areas) and, ultimately, an isotope-wise count vector used as the PF observation in Chapter~3.
The procedure consists of the following steps:
\begin{enumerate}
  \item preprocessing (energy calibration and smoothing),
  \item peak detection,
  \item baseline estimation and net peak area computation,
  \item decomposition of overlapping peaks (spectral stripping),
  \item dead-time correction,
  \item radionuclide library matching and identification,
  \item construction of isotope-wise count sequences from successive spectra,
\end{enumerate}

Steps (1)--(5) transform the raw spectrum into a dead-time-corrected set of photopeaks with estimated energies and net areas, while Steps (6)--(7) match these peaks to a radionuclide library and aggregate them into isotope-wise count sequences.

These steps closely follow the approach of Kemp \textit{et al.}~\cite{ref_kemp2024_tns} and Anderson \textit{et al.}~\cite{ref_anderson2022_tase}.

% -------------------------------------------------- %
\subsection{Preprocessing: Energy Calibration and Smoothing}
\label{subsec:spectrum_preprocessing}

\subsubsection*{Energy Calibration}

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

\subsubsection*{Smoothing}

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

% -------------------------------------------------- %
\subsection{Peak Detection}
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

% -------------------------------------------------- %
\subsection{Baseline Estimation and Net Peak Area}
\label{subsec:spectrum_baseline}

Gamma-ray spectra typically exhibit a significant continuous component due to Compton scattering and environmental backgrounds.
Accurate radionuclide identification requires subtracting this baseline and computing the net area of each photopeak.

\subsubsection*{Baseline Estimation}

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
        p    & \text{if } y_{k} > b_{k} \\
        1-p  & \text{otherwise}
    \end{cases},
\end{align}
with $0 < p < 1$.
This choice forces the baseline to lie predominantly below the data, thus preserving peaks.

\subsubsection*{Net Peak Area}

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

% -------------------------------------------------- %
\subsection{Decomposition of Overlapping Peaks}
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
             {\beta_{jr}\,\epsilon(E_{jr})}m,
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

% -------------------------------------------------- %
\subsection{Dead-Time Correction}
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

% -------------------------------------------------- %
\subsection{Radionuclide Library Matching and Identification}
\label{subsec:spectrum_matching}

Having obtained a set of peak energies $\{E_{p}\}$, net areas $\{N_{p}\}$, and uncertainties $\{\sigma_{N_{p}}\}$, each peak must be associated with candidate gamma lines from the radionuclide library.

\subsubsection*{Energy Matching}

Following Anderson \textit{et al.}~\cite{ref_anderson2022_tase}, the absolute energy residual between
a measured peak energy $E_{\alpha}$ and a library line energy $E_{\beta}$ is defined as
\begin{align}
    d = \lvert E_{\alpha} - E_{\beta} \rvert .
\end{align}

Let $\sigma_{\alpha}$ and $\sigma_{\beta}$ denote the standard uncertainties of the measured and
library energies, respectively.
To accommodate calibration drift and the increase in effective matching tolerance with energy,
we scale the combined variance by an empirical factor $H(E)$~\cite{ref_anderson2022_tase}:
\begin{align}
    H(E) = 1 + \frac{E}{E_{0}},
\end{align}
where $E$ is the average energy of the pair, $E = (E_{\alpha}+E_{\beta})/2$, and $E_{0}$ is a
\emph{constant} energy scale parameter that controls how rapidly the matching tolerance increases
with energy.
In this study, we set $E_{0}=4000~\mathrm{keV}$ following Anderson \textit{et al.}~\cite{ref_anderson2022_tase}.

Define the variance associated with the energy residual as
\begin{align}
    \sigma_{d}^{2} = \zeta\, H(E)\left(\sigma_{\alpha}^{2} + \sigma_{\beta}^{2}\right),
\end{align}
where $\zeta$ is a tuning parameter.

Let the signed residual be $\Delta E = E_{\alpha}-E_{\beta}$.
Assuming $\Delta E$ is normally distributed with mean zero and variance $\sigma_{d}^{2}$, the
normalised deviation is
\begin{align}
    t = \frac{d}{\sigma_{d}}.
\end{align}

Let $\Phi(\cdot)$ denote the cumulative distribution function of the standard normal distribution.
The corresponding two-sided tail probability is
\begin{align}
    \label{eq:spectrum_Z}
    Z(d) = 2\left[1 - \Phi(t)\right].
\end{align}

Larger values of $Z(d)$ indicate a smaller deviation relative to the combined uncertainty and
thus a better match between the measured peak and the library line.
If $Z(d) > Z_{\mathrm{th}}$ for a predefined threshold $Z_{\mathrm{th}}$, the pair $(\alpha,\beta)$ is
considered a possible association.

\subsubsection*{Peak--Isotope Association and Detection Probability}

For each radionuclide $j$, let $\mathcal{P}_{j}$ be the set of peaks whose energies are compatible with at least one gamma line of $j$ according to the above criterion.
Anderson \textit{et al.}~\cite{ref_anderson2022_tase} employ a Bayesian framework based on the method of Stinnett and Sullivan to compute the probability that radionuclide $j$ is present.
Conceptually, for each isotope $j$, a likelihood ratio is constructed between the hypotheses
\begin{align}
    H_{0}: &\quad q_{j} = 0, \\
    H_{1}: &\quad q_{j} > 0,
\end{align}
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

% -------------------------------------------------- %
\subsection{Isotope--wise Count Sequence from Spectra}
\label{subsec:spectrum_isotope_counts}

For source--term estimation in later chapters, it is convenient to summarise each short-time spectrum as a compact vector of isotope--wise counts.
This subsection describes how such counts are constructed from the unfolded spectra obtained by the procedures in Sections~\ref{subsec:spectrum_preprocessing}--\ref{subsec:spectrum_matching}.

I assume that an energy--resolving detector (e.g., a scintillation detector) is used and that the gamma--ray spectrum is recorded at each measurement time along the robot trajectory.
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
  z_{k,h} = \sum_{p=1}^{P_h} w_{h,p} y_{k,h,p}, \\
  w_{h,p} = \frac{r_{h,p}}{\sum_{p'=1}^{P_h} r_{h,p'} }.
  \label{eq:pf_isotope_counts}
\end{align}

Thus, for each time step $k$, I obtain the isotope--wise count vector
\begin{align}
  \bm{z}_k = \{z_{k,h}\}_{h\in\mathcal{H}}.
  \label{eq:pf_isotope_vector}
\end{align}

These isotope--wise counts summarise the unfolded spectra in a form that is directly usable as observations for a set of parallel particle filters, one PF per isotope, in the multi--isotope source--term estimation framework of Chapter~3~\cite{ref_kemp2024_tns}.

\clearpage
\newpage

% ================================================== %
% section
% ================================================== %
\section{Summary}
\label{chap2_summary}
\hspace{9.5pt}

This chapter presented the gamma-ray spectrum unfolding and automated radionuclide identification method employed in this thesis.\par
Section~\ref{chap2_radiation} reviewed the basic properties of ionising radiation and radioactive decay, and summarised representative radionuclides relevant to nuclear accident scenarios, together with their representative $\gamma$-ray lines and half-lives.\par
Section~\ref{chap2_rad_detector} compared non-directional and directional detector classes and motivated the use of a compact, energy-resolving non-directional scintillation spectrometer on a small mobile robot, supported by a qualitative comparison of practical constraints in high-dose environments.\par
Section~\ref{chap2_spectrum_problem} modelled the measured spectrum as a linear combination of radionuclide responses and background using a detector response matrix and Poisson counting statistics, thereby formulating spectrum unfolding as a linear inverse problem over radionuclide activity parameters.\par
Section~\ref{chap2_spectrum_procedure} then described a practical peak-based unfolding pipeline: spectra are calibrated and smoothed, photopeaks are detected and quantified by baseline subtraction and net peak-area estimation, overlapping peaks are decomposed using library intensity ratios (spectral stripping), and dead-time losses are corrected using a non-paralysable model.\par
Detected peaks are matched to a radionuclide library based on energy compatibility, and the results are aggregated into isotope-wise count sequences.
These isotope-wise count vectors provide compact, robust observations that are directly used in the multi-isotope STE framework of Chapter~3.\par

\clearpage
\newpage
