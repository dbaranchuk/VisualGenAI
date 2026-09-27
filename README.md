# Visual GenAI Course

<p align="center">
<img src="assets/logo_visual_genai.jpg" width="1080px"/>
</p>

<p align="center">
  <a href="#syllabus"><img src="https://img.shields.io/badge/lectures-12-blue" alt="lectures"/></a>
  <a href="#assignments"><img src="https://img.shields.io/badge/assignments-7-green" alt="assignments"/></a>
  <a href="#exam"><img src="https://img.shields.io/badge/exam-topics-orange" alt="exam"/></a>
  <img src="https://img.shields.io/badge/lectures-EN%20slides%20%2F%20RU%20video-lightgrey" alt="language"/>
</p>

A graduate-level course on modern generative modeling for visual domains — **images, video, and 3D**.
The main focus is on **diffusion models**: their theoretical interpretations and the advanced
training and sampling methods behind today's high-quality, fast generators. Special attention is
given to **few-step** diffusion-based models used in production image/video services, and to recent
advances in **autoregressive** visual generation and its integration with diffusion.

## Goals

* Develop a deep understanding of leading visual generative paradigms.
* Learn novel, effective diffusion-based generative frameworks.
* Master the most recent training and inference techniques behind state-of-the-art generative models.

## Contents

* [Syllabus](#syllabus)
* [Research Seminars on Diffusion Models](#research-seminars-on-diffusion-models)
* [Assignments](#assignments)
* [Exam](#exam)
* [Contacts](#contacts)
* [Course staff](#course-staff)
* [References](#references)

<hr>

## Syllabus

Each lecture comes with English slides (`.pdf` in this repo) and a recorded lecture/seminar (RU).

|  # | Topic | Materials |
|:--:|-------|-----------|
|  1 | **Introduction to Diffusion Models** — Denoising Diffusion Probabilistic Models (DDPMs) & Denoising Score Matching (DSM) | [Slides](week1_ddpm_dsm/ddpm_dsm_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/kkxLSdtAVcVxgw) · [Seminar (RU)](https://disk.yandex.ru/i/D_2_7Wj-ym4QEQ) |
|  2 | **Continuous Diffusion Models** — Probability Flow ODE and SDE formulations | [Slides](week2_continuous_time_diffusion/continuous_time_diffusion_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/YkIe5QbG7CdorA) |
|  3 | **Flow Matching** and its connection to diffusion models | [Slides](week3_flow_matching_and_solvers/flow_matching_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/jAJW937u__lmtg) |
|  4 | **Efficient PF-ODE/SDE Solvers** — Euler methods, DDIM, and DPM-Solver | [Slides](week3_flow_matching_and_solvers/solvers_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/DeJsKA1M4Qr9Zw) |
|  5 | **Diffusion Models in Practice** — diffusion spaces, recent architectures, design choices, training & sampling techniques | [Slides](week4_practical_dm/practical_dm_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/rW8GEXnSNN0avQ) · [Seminar (RU)](https://disk.yandex.ru/i/1dVEoOIKN0-4bA) |
|  6 | **Flow Map Models** — learnable PF-ODE integrators for faster sampling (Consistency Models, MeanFlow) | [Slides](week5_few_step_models_flow_maps/few_step_flow_map_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/ESwQxH5EXe3f5g) · [Supplementary](week5_few_step_models_flow_maps/flow-map-summary.pdf) |
|  7 | **Distribution Matching** for few-step generators (DMD, ADD, SwD, Drifting Models) | [Slides](week6_few_step_models_distribution_matching/distribution_matching_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/XGViuRbTRNBeDg) |
|  8 | **Autoregressive Visual Generation** — discrete tokenizers VQ-VAE/VQ-GAN, scale-wise models (VAR, Switti), continuous AR (MAR), diffusion as AR | [Slides](week7_visual_ar_models/visual_ar_models_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/3dE0XlI793U_DA) |
|  9 | **Video Generation** — architectures, challenges, and AR video diffusion models | [Slides](week8_video_diffusion_and_efficient_genai/video_generation_and_efficient_genai_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/ITGNF6ukuy1ufQ) · [Seminar (RU)](https://disk.yandex.ru/i/DPQkF553n4rkeg) |
| 10 | **Efficient Diffusion Models** — model-level optimizations (caching, sparse attention, quantization, …) | [Slides](week8_video_diffusion_and_efficient_genai/video_generation_and_efficient_genai_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/ITGNF6ukuy1ufQ) |
| 11 | **Multimodal Generative Models** — architectures, training setups, and conditioning in diffusion (ControlNet, IP-Adapter) | [Slides](week9_multimodal_generation_and_conditioning/multimodal_generation_and_conditioning.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/EVl_Y3fF0KL8SA) |
| 12 | **3D Generative Models** — intro to 3D modeling and multi-view diffusion models | [Slides](week10_3d_generative_models/3d_generative_models_lecture.pdf) · [Lecture (RU)](https://disk.yandex.ru/i/kAUxkOnFmZxJmw) |

<hr>

## Research Seminars on Diffusion Models

* [Seminar slides](research_seminars/slides/)
* [Research seminar Telegram group](https://t.me/+gE2ERaknHecyMjZi)

<hr>

## Assignments

Homeworks live in [`assignments/`](assignments). Each contains a starter notebook (and a `task.pdf`
/ `theory.pdf` where applicable).

|  # | Topic | Starter |
|:--:|-------|---------|
| 1 | Intro to diffusion: DDPM & DSM | [`hw1/`](assignments/hw1) — `practice_template.ipynb`, [`task.pdf`](assignments/hw1/task.pdf) |
| 2 | Efficient solvers: DPM-Solver | [`hw2/`](assignments/hw2) — `practice_dpm_solver.ipynb`, [`theory.pdf`](assignments/hw2/theory.pdf) |
| 3 | Flow Matching training | [`hw3/`](assignments/hw3) — `fm_training.ipynb` |
| 4 | Flow Map Models | [`hw4/`](assignments/hw4) — `flow_map_models.ipynb` |
| 5 | Distribution matching: ADD / MMD distillation | [`hw5/`](assignments/hw5) — `add_mmd_distillation.ipynb` |
| 6 | Autoregressive generation: MAR with a flow-matching head | [`hw6/`](assignments/hw6) — `mar_fm_head.ipynb` |
| 7 | AR video diffusion sampling | [`hw7/`](assignments/hw7) — `ar_video_diffusion_sampling.ipynb` |

<hr>

## Exam

The full list of examinable topics is in
[`exam/exam_topics_visual_genai.md`](exam/exam_topics_visual_genai.md). The
[`exam/slot-booking-service/`](exam/slot-booking-service) is a small web app students use to book
exam slots (see its own [README](exam/slot-booking-service/README.md)).

<hr>

## Contacts

* [Dmitry Baranchuk](mailto:dmitrybaranchuk@gmail.com) — Telegram [@vernold](https://t.me/vernold)
* [Nikita Starodubcev](mailto:jke013333@gmail.com) — Telegram [@nikitastariy](https://t.me/nikitastariy)

## Course staff

* [Dmitry Baranchuk](https://dbaranchuk.github.io/)
* [Nikita Starodubcev](https://scholar.google.com/citations?user=o6pRm_gAAAAJ&hl=en)
* [Denis Rakitin](https://scholar.google.com/citations?user=zIl8Z3gAAAAJ&hl=en)
* [Denis Kuznedelev](https://scholar.google.com/citations?user=L78B2lcAAAAJ&hl=en)
* [Ilya Drobyshevsky](https://scholar.google.com/citations?user=BovM6psAAAAJ&hl=en)
* [Ilya Sudakov](https://scholar.google.com/citations?user=R4hnjs4AAAAJ&hl=en)
* [Sergey Kastrulin](https://scholar.google.com/citations?user=765_fJYAAAAJ&hl=en)
* [Kirill Struminsky](https://scholar.google.com/citations?user=q69zIO0AAAAJ&hl=en)

## References

* The introduction to diffusion models follows [CS236](https://deepgenerativemodels.github.io/) by Stefano Ermon.
* Some explanations are inspired by [The Principles of Diffusion Models](https://the-principles-of-diffusion-models.github.io/).
* Numerous papers and blog posts that led us to this course.
