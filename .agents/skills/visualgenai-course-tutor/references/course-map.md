# Course map

Snapshot: 28 September 2026. Paths below are relative to the repository root. Resolve them there, not relative to this skill folder. The README is the current source for recording links and syllabus numbering. Prerequisites and study checkpoints below are tutoring recommendations, not additional official requirements.

## Lectures and practice

| Syllabus topic | Subject | PDF | Useful prerequisites | Practice/checkpoint |
|---|---|---|---|---|
| 1 | DDPM and DSM | `course_materials_spring_2026/week1_ddpm_dsm/ddpm_dsm_lecture.pdf` | Probability, conditional expectation, gradients | HW1: Gaussian PF-ODE trajectories and ideal denoisers |
| 2 | Continuous diffusion | `course_materials_spring_2026/week2_continuous_time_diffusion/continuous_time_diffusion_lecture.pdf` | Topic 1; ODEs and Gaussian increments | Revisit HW1; derive SDE/PF-ODE relations |
| 3 | Flow matching | `course_materials_spring_2026/week3_flow_matching_and_solvers/flow_matching_lecture.pdf` | Topics 1–2; conditional expectation | HW3: implement flow-matching training |
| 4 | ODE/SDE solvers | `course_materials_spring_2026/week3_flow_matching_and_solvers/solvers_lecture.pdf` | Topics 2–3; Taylor expansion | HW2: Euler, DPM solvers, timestep schedules |
| 5 | Practical diffusion | `course_materials_spring_2026/week4_practical_dm/practical_dm_lecture.pdf` | Topics 1–4; neural-network basics | HW3: pixel/latent training and representation alignment |
| 6 | Flow maps | `course_materials_spring_2026/week5_few_step_models_flow_maps/few_step_flow_map_lecture.pdf` | Topics 2–4; stop-gradient and distillation | HW4: consistency distillation and simplified MeanFlow |
| 7 | Distribution matching | `course_materials_spring_2026/week6_few_step_models_distribution_matching/distribution_matching_lecture.pdf` | Scores, KL, topic 6; adversarial training | HW5: latent ADD and MMD distillation |
| 8 | Visual autoregression | `course_materials_spring_2026/week7_visual_ar_models/visual_ar_models_lecture.pdf` | Topic 5; tokenization, attention, likelihood | HW6: train an FM head with a frozen MAR backbone |
| 9–10 | Video generation and efficient diffusion | `course_materials_spring_2026/week8_video_diffusion_and_efficient_genai/video_generation_and_efficient_genai_lecture.pdf` | Topics 4–5 and 8; causal attention | HW7: AR video sampling and KV caches |
| 11 | Multimodal generation and conditioning | `course_materials_spring_2026/week9_multimodal_generation_and_conditioning/multimodal_generation_and_conditioning.pdf` | Topics 5 and 8; guidance | Compare conditioning architectures; exam topic 8.2 |
| 12 | 3D generation | `course_materials_spring_2026/week10_3d_generative_models/3d_generative_models_lecture.pdf` | Topics 5 and 7; differentiable rendering as needed | Derive SDS; compare multi-view models; exam topic 8.3 |

## Seminar and supplementary slides

- Architecture seminar: `course_materials_spring_2026/week4_practical_dm/dm_architectures_seminar.pdf`.
- Flow-map summary: `course_materials_spring_2026/week5_few_step_models_flow_maps/flow-map-summary.pdf`.
- AR video seminar: `course_materials_spring_2026/week8_video_diffusion_and_efficient_genai/autoregression_video_diffusion_seminar.pdf`.

## Assignment entry points

- HW1: `course_materials_spring_2026/assignments/hw1_diffusion_fundamentals/practice_template.ipynb`.
- HW2: `course_materials_spring_2026/assignments/hw2_diffusion_solvers/practice_dpm_solver.ipynb`.
- HW3: `course_materials_spring_2026/assignments/hw3_flow_matching/fm_training.ipynb`.
- HW4: `course_materials_spring_2026/assignments/hw4_flow_maps/flow_map_models.ipynb`.
- HW5: `course_materials_spring_2026/assignments/hw5_distribution_matching/add_mmd_distillation.ipynb`.
- HW6: `course_materials_spring_2026/assignments/hw6_masked_ar/mar_fm_head.ipynb`.
- HW7: `course_materials_spring_2026/assignments/hw7_video_ar_diffusion/ar_video_diffusion_sampling.ipynb`.

Also read HW1's `task.pdf` and HW2's `theory.pdf` in their assignment folders. For execution, inspect notebook setup cells and existing environments first. Assignments 2–7 involve pretrained models and/or GPU work; do not promise a fixed runtime or memory requirement without checking their configuration. Do not execute cells just to read a notebook.

## Revision route

Use `course_materials_spring_2026/exam/exam_topics_visual_genai.md` for the official topic list. A useful foundation-first route is probability/score/ELBO → SDE/PF-ODE → flow matching → solvers/guidance → practical architectures → flow maps/distribution matching → AR/video/multimodal/3D. For a narrow question, revisit only the missing prerequisite.

Check understanding with derivations (score/noise conversion, reverse-time signs), tiny implementations (one sampler step), and comparisons (flow maps versus distribution matching). The exam booking application is administrative, not study material.

## Research Seminars on Diffusion Models

- Searchable index: `research_seminars/INDEX.md` — dates, descriptions, recording links, and paper references.
- PDF archive: `research_seminars/diffusion_research_seminars_09_2026.pdf` — English guide with recording access codes.
- Slide folder: `research_seminars/slides/` — check which files currently exist; it was a placeholder at this snapshot.
- Group: https://t.me/+gE2ERaknHecyMjZi

Search the archive by topic or paper title. Return the exact entry and original link, and disclose when the recording itself was not reviewed. Course recordings and research seminar recordings are separate resources.
