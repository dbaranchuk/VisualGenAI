# Notation and mathematical checks

These notes describe conventions in the slides corrected in commit `dee43e5`
(September 2026). Read the current PDF before explaining it; students may have
older copies. Page numbers are 1-based PDF pages, including animation builds.
Resolve filenames using the course map.

## Conventions that change across materials

- For `x_t = alpha_t x_0 + sigma_t epsilon`, the conditional noising score is
  `-epsilon/sigma_t`. The marginal score averages this over the posterior clean
  sample given `x_t`.
- The common FM path has `alpha_t=1-t` and `sigma_t=t`. Its conditional velocity is
  `epsilon-x_0`; sampling goes from noise at `t=1` to data at `t=0`.
- The solvers lecture also uses `x_t=sqrt(alpha_t)x_0+sqrt(1-alpha_t)epsilon`.
  Here `alpha_t` is a squared signal amplitude.
- An endpoint flow map `F(x_s,s,t)` returns `x_t`. An average velocity
  `F=(x_s-x_t)/(s-t)` has different units and boundary conditions. Both are
  called `F` in different lectures; state which meaning applies.
- Define CFG through its formula: `s_cfg=s_uncond+gamma(s_cond-s_uncond)`.
  In the alternative form `(1+w)s_cond-w*s_uncond`, `gamma=1+w`.

## Useful checks when explaining or implementing the slides

| Topic and source | Check |
|---|---|
| DDPM/DSM pp.56, 114–115 | The reverse conditional mixture averages over the posterior clean sample given the noisy input. Classifier guidance differentiates the selected class log probability after log-softmax. |
| Continuous diffusion pp.23, 28 | With reverse clock `1-t`, evaluate the backward-ODE diffusion coefficient as `g(1-t)^2/2`. |
| Continuous diffusion p.33; flow matching p.13 | Prediction reparameterizations change timestep-dependent loss weights. For the linear FM path, noise MSE has weight `1/(1-t)^2`, and clean-data MSE has weight `1/t^2`, for `0<t<1`. |
| Solvers p.30 | For signed reverse step `h<0`, retain signed `h` in the drift and use noise variance `abs(h)` and amplitude `sqrt(abs(h))`. |
| Flow maps p.16 | Holding `x_0` fixed in `dx/dt=(x-x_0)/t` gives `x_(n-1)=x_0+(t_(n-1)/t_n)(x_n-x_0)`. |
| Flow maps pp.19, 23 | With VE variance `t_n^2` and clean boundary zero, inject `t_n*epsilon`; visit sampling noise levels in decreasing order. |
| Distribution matching pp.21–22 | For average velocity `F(x_t,t,s)`, use `F=v+(s-t)D_t F`. The total derivative follows the trajectory while holding `s` fixed. |
| Distribution matching p.59 | A posterior score identity alone does not establish an unbiased generator-gradient substitution: the generator Jacobian depends on the same clean sample. |
| Distribution matching pp.91–104 | The Gaussian score connection has explicit kernel and normalization assumptions. Keep that derivation distinct from the paper's kernel and implementation details when comparing methods. |
| Visual AR pp.8, 10 | A reconstruction loss minimized with positive codebook and commitment penalties uses negative log likelihood. |
| 3D p.12 | In clean-image parameterization, the SDS gradient uses a positive weight times `J_theta^T (x_0-xhat_0)`. Its regression surrogate stops the teacher target and includes the matching factor of one half. |

Useful primary sources: [MeanFlow](https://arxiv.org/html/2505.13447v1),
[consistency sampling, Algorithm 1](https://arxiv.org/html/2303.01469v2),
[classifier-guidance implementation](https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py),
[DMD](https://arxiv.org/html/2311.18828v3),
[DreamFusion](https://arxiv.org/abs/2209.14988), and
[Drifting](https://arxiv.org/html/2602.04770v1).
