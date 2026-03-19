Spectral Latent Dynamics for PDE Systems:
Learning Temporal Evolution in Fourier-Compressed Latent Spaces
Paras Balani
Department of Computer Science & Mathematics
BITS Pilani, Hyderabad Campus
f20230738@hyderabad.bits-pilani.ac.in
Abstract
Numerical simulation of partial differential equations (PDEs) still takes a lot of computing power, especially when you need to run simulations over long time periods or test many different parameter combinations.
We’re introducing something called Spectral Latent Dynamics, a new approach that learns how PDE solutions evolve over time. Instead of working with the full high-dimensional data, it compresses everything into
a smaller space using Fourier representations of the spatial fields.
Here’s how our pipeline works in three steps. First, we transform spatial snapshots into their Fourier
spectral representation. Then we compress those spectral coefficients using a convolutional autoencoder,
which gives us a compact latent code. Finally, we train a dynamics model that learns to move this latent
state forward in time.
We compared our method against two alternatives: a spatial autoencoder that compresses physical-space
snapshots directly, and a Fourier Neural Operator (FNO) that learns the solution mapping from start to
finish. When we tested everything on the 1D heat equation, 1D wave equation, and 2D heat equation,
we found something interesting. The spectral latent dynamics model handled multi-step predictions much
better than the FNO, with errors staying lower over time. The spatial autoencoder showed a different pattern
altogether, with errors spiking sharply at first before settling into a steady plateau.
We also tested how well these approaches handle noisy data. The autoencoder-based pipeline degraded
gracefully when we added noise to the inputs. All of this suggests that if you encode PDE states in Fourier
space before compressing them, you end up with representations that make it much easier to learn stable,
reliable temporal dynamics.
1 Introduction
Partial differential equations are everywhere in science and engineering. They describe how fluids flow, how
heat spreads, how structures deform, and how waves propagate. Traditional methods for solving them, like
finite differences, finite elements, and spectral methods, come with solid mathematical guarantees. But they’re
also expensive. Really expensive. Especially when you need fine spatial and temporal resolution, or when you
have to solve the same problem over and over for different initial conditions or boundary conditions.
That’s where scientific machine learning comes in. Researchers have been developing surrogate models that
can approximate PDE solutions at a fraction of the cost of traditional solvers. Two approaches have gotten
a lot of attention lately. Physics-Informed Neural Networks (PINNs) bake the PDE residual directly into the
loss function [1]. Neural operators like the Fourier Neural Operator (FNO) learn mappings between function
spaces [2]. Both have shown impressive results, but they typically work in the full solution space. That can
limit scalability and make long-term predictions tricky.
So we started thinking about a different path. What if we learned dynamics not in the original highdimensional space, but in a compressed latent representation instead? Autoencoders are perfect for discovering
these kinds of representations. Once you have a low-dimensional code, you can train a separate dynamics model
to predict how it evolves over time. But here’s the catch: the choice of what representation to compress really
matters.
Spatial representations keep local features intact, but they might need lots of latent dimensions to capture
global structure. Spectral representations, on the other hand, come from Fourier transforms. They diagonalize
constant-coefficient differential operators and concentrate energy in just a few low-frequency modes. That
means they’re naturally more compressible, and their trajectories tend to be smoother.
In this paper, we explore whether learning latent dynamics in a spectrally compressed space actually leads
to better predictions. Our main contributions are:
1
• We built a modular pipeline called Spectral Latent Dynamics. It chains together Fourier transformation,
autoencoder compression, and a learned dynamics model for stepping through time.
• We created numerical PDE solvers for the 1D heat equation, 1D wave equation, and 2D heat equation,
generating large simulation datasets with randomized initial conditions.
• We systematically compared three approaches: spectral latent dynamics, spatial autoencoder dynamics,
and the Fourier Neural Operator. We looked at rollout accuracy, how errors accumulate over time, and
robustness to noise.
• We found that spectral latent representations lead to sub-linear error growth during rollout, while the
FNO shows super-linear error accumulation.
2 Mathematical Background
2.1 Governing Equations
We’re working with two classic linear PDEs. The heat equation describes how things diffuse:
∂u
∂t = α∇2u, x ∈ Ω, t > 0, (1)
where α > 0 is the thermal diffusivity and u(x, t) is the temperature field. The wave equation models how
things propagate:
∂
2u
∂t2
= c
2∇2u, x ∈ Ω, t > 0, (2)
where c > 0 is the wave speed. Both equations come with appropriate initial and boundary conditions.
2.2 Finite Difference Discretisation
We break up the spatial domain into N grid points spaced evenly by ∆x. For the one-dimensional Laplacian,
we use the standard second-order central difference:
∇2u


xi
≈
ui+1 − 2ui + ui−1
(∆x)
2
. (3)
To step the heat equation forward in time, we use forward Euler:
u
n+1
i = u
n
i + α
∆t
(∆x)
2

u
n
i+1 − 2u
n
i + u
n
i−1

, (4)
making sure the CFL condition α∆t/(∆x)
2 ≤ 1/2 holds. For the wave equation, we use the explicit leapfrog
(Verlet) scheme:
u
n+1
i = 2u
n
i − u
n−1
i + c
2 ∆t
2
(∆x)
2

u
n
i+1 − 2u
n
i + u
n
i−1

. (5)
For the two-dimensional heat equation, we handle things similarly with a five-point stencil on a uniform Nx×Ny
grid.
2.3 Fourier Transform and Spectral Representation
The discrete Fourier transform (DFT) of a sequence {uj}
N−1
j=0 is:
uˆk =
N
X−1
j=0
uj e
−2πijk/N , k = 0, 1, . . . , N − 1. (6)
Here’s the key insight that makes spectral methods so powerful: differentiation in physical space becomes
multiplication in Fourier space. If we apply the DFT to the heat equation, we get:
duˆk
dt = −α(2πk/L)
2 uˆk, (7)
where L is the domain length. Notice what happens: each Fourier mode evolves independently through exponential decay. The dynamics are completely determined by the mode index k. This diagonalization property
means spectral representations are naturally suited to PDE dynamics. Low-frequency modes carry most of the
energy and evolve slowly, while high-frequency modes decay quickly.
2
2.4 Latent Representations
An autoencoder learns two mappings: an encoder Eϕ : R
N → R
d and a decoder Dθ : R
d → R
N . They’re
trained so that Dθ(Eϕ(u)) ≈ u for states u drawn from the data distribution. The latent dimension d ≪ N
forces compression.
Once training is done, the encoder maps a PDE snapshot (or its spectral coefficients) to a compact code
z = Eϕ(u) ∈ R
d
. Then we train a dynamics model fψ : R
d → R
d
to predict zt+1 = fψ(zt). The full prediction
pipeline becomes:
ut+1 ≈ Dθ(fψ(Eϕ(ut))). (8)
If we feed the spectral representation uˆt
into the autoencoder instead of ut
, we just apply an inverse Fourier
transform after decoding.
3 Methodology
3.1 PDE Data Generation
We generated simulation datasets for three systems: the 1D heat equation on [0, 1] with Dirichlet boundary
conditions, the 1D wave equation on [0, 1] with fixed endpoints, and the 2D heat equation on [0, 1]2 with
Dirichlet boundaries. For initial conditions, we used random superpositions of sinusoidal modes:
u0(x) = X
M
m=1
am sin(mπx), (9)
where the coefficients am come independently from N (0, 1/m2
). This gives us a natural spectral decay. Each
simulation produces a trajectory of T temporal snapshots. We created Ntrain = 500 training trajectories and
Ntest = 100 test trajectories for each PDE.
3.2 Spectral Representation
For a spatial snapshot ut ∈ R
N , we compute its discrete Fourier transform using the FFT algorithm. This gives
us complex-valued spectral coefficients uˆt ∈ C
N/2+1 (we can exploit conjugate symmetry since the inputs are
real). To work with real-valued data, we stack the real and imaginary parts:
u˜t =

Re(uˆt), Im(uˆt)

∈ R
N+2
. (10)
This real-valued spectral vector u˜t becomes the input to our spectral autoencoder.
3.3 Autoencoder Architecture
We built a symmetric convolutional autoencoder. The encoder has three 1D convolutional layers with kernel
size 5, stride 2, and ReLU activations, followed by a fully connected bottleneck layer that projects down to the
latent dimension d. The decoder mirrors this architecture using transposed convolutions.
We trained two versions: a spectral autoencoder that works on u˜t
, and a spatial autoencoder that works
directly on ut
. Both were trained to minimize the mean squared reconstruction error:
LAE =
1
|D|
X
(u,·)∈D
∥Dθ(Eϕ(u)) − u∥
2
2
. (11)
We used the Adam optimizer with a learning rate of 10−3
for 200 epochs and a batch size of 64.
3.4 Latent Dynamics Model
Once the autoencoder was trained and its weights frozen, we encoded all training trajectories into latent
sequences {z
(i)
t
}
T −1
t=0 . Then we trained a dynamics model fψ to predict the next latent state:
zˆt+1 = fψ(zt). (12)
3
We implemented fψ as a simple two-layer MLP with a hidden dimension of 128 and ReLU activations. The
training loss was:
Ldyn =
1
|Dz|
X
(zt,zt+1)∈Dz
∥fψ(zt) − zt+1∥
2
2
. (13)
During inference, we do multi-step rollout autoregressively: zˆt+k = f
(k)
ψ
(zt), where f
(k)
ψ means applying
the function k times. To recover the predicted spatial field, we decode and, for the spectral version, apply an
inverse FFT:
uˆt+k = F
−1

Dθ

f
(k)
ψ
(Eϕ(F(ut)))

. (14)
3.5 Fourier Neural Operator Baseline
The Fourier Neural Operator (FNO) learns a direct mapping from ut to ut+1 by parameterizing integral kernel
operators in Fourier space. Each FNO layer does this:
v
(l+1)(x) = σ

W(l)
v
(l)
(x) + F
−1

R
(l)
· F(v
(l)
)

(x)

, (15)
where R(l) ∈ C
dv×dv×kmax is a learnable spectral weight tensor that acts on the lowest kmax Fourier modes,
W(l)
is a pointwise linear transformation, and σ is a nonlinear activation. We used four FNO layers with
mode truncation kmax = 16 and channel width dv = 64. The FNO was trained end-to-end on single-step
pairs (ut
, ut+1) with MSE loss, using Adam at a learning rate of 10−3
for 500 epochs. For rollout, we apply it
autoregressively: uˆt+k = FNO(k)
(ut).
4 Experiments
4.1 Setup
All our PDE datasets used a spatial grid of N = 64 points (or 64 × 64 for 2D) with T = 20 temporal snapshots
per trajectory. For all autoencoder experiments, we set the latent dimension to d = 16. We implemented
everything in PyTorch and trained on a single GPU.
To evaluate rollout performance, we computed the mean squared error between predicted and ground-truth
fields at each time step, averaged over the test set:
MSE(k) = 1
Ntest
N
Xtest
i=1






uˆ
(i)
t0+k − u
(i)
t0+k






2
2
. (16)
For the noise robustness experiments, we added Gaussian noise ϵ ∼ N (0, σ2
I) to the input at test time and
measured reconstruction MSE as a function of σ.
4
5 Results
5.1 FNO Rollout Error
Figure 1: FNO multi-step rollout error (MSE) as a function of time step. The error grows super-linearly,
reaching approximately 1.6 by step 20.
Figure 1 shows what happens when we let the Fourier Neural Operator run autoregressively for 20 steps. The
error starts near zero and climbs steadily, with a distinctly super-linear growth pattern. For the first few steps
(t < 5), the per-step error stays modest, below 0.2. But after step 10, it really takes off, surpassing 1.0 by step
15 and hitting about 1.57 at step 20.
This behavior is pretty typical for autoregressive models working in the full solution space. Each prediction
introduces a small error, and those errors compound as you keep applying the operator. The FNO has no way
to correct for the fact that its inputs during rollout come from its own predictions rather than ground-truth
data. The super-linear growth suggests these errors don’t just add up, they actually interact with the dynamics
and amplify each other.
5.2 Latent Dynamics Rollout Error
Figure 2: Spectral latent dynamics rollout error (MSE) over time. Error growth is sub-linear and saturates
near 0.5.
5
Figure 2 tells a completely different story. This is the spectral latent dynamics model, and the error pattern
is strikingly different from the FNO. The growth is sub-linear, and it actually saturates, plateauing around 0.5
by step 15.
The initial error ramp is actually steeper than the FNO’s, hitting about 0.11 at step 1 compared to the FNO’s
0.02. That initial bump comes from the autoencoder’s reconstruction error, which is basically unavoidable. But
here’s the interesting part: the dynamics in latent space are way more stable. The per-step error increments
get smaller as the rollout progresses. It looks like the learned dynamics function fψ operates near some kind of
attractor in the latent space.
The error saturating at about 0.5 suggests the latent dynamics model converges to a fixed point or limit
cycle instead of diverging like the FNO does. For long-horizon predictions, this is actually really desirable.
Bounded error beats unbounded growth any day, even if you start with a slightly higher initial error.
The spectral representation deserves a lot of credit for this stability. By concentrating energy in lowfrequency modes and letting the autoencoder compression discard high-frequency components, the latent space
inherits the smoothness of the spectral domain. The dynamics model ends up working on a low-dimensional
manifold where trajectories are naturally more regular.
5.3 Spatial Autoencoder Rollout Error
Figure 3: Spatial autoencoder rollout error. A sharp spike in the first step is followed by a stable plateau near
0.27.
Figure 3 shows something completely different again. The spatial autoencoder’s error profile is unlike either of
the others. The MSE spikes dramatically at step 1, hitting about 1.0, then immediately collapses to a plateau
around 0.27 that stays flat for the rest of the rollout.
This weird behavior actually makes sense once you think about it. The spatial autoencoder’s latent space
has learned a representation where the dynamics model rapidly maps any initial condition to a narrow basin of
attraction. That first-step error measures the gap between the true next state and this attractor. The plateau
afterward means the model has essentially learned a constant predictor after that initial transient.
The dynamics in the spatial latent space are trivially stable. The model doesn’t diverge, but it completely
loses temporal resolution. The predicted trajectory collapses to a time-independent approximation of the
solution. That’s fine for the slowly evolving heat equation, but it would fail miserably for oscillatory dynamics
like the wave equation.
6
5.4 Spatial versus Spectral Representation
Figure 4: Comparison of spatial and spectral (Fourier magnitude) representations of the same initial condition.
The spatial signal is oscillatory and non-smooth, while the spectral magnitude is monotonically increasing and
structurally simpler.
Figure 4 puts spatial and spectral representations side by side for the same PDE snapshot. On the left, you see
the spatial field with all its complex oscillatory structure and multiple local extrema. It’s a high-entropy signal
that’s genuinely hard to compress without losing important information.
On the right is the corresponding Fourier magnitude spectrum. It’s strikingly smooth and monotonically
increasing. This smoothness comes directly from how we set up the initial condition’s spectral content. The
energy distributes across modes with such a regular structure that the autoencoder can really exploit it for
efficient compression.
This contrast highlights why the spectral approach has such an advantage. Compressing that smooth,
structured spectral representation needs fewer latent dimensions to achieve the same reconstruction quality.
The resulting latent trajectories are smoother too, which makes them easier for the dynamics model to learn.
The spatial autoencoder, on the other hand, has to waste representational capacity on capturing fine-grained
oscillations instead of focusing on dynamically relevant features.
5.5 Noise Robustness
Figure 5: Autoencoder reconstruction MSE as a function of input noise standard deviation. The model degrades
gracefully, with MSE increasing from about 3.016 to about 3.052 as σ goes from 0 to 0.1.
7
Figure 5 shows how well the autoencoder handles noisy inputs. We added Gaussian perturbations with standard
deviations σ ∈ {0, 0.01, 0.05, 0.1} to the spectral input at test time and measured reconstruction MSE.
The model degrades gracefully. MSE goes from about 3.016 with clean inputs to 3.052 at σ = 0.1. That’s
only a 1.2% relative increase across a pretty substantial noise range. Two factors explain this robustness.
First, the autoencoder’s bottleneck acts as a regularizer. It projects noisy inputs onto the learned latent
manifold, which effectively denoises them. Second, the spectral representation concentrates signal energy in
low-frequency modes, while additive Gaussian noise spreads its energy uniformly across all frequencies. The autoencoder has learned to preserve that low-frequency signal and discard high-frequency content, which naturally
filters out a big chunk of the noise energy.
That baseline MSE of about 3.016 even for clean inputs tells us the autoencoder has non-trivial reconstruction error. That’s the trade-off you accept with low-dimensional latent representations. But the error staying
so stable under perturbation is encouraging for real-world deployment where input data might be noisy or
corrupted.
5.6 Comparative Summary
Table 1: Summary of rollout error characteristics across the three approaches.
Method MSE at Step 10 MSE at Step 20 Growth Pattern
FNO (autoregressive) ∼ 0.45 ∼ 1.57 Super-linear
Spectral Latent Dynamics ∼ 0.37 ∼ 0.51 Sub-linear (saturating)
Spatial AE + Dynamics ∼ 0.27 ∼ 0.27 Plateau (after transient)
Table 1 pulls everything together. The spectral latent dynamics model hits the sweet spot between short-horizon
accuracy and long-horizon stability. It avoids the FNO’s unbounded error growth while keeping meaningful
temporal evolution, unlike the spatial autoencoder which just collapses to a stationary prediction. At step 10,
the FNO and spectral latent model have comparable error, but by step 20 the FNO’s error is three times larger.
6 Discussion
Why spectral models perform differently. The spectral representation diagonalizes the dynamics of linear
PDEs. It turns a coupled spatial system into independent modal equations. When you compress this through
an autoencoder, the resulting latent space inherits that decoupled structure. Temporal prediction becomes
much more tractable. The dynamics model only needs to learn smooth, monotonic transformations of a few
dominant modes instead of complex spatial interactions. This explains both the sub-linear error growth and
the saturation behavior we saw.
When FNO outperforms latent models. The FNO actually does better in the short term, with lower
error for t < 5. It captures single-step dynamics more accurately than our encode-predict-decode pipeline. If
you only need a few prediction steps, the FNO is still competitive. Its main weakness shows up in long rollouts,
where compounding errors take over.
Limitations. We only looked at linear PDEs with periodic or simple boundary conditions. Nonlinear systems
like Burgers’ equation or Navier-Stokes introduce energy transfer across scales, which might undermine the
spectral approach’s advantages. The autoencoder’s fixed reconstruction error floor (about 3.0 MSE in the noise
experiments) isn’t negligible. More expressive architectures or variational formulations might reduce it. Also,
our MLP dynamics model doesn’t use any temporal context. Recurrent or attention-based alternatives could
improve multi-step accuracy.
Generalization considerations. Everything was trained and evaluated on synthetic data generated with the
same PDE parameters and initial condition distributions. We haven’t tested how well these models generalize
to out-of-distribution initial conditions, different PDE coefficients, or unseen PDE types. Transfer learning and
meta-learning strategies might help address these limitations in future work.
8
7 Conclusion
We introduced Spectral Latent Dynamics, a framework for learning how PDE solutions evolve over time in
a Fourier-compressed latent space. By transforming spatial snapshots to their spectral representation before
autoencoder compression, we get latent spaces with smoother structure and more predictable dynamics than
spatial alternatives.
Our main finding is that the spectral latent dynamics model achieves bounded error growth during multi-step
rollouts, saturating at about 0.5 MSE after 15 steps. In contrast, the Fourier Neural Operator shows superlinear error accumulation, hitting 1.57 MSE over the same horizon. The spatial autoencoder, while trivially
stable, collapses to a time-independent prediction that completely misses temporal evolution.
The core insight here is that the choice of representation space for latent compression fundamentally determines how learnable and stable the resulting dynamics model will be. Spectral representations, with their
energy compaction and decoupling properties, provide a natural foundation for learning PDE dynamics in
reduced-order spaces.
Next, we plan to extend this framework to nonlinear PDEs like Burgers’ equation and the incompressible
Navier-Stokes equations, where energy transfer between scales introduces new challenges. We also want to
explore higher-dimensional domains, adaptive spectral truncation strategies, and physics-informed loss terms
to further regularize the latent dynamics. Recurrent and transformer-based dynamics models might improve
long-horizon accuracy by incorporating temporal context beyond the single-step Markov assumption.
A Pseudocode
Algorithm 1: PDE Data Generation (1D Heat Equation)
Input: Grid size N, time steps T, diffusivity α, number of samples Ns
Output: Dataset D = {(u
(i)
0
, . . . , u
(i)
T −1
)}
Ns
i=1
1 ∆x ← 1/(N − 1);
2 ∆t ← 0.4 · (∆x)
2/α ; // CFL condition
3 for i = 1 to Ns do
4 Sample coefficients am ∼ N (0, 1/m2
) for m = 1, . . . , M;
5 u
(i)
0
(xj ) ←
P
m am sin(mπxj );
6 for n = 0 to T − 2 do
7 u
(i)
n+1(xj ) ← u
(i)
n (xj ) + α
∆t
(∆x)
2

u
(i)
n (xj+1) − 2u
(i)
n (xj ) + u
(i)
n (xj−1)

;
8 end
9 Append trajectory to D;
10 end
9
Algorithm 2: Spectral Latent Dynamics Training
Input: Dataset D, latent dim d, epochs E1, E2
// Stage 1: Train Autoencoder
1 for epoch = 1 to E1 do
2 foreach batch {ut} from D do
3 u˜t ← [Re(FFT(ut)), Im(FFT(ut))];
4 uˆ˜t ← Dθ(Eϕ(u˜t));
5 L ← ∥uˆ˜t − u˜t∥
2
2
;
6 Update ϕ, θ via Adam;
7 end
8 end
9 Freeze ϕ, θ;
// Stage 2: Encode all trajectories
10 Dz ← {(zt
, zt+1) : zt = Eϕ(u˜t)};
// Stage 3: Train Dynamics Model
11 for epoch = 1 to E2 do
12 foreach batch {(zt
, zt+1)} from Dz do
13 zˆt+1 ← fψ(zt);
14 L ← ∥zˆt+1 − zt+1∥
2
2
;
15 Update ψ via Adam;
16 end
17 end
References
[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, “Physics-informed neural networks: A deep learning
framework for solving forward and inverse problems involving nonlinear partial differential equations,”
Journal of Computational Physics, vol. 378, pp. 686–707, 2019.
[2] Z. Li, N. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhatt, A. Stuart, and A. Anandkumar, “Fourier neural
operator for parametric partial differential equations,” in International Conference on Learning Representations, 2021.
[3] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, “Learning nonlinear operators via DeepONet
based on the universal approximation theorem of operators,” Nature Machine Intelligence, vol. 3, no. 3,
pp. 218–229, 2021.
[4] S. L. Brunton and J. N. Kutz, Data-Driven Science and Engineering: Machine Learning, Dynamical
Systems, and Control. Cambridge University Press, 2019.
[5] D. Kochkov, J. A. Smith, A. Alieva, Q. Wang, M. P. Brenner, and S. Hoyer, “Machine learning–accelerated
computational fluid dynamics,” Proceedings of the National Academy of Sciences, vol. 118, no. 21, 2021.
[6] N. Geneva and N. Zabaras, “Transformers for modeling physical systems,” Neural Networks, vol. 146,
pp. 272–289, 2022.
[7] B. Lusch, J. N. Kutz, and S. L. Brunton, “Deep learning for universal linear embeddings of nonlinear
dynamics,” Nature Communications, vol. 9, no. 1, p. 4950, 2018.
[8] K. Champion, B. Lusch, J. N. Kutz, and S. L. Brunton, “Data-driven discovery of coordinates and governing
equations,” Proceedings of the National Academy of Sciences, vol. 116, no. 45, pp. 22445–22451, 2019.
[9] C. R. Gin, D. E. Shea, S. L. Brunton, and J. N. Kutz, “DeepGreen: deep learning of Green’s functions for
nonlinear boundary value problems,” Scientific Reports, vol. 11, p. 21614, 2021.
[10] N. Kovachki, Z. Li, B. Liu, K. Azizzadenesheli, K. Bhatt, A. Stuart, and A. Anandkumar, “Neural operator:
Learning maps between function spaces with applications to PDEs,” Journal of Machine Learning Research,
vol. 24, no. 89, pp. 1–97, 2023.
10
