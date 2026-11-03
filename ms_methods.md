# Methods

## Generative model

Each level $i$ comprises an exchangeable population of units with a categorical state $s\in\{1,\dots,q_i\}$ and all-to-all recurrent excitation (a complete-graph Potts model). The mean-field posterior is symmetric among non-winning categories, so it is fully described by $p_i=[1+(q_i-1)x_i]/q_i$ for the winner and $r_i=(1-x_i)/q_i$ for each other category. Equivalently, $Q_i=x_i\delta_{s,s^*}+(1-x_i)\,\mathrm{Unif}$. By definition of the winner, $x_i\ge0$. The prior energy per unit is $-\tfrac12 gJ_iJ_s\sum_sQ_i(s)^2$, with $\sum_sQ_i(s)^2=[1+(q_i-1)x_i^2]/q_i$. Self-support follows $J_i=J_0(1+\eta\ell_i)$, with $\ell_i$ the level's position in $[0,1]$ and $\eta=0.68$ from [25].

## Likelihood and precision

A bodily observation is treated as generated either by a committed component, as a Gaussian with precision $g$ and expected squared prediction error $\omega d^2$, or by a broad null component whose log-density does not depend on $g$. The expected negative log-likelihood contributed by level $i$ is $w_ix_i[\tfrac12g\omega d^2-\tfrac12\ln g+c]$, with $\sum_iw_i=1$ and $c$ setting the null relative to a unit-precision component.

**Assumption A1.** The mixture weight $x_i$ is the posterior probability of a latent commitment indicator. Under the categorical posterior itself, a flat belief would still give the winning category weight $1/q_i$, so that $W$ would be $\sum_iw_ip_i$ and flat-state precision would lie slightly below $a/b$ (by about 8% at the last collapse with baseline values, similar in size to the baseline rebound). With $W$ so defined the rebound is «a1_u» instead of «reb_u», and «a1_t» instead of «reb_t» at $\lambda=2$, so it shrinks but does not disappear. The criterion below is unaffected in sign, but the statement that flat-state precision is $a/b$ holds only under A1.

**Assumption A2.** With a hyperprior $g\sim\mathrm{Gamma}(a,b)$ (shape $a$, rate $b$), the conjugate posterior has shape $a+\tfrac12W$ and rate $b+\tfrac12\omega d^2W$, and I track its mean $g^*=(a+\tfrac12W)/(b+\tfrac12\omega d^2W)$. Only the likelihood informs $g$. The same value is then applied as a common gain to prior and sensory messages, on the view that a single neuromodulatory mechanism sets synaptic gain [12,13]. This is not a consequence of a single variational functional. If $g$ were treated as the inverse temperature of a normalised prior, the $g$-gradient would contain the extra term $-\langle U\rangle_Q+\langle U\rangle_{p}$. That term vanishes at the global mean-field solution but not on metastable branches, where the collapses occur, and there its size («om_lo»–«om_hi» per level at the lower critical point) can exceed the retained denominator («den_lo»–«den_hi») considerably. The rebound therefore exists only under A2. $F$ is a Lyapunov function for beliefs at fixed $g$, not for the joint system. The residual of the sensory evidence $h_i$ is treated as absorbed into $d^2$.

Comparing $g^*$ at $W>0$ with its value at $W=0$ gives the criterion in Results: $g^*(W=0)>g^*(W>0)$ if and only if $\rho\equiv\omega d^2a/b>1$, independently of $W$.

## Paths

**Failing support.** $a(t)=a_0(1-t/T)$ and $d^2(t)=e_0^2+Dt/T$, with sensory evidence withdrawn over the first 20% of the path. Path length is uncalibrated and does not correspond to a clinical duration. **Prior relaxation.** $a$ and $d^2$ are constant; sensory evidence is withdrawn; $J_s$ falls from 1 to 0.2, then rises to 0.28 and returns to 1, with sensory evidence restored at the end. Precision is computed from the same update on both paths. **Level-wise relaxation.** $J_s$ is replaced by a level-specific logistic function with staggered onset; the order of onset is imposed.

## Numerics

Beliefs were integrated by forward Euler in $u$ with step 0.002–0.04 and $x$ clipped to $[0,1]$. A level was scored as lost at $x_i<0.15$ and as returned at $x_i>0.5$. Quasi-static thresholds were obtained by stepping $g$ in increments of 0.005 with relaxation of 4 time units per step (Table 1); Table S3 and the lesions used 0.01, and the sweep 0.02. Hysteresis loop area was computed for single-sweep durations from 5 to 2000 units and extrapolated with $a_\infty+cT^{-p}$. A symmetry-breaking field of $2\times10^{-3}$ was applied at every level. Stochastic simulations added Gaussian noise of variance $2\sigma^2\,\mathrm{d}t/\tau_i$ to $u_i$, with $\sigma^2$ between 0.002 and 1.

## Observables

$M=GD$, with $G=g[\mathrm{diag}(J_iJ_s)+\mathrm{diag}(q_i/(q_i-1))K]$, $K$ the coupling matrix, and $D=\mathrm{diag}(\mathrm{d}x_i/\mathrm{d}u_i)$, set to zero at levels clipped at $x_i=0$. Fluctuations were analysed by linearising the simulated log-odds dynamics, $\dot{\delta u}=\mathrm{diag}(\tau_i^{-1})(M-I)\delta u+\xi$, solving the Lyapunov equation, and integrating the analytic spectrum of $\sum_i\delta u_i^2$. Stochastic interaction [29] at lag 50 ms was evaluated over all bipartitions; the partition minimising the value normalised by the lesser part's predictive information was taken as the minimum-information bipartition.

## Simulated perturbation

Each state was held with $g$ clamped and with its own mismatch, and $u$ was first relaxed for 20 time units, since a clipped level sits at a negative fixed point of $u$. Two pulses were used, each at 0.5 s to L4 or to L7, in 300 paired noisy trials with common random numbers ($\sigma^2=0.002$). An evidence pulse ($\delta h=10$ for 20 ms) enters through the gain-weighted channel, as a sensory input would. A direct pulse (1 nat added to $u_i$) bypasses gain at the stimulated site, as a closer analogue of magnetic stimulation. The evoked response is the trial-wise difference from the unperturbed twin. I report the area under $|\delta u_i|$, the part of it outside the stimulated level, the area under the change in commitment, the number of levels with a significant response ($|z|>4$ and $|\text{mean}|>0.02$), and the duration of significance in a 2.5 s window. With seven sources a complexity index would not be meaningful, so I did not compute one.

## Verification

The critical points $s(q)$ and $2(q-1)(q-2)^{-1}\ln(q-1)$ were compared with numerical solutions, and the analytic $R$ of an isolated level with an independent finite-difference solution of the fixed point (Table S2). $F$ was evaluated along trajectories from 50 random initial conditions at fixed $g$ to test the Lyapunov property, and its finite-difference gradient was evaluated at relaxed interior levels. Mean-field hysteresis was compared with heat-bath Monte Carlo of the complete-graph Potts model ($N=600$, $q=7$). Dwell-time distributions were compared with the exponential distribution using a Kolmogorov–Smirnov test after shifting by the minimum and estimating the mean from the data, so the p-values are approximate.

## Parameter sweep

800 parameter sets were drawn uniformly: $q_s\in[4,60]$; $q_5,q_6\in[8,60]$; $q_7\in[3,60]$; $J_0\in[7,12]$; $\eta\in[0,1.2]$; $\kappa\in[0,1.5]$; $\tau$ ratio in $[3,8]$; weight tilt $\lambda\in[0,3]$; $c\in[-2,0]$; $a_0\in[0.8,2]$; $e_0^2\in[0.2,3]$; $D\in[0,16]$; $T\in[150,600]$; $\tau_g\in[0.5,8]$. Sets in which the waking state was not ordered were excluded.
