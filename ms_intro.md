# Introduction

Most theories of consciousness are built from contrasts between conditions with and without a given content. Reports of awareness in the absence of any content therefore pose a sharp test. Long-term meditators describe brief states of vivid, non-dual presence with no thought, self or percept [1–3], and high-density EEG recorded during such states shows a widespread fall in gamma power without the rise in delta power that accompanies sleep [3]. In the terms of that study, the cortex appears "widely awake but minimally active". Integrated information theory (IIT) predicted this pattern [3,4]. Its authors argue that the pattern challenges accounts in which experience depends on computation or information processing [3]. The adversarial test of IIT and global neuronal workspace theory found substantial problems for both, in different places [5]. A second collaboration, comparing IIT with predictive-processing accounts, is under way [6].

A second gap concerns transitions. Theories are usually compared on static contrasts, yet the sequence in which experience comes apart and returns is informative. Emergence from anaesthesia is not the mirror image of induction: it shows hysteresis not explained by pharmacokinetics [7,8], and the return of responsiveness has been associated with activation of brainstem, thalamic and anterior cingulate regions rather than neocortex [9]. Few formal models address sequence, hysteresis and contentless awareness together.

Active inference is well placed to do so. It treats the brain as a hierarchical generative model that minimises variational free energy [10,11] and assigns a central role to precision, the inferred reliability that weights each signal and is thought to be implemented by synaptic gain and neuromodulation [12,13]. Sandved-Smith et al. modelled meta-awareness as a higher level that controls precision [14]. Laukkonen, Friston and Chandaria proposed that consciousness requires a reflexive sharing of the world model throughout the hierarchy, termed epistemic depth, and that minimal phenomenal experience combines maximal depth with a maximally simple model [15]. Active inference has also been described as a framework for, rather than a theory of, consciousness [16,17], and whether its claims can be made quantitative and falsifiable remains open.

Here I use a classical contemplative account as a source of formal constraints. Indo-Tibetan Highest Yoga Tantra contains an unusually explicit process model of the dissolution and re-emergence of experience, developed in the Guhyasamāja literature and transmitted in manuals on dying, sleep and meditation [18–21]. I do not assess its metaphysical claims. I extract four structural claims that any quantitative model could reproduce or fail to reproduce:

1. **Carrier and content.** Every cognition rides on a "wind" (*rlung*, Skt. *prāṇa*) that carries content without being content. Conceptual states are classified by the strength of the wind they require [22].
2. **Ordered dissolution.** Under failing bodily support, sense faculties withdraw in a fixed order, followed by three classes of conceptuality, conventionally numbered 33, 40 and 7, ending in a contentless "clear light" (*'od gsal*) [18–21]. I adopt the staged version, in which each class ceases at its own stage; many Gelug sources have all eighty ceasing together [18].
3. **Reverse emergence.** Experience returns through the same stages in reverse order [18].
4. **Two kinds of contentlessness.** The stage preceding clear light is described as a darkness whose second phase lacks mindfulness (*dran med*), which should not be read as simple unconsciousness [18]. Clear light follows it and is described as lucid. The tradition further distinguishes the clear light of dying from that attained through meditation [18].

Claim 4 is the most demanding. If precision simply falls during dissolution, a lucid state cannot follow a dull one.

I formalise these claims in a hierarchical categorical active-inference model, with a precision update obtained from a mixture likelihood and a Gamma hyperprior under two stated assumptions. I then ask which of the four claims follow, under which assumptions, and where the model disagrees with both the tradition and the neuroscience. One claim is only partly reproduced. I report lesion analyses, stochastic simulations and a broad parameter sweep, and I state predictions that could falsify the account.

# Model

The full derivation is given in Methods. Each of seven levels holds a categorical belief over $q_i$ alternatives, written as a mixture of a committed component and a uniform component, $Q_i(s)=x_i\,\delta_{s,s^*}+(1-x_i)/q_i$. The responsibility $x_i\in[0,1]$ is the commitment of level $i$: at $x_i=0$ the belief is flat and predicts nothing; at $x_i=1$ it is fully determinate. Levels are coupled to their neighbours. Given a global precision $g$, the free energy is

$$
F=\sum_i\Big[-\frac{gJ_iJ_s}{2}\frac{1+(q_i-1)x_i^2}{q_i}-S(x_i;q_i)\Big]-g\sum_i\kappa\,x_ix_{i+1}-g\sum_i h_ix_i+\sum_i w_ix_i\Big[\tfrac12 g\omega d^2-\tfrac12\ln g+c\Big]
$$

The first term is a mean-field Potts energy [24]: $J_i$ is the recurrent self-support of level $i$, increasing up the hierarchy as in large-scale cortical models [25], and $J_s$ scales the precision of these priors, which attention may relax [26,27]. $S$ is the belief entropy, $\kappa$ the inter-level coupling and $h_i$ the sensory evidence. The last term is the expected negative log-likelihood of a bodily (homeostatic) mismatch $d^2$ under the committed component, relative to a broad null component ($c$). The mixture is treated as a latent indicator of commitment: a flat belief assigns the observation to the null component and pays no precision-weighted error (assumption A1).

Beliefs descend the free energy by natural gradient,

$$
\tau_i\dot u_i=gJ_iJ_sx_i+\frac{q_i}{q_i-1}\Big[g\big(\kappa(x_{i-1}+x_{i+1})+h_i\big)-w_i\big(\tfrac12 g\omega d^2-\tfrac12\ln g+c\big)\Big]-u_i,
$$

where $u_i=\ln(p_i/r_i)$ is the posterior log-odds of the winning category. Time constants $\tau_i$ increase up the hierarchy (60–300 ms), an order of magnitude taken from cortical estimates [25,28]. Precision has a Gamma hyperprior with shape $a$ and rate $b$. Its posterior mean, tracked with time constant $\tau_g$, is

$$
g^*=\frac{a+\tfrac12 W}{b+\tfrac12\omega d^2 W},\qquad W=\sum_i w_ix_i .
$$

Internal coherence does not appear in $g^*$. Precision is inferred only from how well committed beliefs predict the data, and the inferred value is then applied as a common gain to all messages (assumption A2). This is a modelling choice with substantial consequences, discussed in Methods. I use three observables:

- **Commitment** $A=\overline{x_i}$, a proxy for content-specific population activity and, following [3], for gamma power.
- **Emptiness** $\varepsilon=\sum_iS_i/\sum_i\ln q_i$, the normalised posterior entropy.
- **Readiness** $R=N^{-1}\mathbf 1^{\top}(I-M)^{-1}g\mathbf v$, with $v_i=q_i/(q_i-1)$, the change in log-odds per unit of sensory evidence (nat), where $M$ is the closed-loop gain matrix. On an isolated flat level, $R=\frac{q}{q-1}\,g/(1-gJ/q)$. A level clipped at $x_i=0$ contributes no loop gain. Its $u_i$ is then a sub-threshold drive: evidence moves $u_i$, but the posterior stays flat until $u_i>0$, so at clipped levels $R$ measures drive and not a change in the posterior.

I also compute stochastic interaction at the minimum-information bipartition as a proxy for integrated information [29], which is not the $\Phi$ of IIT 4.0 [4], and simulated responses to two kinds of pulse.

Levels are labelled L1–L7. L1–L4 are mapped to the four "elements" with their sense faculties, and L5–L7 to the three classes of conceptuality (Table 1). This mapping is an input. So are the alphabet sizes of L5–L7, which take the traditional numbers 33, 40 and 7 as a heuristic that is relaxed in the parameter sweep. The sensory alphabet size is free.
