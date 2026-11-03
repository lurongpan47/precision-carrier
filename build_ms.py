# -*- coding: utf-8 -*-
"""Fill computed values into the manuscript, then build Word (pandoc) and PDF (XeLaTeX)."""
import json, re, subprocess, sys
import numpy as np
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
import wmp as m

R = json.load(open('results.json'))
A = np.load('arrays.npz')
P = m.Params()


def f(x, n=2):
    return f'{x:.{n}f}'


def pct(x):
    return f'{100 * x:.0f}%'


def sci(x):
    if x <= 0:
        return '0'
    a, e = f'{x:.1e}'.split('e')
    return f'${a}\\times10^{{{int(e)}}}$'


def table(head, rows, widths=None):
    widths = widths or [1] * len(head)
    s = sum(widths)
    dashes = ['-' * max(3, int(round(60 * w / s))) for w in widths]
    out = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(':' + d for d in dashes) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    return '\n'.join(out)


T = {}
dy, tl = R['dying'], R['dying_tilted']
st = dy['stage_t']
S = {s['key']: s for s in R['states']}
l, r_ = R['ramp']['loss'], R['ramp']['ret']
sw = R['sweep']
T.update(lyap_viol='no' if R['lyapunov']['violations'] == 0 else str(R['lyapunov']['violations']), mc_N=R['mc']['N'], mc_down=f(R['mc']['down']), mc_up=f(R['mc']['up']), mf_lower=f(R['mc']['mf_lower']))
T.update(sw_complete=sw['complete'], sw_tau=f(sw['tau_pred_median']), sw_last=pct(sw['last_equals_pred']), sw_canon=pct(sw['canonical_overall']), sw_canon_pred=pct(sw['canonical_given_pred']), sw_npred=sw['n_pred_mono'])
T['q0_min'] = str(R['q_pred_min'])
T.update(eta0_span=f(R['eta0_stage_t'][5] - R['eta0_stage_t'][0], 0), base_span=f(st[5] - st[0], 0), noise_1=pct(R['noise_paths'][-1]['full_order']), sw_fb_pred=pct(sw['first_back_has_pred']), sw_valid=sw['valid'], sw_n_gt2=sw['n_rho_gt2'])
noise_ok = [d for d in R['noise_paths'] if d['full_order'] == 1]
T['noise_tmax'] = f"{noise_ok[-1]['T']:g}"
a_ = R['alphabet']
T.update(s33=f(a_['s']['33']), s40=f(a_['s']['40']), s7=f(a_['s']['7']), req_ratio=f(a_['req'], 3), J65=f(a_['J65'], 3), g5=f(l[4]), g6=f(l[5]), g7=f(l[6]), r7=f(r_[6]), r1=f(r_[0]))
T.update(sw_retmono_can=pct(sw['ret_mono_given_canonical']), sw_noreturn=sw['no_return'])
la = R['loop_area']
T.update(area_fast=f(la[0]['area']), area_slow=f(la[-1]['area']), area_inf=f(R['loop_fit']['a_inf']))
T.update(g_min=f(dy['g_min']), g_max=f(dy['g_max']), reb_u=f(dy['rebound']), R_nadir=f(dy['R_nadir']), R_after=f(dy['R_after']), reb_t=f(tl['rebound']), Rt_nadir=f(tl['R_nadir']), Rt_after=f(tl['R_after']), R_wake=f(dy['R_wake']))
T.update(sw_reb_gt=pct(sw['rebound_gt1_given_rho_gt1']), sw_n_gt=sw['n_rho_gt1'], sw_reb_lt=pct(sw['rebound_gt1_given_rho_lt1']), sw_n_lt=sw['n_rho_lt1'], sw_spear=f(sw['spearman_reb_rho']))
T.update(ratio_hold_sleep=f(S['medcl']['R'] / S['sleep']['R'], 1), ratio_hold_dy=f(S['medcl']['R'] / S['dycl']['R'], 1))
T['si_ratio'] = f"{float(f'{S['medcl']['si'] / S['sleep']['si']:.1g}'):g}"
T.update(amp_sleep=f(S['sleep']['R'] / S['sleep']['g']), amp_hold=f(S['medcl']['R'] / S['medcl']['g']), amp_crit=f(S['medcrit']['R'] / S['medcrit']['g']))
PT = {(d['key'], d['site'], d['kind']): d for d in R['perturb']}
def ptv(k, s, kd='evidence', fld='auc_total'):
    return PT[(k, s, kd)][fld]
T.update(pt_hold_sleep=f(ptv('medcl', 'L7') / ptv('sleep', 'L7'), 1), pt_hold_lv=ptv('medcl', 'L7', fld='levels'), pt_sleep_lv=ptv('sleep', 'L7', fld='levels'),
         ptd_hold_sleep=f(ptv('medcl', 'L7', 'direct') / ptv('sleep', 'L7', 'direct'), 1), ptd4_hold_sleep=f(ptv('medcl', 'L4', 'direct') / ptv('sleep', 'L4', 'direct'), 1),
         ptd_ns_hold=f(ptv('medcl', 'L7', 'direct', 'auc_nonsite'), 3), ptd_ns_sleep=f(ptv('sleep', 'L7', 'direct', 'auc_nonsite'), 3), pt_nadir=f(ptv('nadir', 'L7'), 3), pt_dycl=f(ptv('dycl', 'L7'), 3))
T.update(g_ratio=f(S['medcl']['g'] / S['sleep']['g'], 1), R_entry=f(dy['R_entry']), x7_nadir=f(S['nadir']['x7']), local_reb=', '.join(f(v) for v in dy['local_rebound']))
T.update(om_lo=f(min(R['omitted_U_at_spinodal']), 1), om_hi=f(max(R['omitted_U_at_spinodal']), 1), den_lo=f(R['denominator_range'][0], 1), den_hi=f(R['denominator_range'][1], 1))
_les = R['lesions']
sw2 = R['sweep']
_gr = {(d['D'], d['lam']): d for d in R['R_ratio_grid']}
_rei = [d for d in R['reignite'] if d['key'] == 'dycl' and d['x7'] > .5]
_amp_e = [d['ratio'] for d in R['pulse_amp'] if d['kind'] == 'evidence']
_amp_d = [d['ratio'] for d in R['pulse_amp'] if d['kind'] == 'direct']
_k0 = [v for v in R['kappa0_c2']['ret'][:6]]
_Pm = m.Params()
assert sw2['fail_ret'] == sw2['no_return']
T['reb_pct'] = f"{round((dy['rebound'] - 1) * 100):d}%"
T.update(sw_ramp=sw2['ramp'], sw_retmono=pct(sw2['ret_mono']), sw_n_both=sw2['n_both_canonical'], sw_retstrict=pct(sw2['ret_strict']), sw_belowwake=pct(sw2['ret_below_wake_of_valid']),
         sw_failcost=sw2['fail_cost_pos'], sw_rr_gt1=pct(sw2['R_ratio_gt1']), sw_rr_n=sw2['n_R_ratio'], sw_rr_clip=pct(sw2['R_ratio_all_clipped']),
         g_until=f(dy['t_g_above_peak_until'], 0), g_below=f(dy['t_g_below_trough_from'], 0), st6=f(st[5], 0), st7=f(st[6], 0), Rp_min=f(dy['Rpost_min']),
         rr_d16l2=f(_gr[(16, 2)]['ratio']), rr_base=f(_gr[(8, 0)]['ratio']), bin_rho=f(_les[3]['rho_c']), net_zero=f(R['net_cost_zero_g']),
         qJ_min_sens=f(float((_Pm.q / _Pm.J)[:6].min()), 1), k0_lo=f(min(_k0), 1), k0_hi=f(max(_k0), 1), R_hold=f(S['medcl']['R']), R_crit=f(S['medcrit']['R']),
         amp_ev_lo=f(min(_amp_e), 0), amp_ev_hi=f(max(_amp_e), 0), amp_d_lo=f(min(_amp_d), 1), amp_d_hi=f(max(_amp_d), 1), sg_tot=f(R['sleep_g_cross']['total']), sg_ns=f(R['sleep_g_cross']['nonsite']),
         ptd_last=f(ptv('nadir', 'L7', 'direct')), ptd_dycl=f(ptv('dycl', 'L7', 'direct')), ptd_sleep=f(ptv('sleep', 'L7', 'direct')), gj7_dy=f(S['dycl']['g'] * _Pm.J[6], 1),
         ptx_dycl=f(ptv('dycl', 'L7', fld='auc_x'), 4), ptx_sleep=f(ptv('sleep', 'L7', fld='auc_x'), 4), ptx_hold=f(ptv('medcl', 'L7', fld='auc_x'), 3), rei_kick=f"{_rei[0]['kick']:g}" if _rei else 'NA',
         a1_u=f(R['A1_variant'][0]['rebound']), a1_t=f(R['A1_variant'][1]['rebound']))
T.update(w_full=f(_les[0]['width_mean']), w_bin=f(_les[3]['width_mean']))
T.update(g_med0=f(R['med']['g0']), g_medh=f(R['med']['g_hold']))
T.update(gjs=f(R['gjs_zero_field'], 3), js_limit=f(R['js_limit']))
dw = [d for d in R['dwell'] if 'mean' in d and d['censored'] == 0]
T.update(dw_a=f(dw[0]['mean'], 0), dw_ja=f(dw[0]['Js']), dw_b=f(dw[-1]['mean'], 0), dw_jb=f(dw[-1]['Js']))
T['dw_note'] = 'coefficient of variation ' + f(min(d['cv'] for d in dw)) + '–' + f(max(d['cv'] for d in dw)) + '; Kolmogorov–Smirnov $p\\geq$ ' + f(min(d['ks_p'] for d in dw))
T.update(infra_crit=f(S['medcrit']['infra'] / S['medcl']['infra'], 1), delta_crit=f(S['medcrit']['delta'] / S['medcl']['delta'], 1))

# Tables
CL = [('Earth into water', 'Form; eye', 'Mirage'), ('Water into fire', 'Feeling; ear', 'Smoke'), ('Fire into wind', 'Discrimination; nose', 'Fireflies'), ('Wind into consciousness', 'Formations; tongue, body', 'Flame'),
      ('White appearance', '33 conceptions (object-related)', 'Moonlit sky'), ('Red increase', '40 conceptions (subject-related)', 'Sunlit sky'), ('Black near-attainment', '7 conceptions', 'Darkness')]
T['table1'] = table(['Level', '$q$', 'Classical stage', 'Aggregate; faculty or class', 'Inner sign', 'Time lost', '$g$ at loss', '$g$ at return'],
                    [[f'L{i + 1}', int(P.q[i]), CL[i][0], CL[i][1], CL[i][2], f(st[i], 0), f(l[i]), f(r_[i])] for i in range(7)], [.5, .35, 1.3, 1.6, .75, .5, .5, .55])
rows = []
for d in R['lesions']:
    w = d.get('width_mean')
    rows.append([d['name'], 'Yes' if d['order_ok'] else 'No', f(d['rebound']) if d['rebound'] else '—', 'Yes' if d['ret_mono'] else 'No', f(w) if w is not None else 'n/a'])
T['table2'] = table(['Model', 'Canonical order on path', 'Rebound', 'Monotonic return', 'Mean hysteresis width'], rows, [2.6, .9, .6, .8, .9])
rows = []
for s in R['states']:
    ord_ = s['A'] > .9
    e_ = PT[(s['key'], 'L7', 'evidence')]
    d_ = PT[(s['key'], 'L7', 'direct')]
    rows.append([s['name'], f(s['g']), f(s['A'], 3), f(s['eps'], 3), f(s['R']), '—' if ord_ else ('0' if abs(s['si']) < 1e-12 else sci(s['si'])), f"{e_['auc_total']:.2f} ({e_['levels']})", f"{d_['auc_total']:.2f} ({d_['levels']})", f(s['delta'] / S['medcl']['delta']), f(s['infra'] / S['medcl']['infra'])])
T['table3'] = table(['State', '$g$', '$A$', '$\\varepsilon$', '$R$', 'SI (nat)', 'Evidence pulse', 'Direct pulse', 'Delta', '0.05–0.5 Hz'], rows, [2.2, .42, .48, .48, .42, .85, .85, .75, .45, .65])
rows = [[d['q'], f(d['lower'], 3), f(d['coex_formula'], 3), f(d['coex_numeric'], 3), d['upper']] for d in R['analytic']]
t2a = ': **Table S2a. Critical points of the single-level model.**\n\n' + table(['$q$', 'Lower critical $s(q)$', 'First-order point, formula', 'First-order point, numerical', 'Upper critical'], rows)
rows = [[d['bJ'], f(d['fd'], 4), f(d['analytic'], 4)] for d in R['chi_check']]
t2b = ': **Table S2b. Readiness on an isolated flat level ($q=7$).** Finite difference of an independently solved fixed point against the analytic expression.\n\n' + table(['$gJ$', 'Finite difference', 'Analytic, $\\frac{q}{q-1}g/(1-gJ/q)$'], rows)
t2c = f"Lyapunov check: {R['lyapunov']['violations']} increases of $F$ across {R['lyapunov']['trials']} random trajectories at fixed $g$. Finite-difference gradient of $F$ at relaxed interior levels: maximum {R['stationarity_max_grad']:.1e} across {R['stationarity_n']} evaluations."


def lst(v):
    return ', '.join('—' if x is None else f(x) for x in v)


rows = [[d['name'].replace('q_s', '$q_s$ ').replace('eta', '$\\eta$ ').replace('kappa', '$\\kappa$ ').replace(' =', '=').replace('=', ' = '), lst(d['loss']), lst(d['ret'])] for d in R['scan']]
t3 = ': **Table S3. Quasi-static thresholds under single-parameter changes** (L1 to L7; — means no return up to $g=3.3$).\n\n' + table(['Change', 'Loss', 'Return'], rows, [.7, 2.6, 2.6])
rows = [[f"{d['T']:g}", d['complete'], pct(d['full_order']), pct(d['last7']), f(d['kendall_median']), f(d['rebound_median']), f(d['rebound_q10']) + '–' + f(d['rebound_q90'])] for d in R['noise_paths']]
t4 = ': **Table S4. Failing-support path with noise** (200 trajectories at each noise level).\n\n' + table(['$\\sigma^2$', 'Complete', 'Full order', 'L7 last', 'Median Kendall $\\tau$', 'Median rebound', '10–90%'], rows)
rows = [[f(d['rho0']), f(d['g_start']), f(d['g_end'])] for d in R['rho_regimes']]
t5 = ': **Table S5. Precision on the prior-relaxation path against baseline residual $\\rho_0$** ($a/b=1.2$).\n\n' + table(['$\\rho_0$', '$g$ at waking', '$g$ at hold'], rows)
rows = [[f(d['Js']), f(d['bJ']), f(d['R']), f(d['loop']), f(d['dF'], 4) if d['dF'] > 0 else 'none', sci(d['si']), f(d['delta'] / S['medcl']['delta']), f(d['infra'] / S['medcl']['infra'])] for d in R['tradeoff']]
t6 = ': **Table S6. Flat branch at $g=1.2$.** Band power relative to $J_s=0.20$. "None": no contentful solution, so no barrier.\n\n' + table(['$J_s$', '$gJ_7J_s$', '$R$', 'Loop gain', 'Barrier $\\Delta F$', 'SI (nat)', 'Delta', '0.05–0.5 Hz'], rows)
rows = [[f(d['Js']), d['n'], d['censored'], f(d['mean'], 0) if 'mean' in d else '—', f(d['cv']) if 'cv' in d else '—', f(d['ks_p']) if 'ks_p' in d else '—', f(d['dF'], 4)] for d in R['dwell']]
t7 = ': **Table S7. Noisy escape from the flat branch** ($g=1.2$, $\\sigma^2=0.02$, 300 trials, ceiling 4000 time units).\n\n' + table(['$J_s$', 'Escaped', 'Censored', 'Mean time', 'CV', 'KS $p$', 'Barrier $\\Delta F$'], rows)
GL = [('Wind', 'རླུང་', 'rlung', 'prāṇa', 'Precision $g$'), ('Mind', 'སེམས་', 'sems', 'citta', 'Beliefs $q_i$, commitment $x_i$'), ('Habitual tendency', 'བག་ཆགས་', 'bag chags', 'vāsanā', 'Prior self-support $J_i$'),
      ('Central channel', 'དབུ་མ་', 'dbu ma', 'avadhūtī', 'Not modelled'), ('White appearance', 'སྣང་བ་', 'snang ba', '', 'Loss of L5'), ('Red increase', 'མཆེད་པ་', 'mched pa', '', 'Loss of L6'),
      ('Black near-attainment', 'ཉེར་ཐོབ་', 'nyer thob', '', 'Only L7 committed'), ('With / without mindfulness', 'དྲན་ཅན་ / དྲན་མེད་', 'dran can / dran med', '', 'Not separately modelled'), ('Clear light', 'འོད་གསལ་', "'od gsal", 'prabhāsvara', 'All flat after the last collapse'),
      ('Object- / subject-related conception', 'གཟུང་རྟོག་ / འཛིན་རྟོག་', "gzung rtog / 'dzin rtog", '', 'L5, L6'), ('Forward / reverse sequence', 'ལུགས་འབྱུང་ / ལུགས་ལྡོག་', "lugs 'byung / lugs ldog", 'anuloma / pratiloma', 'Loss / return'), ('Post-mortem meditative state', 'ཐུགས་དམ་', 'thugs dam', '', 'None')]
t1 = ': **Table S1. Glossary.** Correspondences in the last column are modelling choices, not translations.\n\n' + table(['English', 'Tibetan', 'Wylie', 'Sanskrit', 'In the model'], [list(g) for g in GL], [1.6, 1.4, 1.2, .8, 1.3])
t8 = ''': **Table S8. Baseline parameters.**

| Quantity | Value | Basis |
|:----------|:------|:------------|
| Levels $N$ | 7 | Four elements, three conceptual classes |
| Alphabets $q_i$ | 33, 33, 33, 33, 33, 40, 7 | L5–L7 from the tradition (heuristic); sensory $q_s$ free |
| Self-support $J_i=J_0(1+\\eta\\ell_i)$ | $J_0=9$, $\\eta=0.68$, $\\ell_i\\in[0,1]$ uniform | $\\eta$ from [25]; $J_0$ free |
| Coupling $\\kappa$ | 0.6 | Free |
| Time constants $\\tau_i$ | 60–300 ms, geometric | Order of magnitude from [28] |
| Precision time constant $\\tau_g$ | 2 | Free |
| Hyperprior $a_0$, $b$ | 1.2, 1 | Free |
| Mismatch: $\\omega$, $e_0^2$, slope $D$ | 1, 1.25, 8 | Free; $\\rho_0=1.5$ |
| Null constant $c$ | $-1$ | Free |
| Likelihood weights $w_i$ | Uniform; $\\propto\\exp(\\lambda\\ell_i)$ in tilted runs | $\\lambda=2$ in tilted runs |
| Sensory evidence | $h_1=0.5$; symmetry-breaking $2\\times10^{-3}$ at all levels | Free |
| Path length $T$ | 300 | Arbitrary units |
'''
SUP = '\n\n'.join(['# Supplementary information', t1, t2a, t2b, t2c, t3, t4, t5, t6, t7, t8])

HEAD = '''---
title: "Inferred precision as the carrier of conscious content: a hierarchical active inference model of graded dissolution, contentless awareness and re-emergence"
---

**Lurong Pan**

Independent Scholar. Correspondence: lurongpan47@gmail.com

# Abstract

«ABSTRACT»

**Keywords:** consciousness; active inference; precision; meditation; minimal phenomenal experience; anaesthesia; dying; neural inertia; computational phenomenology

'''
ABSTRACT = '''Reports of awareness without content, and the ordered way experience dissolves and returns, challenge theories of consciousness. Indo-Tibetan Highest Yoga Tantra offers an explicit process model of these phenomena, and I use four of its structural claims as constraints on a quantitative model: content rides on a carrier, dissolution is ordered, emergence reverses it, and contentless states come in more than one kind. I formalise them in a seven-level categorical active-inference model in which precision is inferred from a responsibility-weighted likelihood and acts as a common gain. Withdrawing global precision produces graded collapse in an order predicted by one threshold (the last level was correctly predicted in «sw_last» of random parameter sets); agreement with the classical sequence itself is partly built in, because levels are labelled in the traditional order. Re-emergence runs as a coupled front with hysteresis that persists in the mean-field slow limit, although return failed to complete in many parameter sets. Under stated assumptions, abandoning content raises the precision target if and only if the normalised residual exceeds one. On the failing path, where it does, precision rebounds transiently after the last collapse (by «reb_pct» at baseline). Responsiveness does not recover with it: while commitment costs more than the null component, the flat posterior does not change under small evidence, so the classical claim of lucidity following darkness is only partly reproduced. Readiness in flat states scales with precision and recurrent amplification, so contentless states at similar commitment can differ, although the separation from deep sleep rests largely on an assumed precision. The model conflicts with persistent auditory responses in the dying, does not explain thought-first reports in meditation or feedback-specific disruption under anaesthesia, and does not capture sleep slow waves. Falsifiable predictions include wider spread of TMS-evoked responses in pure presence than in deep sleep.'''

md = HEAD + ''.join(open(p, encoding='utf8').read() + '\n\n' for p in ['ms_intro.md', 'ms_results.md', 'ms_disc.md', 'ms_methods_placeholder']) if False else None
intro = open('ms_intro.md', encoding='utf8').read()
res = open('ms_results.md', encoding='utf8').read()
disc = open('ms_disc.md', encoding='utf8').read()
meth = open('ms_methods.md', encoding='utf8').read()
# order: Intro, Model, Results, Discussion, Methods, Declarations, References, Supplement
dd, refs = disc.split('# Declarations')
md = HEAD.replace('«ABSTRACT»', ABSTRACT) + intro + '\n\n' + res + '\n\n' + dd + '\n\n' + meth + '\n\n# Declarations' + refs + '\n\n' + SUP + '\n'


def sub(mt):
    k = mt.group(1)
    if k not in T:
        raise KeyError(k)
    return str(T[k])


md = re.sub(r'«([^»]+)»', sub, md)
# a formula must not end in a relation, or Word renders a stray symbol: move the number inside the math
md = re.sub(r'\$([^$\n]*(?:=|>|<|\\geq|\\approx))\$ ([0-9][0-9.,]*[0-9]|[0-9])', r'$\1 \2$', md)
md = md.replace("'od gsal", "\\'od gsal").replace("'dzin", "\\'dzin").replace("'byung", "\\'byung")
open('manuscript.md', 'w', encoding='utf8').write(md)

# --- PDF via XeLaTeX ---
tex_md = re.sub(r'[ༀ-࿿]+', lambda mt: '\\tibt{' + mt.group(0) + '}', md)
tex_md = re.sub(r'^(: \*\*Table)', lambda mt: '\\FloatBarrier\n\n' + mt.group(1), tex_md, flags=re.M)
tex_md = tex_md.replace('\\FloatBarrier\n\n: **Table 1.', '\\clearpage\n\n: **Table 1.')
open('manuscript_tex.md', 'w', encoding='utf8').write(tex_md)
header = r'''
\usepackage{fontspec}
\setmainfont{Liberation Serif}
\setsansfont{Liberation Sans}
\newfontfamily\tibfont{Noto Serif Tibetan}[Script=Tibetan,Scale=0.95]
\newcommand{\tibt}[1]{{\tibfont #1}}
\usepackage{caption}
\captionsetup{labelformat=empty,font=small,justification=raggedright,singlelinecheck=false}
\usepackage{placeins}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\footnotesize\setlength{\tabcolsep}{3pt}}
\usepackage{setspace}
\setstretch{1.12}
\usepackage{titlesec}
\titleformat{\section}{\sffamily\large\bfseries}{}{0pt}{}
\titleformat{\subsection}{\sffamily\normalsize\bfseries}{}{0pt}{}
\usepackage{xurl}
\makeatletter\def\fps@figure{!tbp}\makeatother
\renewcommand{\topfraction}{.85}\renewcommand{\textfraction}{.12}\renewcommand{\floatpagefraction}{.7}
'''
open('header.tex', 'w').write(header)
subprocess.run(['pandoc', 'manuscript_tex.md', '-o', 'manuscript.pdf', '--pdf-engine=xelatex', '-H', 'header.tex', '-V', 'geometry:a4paper,margin=2.4cm', '-V', 'fontsize=10.5pt', '-V', 'colorlinks=true', '-V', 'linkcolor=black', '-V', 'urlcolor=blue!50!black', '--columns=40', '-f', 'markdown+tex_math_dollars+pipe_tables+table_captions+implicit_figures+raw_tex+smart'], check=True)

# --- Word ---
ref = 'reference.docx'
open(ref, 'wb').write(subprocess.run(['pandoc', '--print-default-data-file', 'reference.docx'], capture_output=True).stdout)
d = Document(ref)
s = d.sections[0]
s.page_width, s.page_height = Cm(21), Cm(29.7)
s.left_margin = s.right_margin = Cm(2.4)
s.top_margin, s.bottom_margin = Cm(2.3), Cm(2.2)
st_ = {x.name.lower(): x for x in d.styles}


def setf(style, latin, size=None, bold=None, color=None):
    rPr = style.element.get_or_add_rPr()
    rf = rPr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rPr.insert(0, rf)
    for a_k in list(rf.attrib):
        del rf.attrib[a_k]
    for k in ['ascii', 'hAnsi', 'cs', 'eastAsia']:
        rf.set(qn('w:' + k), latin)
    if size:
        style.font.size = Pt(size)
    if bold is not None:
        style.font.bold = bold
    if color is not None:
        style.font.color.rgb = color
    style.font.italic = False


for nm in ['normal', 'body text', 'first paragraph', 'compact']:
    setf(st_[nm], 'Times New Roman', 10.5)
for nm in ['body text', 'first paragraph']:
    st_[nm].paragraph_format.space_after = Pt(6)
    st_[nm].paragraph_format.line_spacing = 1.2
ink = RGBColor(0x11, 0x11, 0x11)
for nm, sz, sb in [('heading 1', 13, 14), ('heading 2', 11, 10)]:
    setf(st_[nm], 'Arial', sz, True, ink)
    st_[nm].paragraph_format.space_before = Pt(sb)
    st_[nm].paragraph_format.space_after = Pt(4)
    st_[nm].paragraph_format.keep_with_next = True
setf(st_['title'], 'Arial', 16, True, ink)
st_['title'].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
for nm in ['image caption', 'table caption']:
    if nm in st_:
        setf(st_[nm], 'Arial', 8.5, color=RGBColor(0x33, 0x33, 0x33))
        st_[nm].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
fp = s.footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
rn = fp.add_run()
for t_, txt in [('begin', None), (None, 'PAGE'), ('end', None)]:
    if t_:
        el = OxmlElement('w:fldChar')
        el.set(qn('w:fldCharType'), t_)
    else:
        el = OxmlElement('w:instrText')
        el.text = txt
    rn._r.append(el)
d.save(ref)
OUT = 'manuscript.docx'
docx_md = open('manuscript.md', encoding='utf8').read().replace('^*', '^{\\star}')
open('manuscript_docx.md', 'w', encoding='utf8').write(docx_md)
subprocess.run(['pandoc', 'manuscript_docx.md', '-o', OUT, '--reference-doc=' + ref, '--columns=40', '-f', 'markdown+tex_math_dollars+pipe_tables+table_captions+implicit_figures+smart'], check=True)
d = Document(OUT)
total = 21 - 4.8
for t_ in d.tables:
    tblPr = t_._tbl.tblPr
    for old in tblPr.findall(qn('w:tblBorders')):
        tblPr.remove(old)
    b = OxmlElement('w:tblBorders')
    for side, sz, col in [('top', 8, '111111'), ('bottom', 8, '111111'), ('insideH', 4, 'D9D8D3')]:
        e = OxmlElement('w:' + side)
        e.set(qn('w:val'), 'single')
        e.set(qn('w:sz'), str(sz))
        e.set(qn('w:space'), '0')
        e.set(qn('w:color'), col)
        b.append(e)
    tblPr.append(b)
    t_.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(t_.rows):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(OxmlElement('w:cantSplit'))
        if ri == 0:
            trPr.append(OxmlElement('w:tblHeader'))
        for c in row.cells:
            for p in c.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.1
                for r in p.runs:
                    r.font.size = Pt(8.5)
                    if ri == 0:
                        r.font.bold = True
                    if any('ༀ' <= ch <= '࿿' for ch in r.text):
                        rPr = r._r.get_or_add_rPr()
                        rf = rPr.find(qn('w:rFonts'))
                        if rf is None:
                            rf = OxmlElement('w:rFonts')
                            rPr.insert(0, rf)
                        for a_k in ['ascii', 'hAnsi', 'cs', 'eastAsia']:
                            rf.set(qn('w:' + a_k), 'Noto Serif Tibetan')
                        r.font.size = Pt(10)
maxw = Cm(total)
for s_ in d.inline_shapes:
    if s_.width > maxw:
        ratio = maxw / s_.width
        s_.height = int(s_.height * ratio)
        s_.width = maxw
for p in d.paragraphs:
    if p._p.xpath('.//w:drawing') and not p.text.strip():
        p.paragraph_format.keep_with_next = True
d.save(OUT)
# schema clean-up of pandoc's output so that the file validates against OOXML
import zipfile, shutil, lxml.etree as ET
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
ORDER = {
    'pPr': 'pStyle keepNext keepLines pageBreakBefore framePr widowControl numPr suppressLineNumbers pBdr shd tabs suppressAutoHyphens kinsoku wordWrap overflowPunct topLinePunct autoSpaceDE autoSpaceDN bidi adjustRightInd snapToGrid spacing ind contextualSpacing mirrorIndents suppressOverlap jc textDirection textAlignment textboxTightWrap outlineLvl divId cnfStyle rPr sectPr pPrChange'.split(),
    'tblPr': 'tblStyle tblpPr tblOverlap bidiVisual tblStyleRowBandSize tblStyleColBandSize tblW jc tblCellSpacing tblInd tblBorders shd tblLayout tblCellMar tblLook tblCaption tblDescription tblPrChange'.split(),
    'tcPr': 'cnfStyle tcW gridSpan hMerge vMerge tcBorders shd noWrap tcMar textDirection tcFitText vAlign hideMark'.split(),
    'settings': 'view zoom proofState stylePaneFormatFilter doNotTrackMoves defaultTabStop drawingGridHorizontalSpacing drawingGridVerticalSpacing displayHorizontalDrawingGridEvery displayVerticalDrawingGridEvery characterSpacingControl savePreviewPicture footnotePr endnotePr compat rsids mathPr themeFontLang clrSchemeMapping decimalSymbol listSeparator'.split(),
    'style': 'name aliases basedOn next link autoRedefine hidden uiPriority semiHidden unhideWhenUsed qFormat locked personal personalCompose personalReply rsid pPr rPr tblPr trPr tcPr tblStylePr'.split(),
}
tmp = OUT + '.tmp'
with zipfile.ZipFile(OUT) as zi, zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zo:
    for it in zi.infolist():
        data = zi.read(it.filename)
        if it.filename in ('word/document.xml', 'word/numbering.xml', 'word/settings.xml', 'word/styles.xml'):
            root = ET.fromstring(data)
            for e in root.iter('{%s}pgMar' % W):
                for k, v in [('header', '708'), ('footer', '708'), ('gutter', '0')]:
                    if e.get('{%s}%s' % (W, k)) is None:
                        e.set('{%s}%s' % (W, k), v)
            for e in root.iter('{%s}nsid' % W):
                e.set('{%s}val' % W, e.get('{%s}val' % W).rjust(8, '0'))
            for e in root.iter('{%s}zoom' % W):
                if e.get('{%s}percent' % W) is None:
                    e.set('{%s}percent' % W, '100')
            for dp in root.iter('{%s}dPr' % M):
                sep = dp.find('{%s}sepChr' % M)
                beg = dp.find('{%s}begChr' % M)
                if sep is not None and beg is not None:
                    dp.remove(sep)
                    beg.addnext(sep)
            for parent, order in ORDER.items():
                for e in root.iter('{%s}%s' % (W, parent)):
                    kids = list(e)
                    kids.sort(key=lambda c: order.index(ET.QName(c).localname) if ET.QName(c).localname in order else len(order))
                    for c in kids:
                        e.remove(c)
                    e.extend(kids)
            for e in root.iter('{%s}rPr' % W):
                if e.text and e.text.strip():
                    e.text = None
                for c in e:
                    if c.tail and c.tail.strip():
                        c.tail = None
            for rp in root.iter('{%s}rPr' % M):
                if rp.find('{%s}nor' % M) is not None and rp.find('{%s}sty' % M) is not None:
                    rp.remove(rp.find('{%s}sty' % M))
            if it.filename == 'word/settings.xml':
                for tag_ in ['embedSystemFonts', 'embedTrueTypeFonts', 'saveSubsetFonts']:
                    for e_ in root.findall('{%s}%s' % (W, tag_)):
                        root.remove(e_)
                z = root.find('{%s}zoom' % W)
                if z is not None:
                    root.remove(z)
                    root.insert(0, z)
            data = ET.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
        zo.writestr(it, data)
shutil.move(tmp, OUT)
print('built')
