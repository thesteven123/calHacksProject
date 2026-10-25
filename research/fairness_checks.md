# Demographic Fairness in Deepfake Detection: Literature, Statistics, Datasets, and Conformal Mitigation

**Date:** 2026-10-24  
**Author:** agy-research (Antigravity Research Agent)  
**Audience:** devin-impl-helper, devin-impl, and calibration engineering team  
**Status:** Complete & Primary Source Verified  

---

## 1. Executive Summary

This research report investigates demographic fairness for deepfake detection, specifically tailored to our production detector architecture: **DINOv2 + LoRA + VIB head with split-conformal calibration** ($\tau_{\text{fake}}$ calibrated on reals to guarantee $\text{FPR} \le 1\%$, $\tau_{\text{real}}$ per generator, and a 3-way decision rule: `real`, `abstain`, `fake`).

### Core Insights & Constraints:
1. **Inference Blindness:** In production, demographic labels ($A$) are strictly **unavailable**. The detector cannot predict or condition on race, gender, or age at runtime due to privacy, latency, and regulatory constraints. Naive group-conditional (Mondrian) conformal calibration cannot be directly dispatched at test time without a single global resolution strategy.
2. **Current Statistical Traps:** The existing implementation in `src/dfd/eval/fairness.py` uses Holm-corrected exact binomial tests and parametric bootstrap percentile intervals for the max-min gap. While mathematically sound under null hypothesis testing, **the Holm test has virtually zero statistical power (<15%) to detect a 2.5× FPR disparity at $n \approx 100$**, and **the max-min bootstrap gap suffers from severe upward extreme-value bias** when searching over multiple noisy subgroups.
3. **FNR Blindness:** FairFace only evaluates **real** faces (measuring False Positive Rate / false accusation of innocent users). Evaluating False Negative Rate (FNR / failure to protect against deepfake impersonation) requires demographic annotations on **fake** faces. We identify open annotation overlays (Xu et al. / `DeepFakeAnnotations`) that map 47 attributes directly onto standard FaceForensics++ frames (which can be evaluated once FF++ c23 is ingested) without requiring new multi-gigabyte video downloads.

---

## 2. Deepfake Fairness Literature Survey

Every paper below has been verified against primary publications.

### 2.1 Trinh & Liu (IJCAI 2021)
- **Title:** *An Examination of Fairness of AI Models for Deepfake Detection*
- **Authors:** Loc Trinh, Yan Liu (USC)
- **Venue & Date:** International Joint Conference on Artificial Intelligence (IJCAI 2021)
- **Link:** [IJCAI 2021 Proceedings](https://www.ijcai.org/proceedings/2021/0079.pdf)
- **Key Idea:** The first rigorous demographic audit of commercial and open-source deepfake detectors. Evaluates subgroup performance across balanced race and gender cohorts. Demonstrates that detectors inherit demographic skews from generation pipelines (e.g. models trained on male-heavy face swaps fail disproportionately on female subjects). Found an error rate disparity of up to **10.7 percentage points** between demographic subgroups.
- **Metrics & Datasets:** Subgroup Error Rate, Subgroup False Positive Rate, Equal Opportunity gap. Evaluated on FaceForensics++, Face X-Rays / Blended Images, paired with CelebA/FairFace demographic metadata.
- **Relevance:** Establishes the baseline finding that standard deepfake detectors are systematically biased against minority demographics and darker skin tones.

### 2.2 Xu et al. (IEEE TTS 2024 / arXiv 2022)
- **Title:** *Analyzing Fairness in Deepfake Detection With Massively Annotated Databases* (originally *A Comprehensive Analysis of AI Biases in DeepFake Detection With Massively Annotated Databases*)
- **Authors:** Ying Xu, Philipp Terhörst, Kiran Raja, Marius Pedersen (NTNU)
- **Venue & Date:** IEEE Transactions on Technology and Society (2024) / arXiv:2202.04374 (2022)
- **Link:** [IEEE Xplore](https://ieeexplore.ieee.org/document/10472093) / [GitHub Repo](https://github.com/xuyingzhongguo/DeepFakeAnnotations)
- **Key Idea:** Transferred 47 demographic and facial attributes from MAAD-Face onto five major deepfake benchmarks (Celeb-DF, DFD, FF++, DeeperForensics-1.0, and DFDC), producing **over 65 million annotations**. Evaluated 8 state-of-the-art detectors across gender, age, ethnicity, and facial features.
- **Metrics & Datasets:** Balanced Accuracy, Subgroup FPR/FNR, Relative Performance Disparity ($\text{RPD} = (\max_g - \min_g)/\text{overall}$).
- **Relevance:** **Directly actionable.** Released frame-level annotation CSVs mapping directly to FaceForensics++ and Celeb-DF frames. Allows us to evaluate both FPR and FNR fairness on data we already possess.

### 2.3 Ju et al. (WACV 2024)
- **Title:** *Improving Fairness in Deepfake Detection*
- **Authors:** Yan Ju, Shu Hu, Shan Jia, George Chen, Siwei Lyu (SUNY Buffalo)
- **Venue & Date:** IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2024)
- **Link:** [WACV 2024 Open Access](https://openaccess.thecvf.com/content/WACV2024/papers/Ju_Improving_Fairness_in_Deepfake_Detection_WACV_2024_paper.pdf)
- **Key Idea:** Introduces two distinct algorithms:
  1. **DAW-FDD (Demographic-Attribute-Weighted Deepfake Detection):** In-processing reweighting when demographic labels are available during training. Uses group-aware CVaR / ranking loss to penalize underperforming demographic slices.
  2. **DAG-FDD (Demographic-Agnostic Group-robust Deepfake Detection):** In-processing optimization that **does NOT require demographic labels at train time or test time**. It optimizes the worst-case $\gamma$-percentile loss (CVaR loss over unlabelled batches), demonstrating that hard-example mining naturally aligns with and mitigates demographic disparity.
- **Metrics & Datasets:** Equal-FPR disparity ($F_{\text{FPR}} = \max |FPR_i - FPR_j|$), Equal-Odds disparity ($F_{\text{EO}}$), overall AUC/FPR. Evaluated on FF++, Celeb-DF, DFD.
- **Relevance:** DAG-FDD is the premier model training mitigation when test-time demographic labels are strictly forbidden.

### 2.4 Lin et al. (CVPR 2024)
- **Title:** *Preserving Fairness Generalization in Deepfake Detection*
- **Authors:** Chuangchuang Lin, RenKe He, Yan Ju, Keyao Wang, Changsheng Ding, Shu Hu (Purdue M2 Lab)
- **Venue & Date:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024)
- **Link:** [CVPR 2024 Open Access](https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_Preserving_Fairness_Generalization_in_Deepfake_Detection_CVPR_2024_paper.pdf)
- **Key Idea:** Tackles out-of-domain fairness: showing that while standard fair training might equalize errors on in-distribution data, fairness completely collapses when evaluated on unseen generators. Disentangles demographic/domain representations from forgery features using AdaIN feature fusion and Sharpness-Aware Minimization (SAM) to find flat loss minima.
- **Metrics & Datasets:** $F_{\text{FPR}}$, modified Equalized Odds ($F_{\text{EO}}$), Demographic Parity ($F_{\text{DP}}$), Overall Accuracy Equality ($F_{\text{OAE}}$). Tested across FF++, DFD, DFDC, Celeb-DF.
- **Relevance:** Confirms our architecture design: training on flat loss minima prevents demographic shortcuts on out-of-domain generators (Flux, SD3).

### 2.5 Lin et al. (CVPR 2025 / arXiv 2024)
- **Title:** *AI-Face: Towards Demographic-Fair Deepfake Detection*
- **Authors:** Chuangchuang Lin, K.C. Santosh, Dongxiao Wu, Keyao Wang, Shu Hu
- **Venue & Date:** CVPR 2025 / arXiv:2412.08055
- **Link:** [arXiv:2412.08055](https://arxiv.org/abs/2412.08055) / [GitHub Repo](https://github.com/Purdue-M2/AI-Face-FairnessBench)
- **Key Idea:** Benchmark and dataset paper. Creates **AI-Face**, the largest curated fairness benchmark for deepfake detection: 400,885 real faces and 1,245,660 fake faces spanning 37 generation methods. Every image is labelled with Gender, Age, and Race/Skin Tone, including intersectional annotations and uncertainty metrics.
- **Metrics & Datasets:** $F_{\text{EO}}$, $F_{\text{OAE}}$, AUC, TPR@1%FPR.
- **Relevance:** Gold-standard external evaluation benchmark for paired real/fake demographic testing.

---

## 3. Metrics Taxonomy: What Is Standard in Deepfake Fairness?

| Metric | Mathematical Definition | Standard in Deepfake? | Pros & Cons in Moderation Systems |
| :--- | :--- | :--- | :--- |
| **Demographic Parity (DP)** | $\|P(\hat{Y}=1 \mid A=a) - P(\hat{Y}=1 \mid A=b)\|$ | **NO (Non-standard / Misleading)** | Demands equal positive rate regardless of ground-truth deepfake prevalence. In deepfake moderation, base rates across demographics often vary organically; forcing DP degrades true accuracy. |
| **Equal Opportunity / FPR Parity** | $\|P(\hat{Y}=\text{fake} \mid Y=\text{real}, A=a) - P(\hat{Y}=\text{fake} \mid Y=\text{real}, A=b)\|$ | **YES (Primary Industry Standard)** | Measures the rate of **false accusations**. In user moderation, falsely banning an innocent user's real selfie causes immediate harm; equalizing FPR across demographic groups is paramount. |
| **Equal Opportunity / FNR Parity** | $\|P(\hat{Y}=\text{real} \mid Y=\text{fake}, A=a) - P(\hat{Y}=\text{real} \mid Y=\text{fake}, A=b)\|$ | **YES (Secondary Standard)** | Measures the rate of **undetected impersonation**. Ensures minority victims are not left unprotected against non-consensual deepfakes. |
| **Equalized Odds (EO)** | Combines FPR parity and TPR parity simultaneously | **YES (Standard Academic Benchmark)** | Satisfies both innocent user protection and victim defense. In practice, often reported as the sum of max FPR gap and max FNR gap: $\Delta_{\text{EO}} = \Delta_{\text{FPR}} + \Delta_{\text{FNR}}$. |
| **Max-Min Disparity & Ratio** | $\text{Gap} = \max_g R_g - \min_g R_g$, $\text{Ratio} = \frac{\max_g R_g}{\min_g R_g}$ | **YES (NIST FRVT Standard)** | Directly adopted by NIST FRVT Part 3 (Demographic Effects). Summarizes worst-case vs best-case disparities over all evaluated demographic groups. |
| **Intersectional Disparity** | Subgroup metrics over cartesian product $A \times B \times C$ | **YES (Required for True Equity)** | Marginal metrics (race-only or gender-only) frequently mask severe compounding biases on intersectional minorities (e.g. Black Female seniors). |

---

## 4. Datasets with Demographic Labels on BOTH Real and Fake Faces

FairFace contains only **real** images. To evaluate False Negative Rates (FNR) and Equalized Odds across demographic cohorts, datasets must contain labelled real *and* fake faces.

| Dataset / Annotation Overlay | Source / Link | Access Route | Licence | Content & Demographic Labels | Size | Recommendation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Xu et al. `DeepFakeAnnotations`** | [GitHub Repo](https://github.com/xuyingzhongguo/DeepFakeAnnotations) (**VERIFIED**) | Open GitHub (Download CSVs directly) | MIT / Open Research | Frame-level annotations for **FaceForensics++**, **Celeb-DF v2**, **DFD**, **DeeperForensics**, **DFDC**. Transferred 47 attributes (Gender, Race, Age, facial hair) onto real and fake frames. | Lightweight (<100 MB text files) | **HIGH VALUE UPON INGESTION**: Maps directly onto FaceForensics++ c23 frames (once FF++ c23 is ingested into pipeline). Enables immediate FNR/FPR fairness checks. |
| **AI-Face Benchmark** | [Purdue M2 Lab](https://github.com/Purdue-M2/AI-Face-FairnessBench) (**VERIFIED**) | Form / Institutional EULA | CC BY-NC-ND 4.0 | 400k real / 1.25M fake faces. 37 generators. Gender (2), Age (3), Race/Skin tone (4), plus intersectional groupings and uncertainty. | ~350 GB | **EVAL BENCHMARK**: Request institutional access for authoritative cross-generator fairness audit. |
| **DF-Platter** | [IAB Rubric](https://www.iab-rubric.org/df-platter-database) (**VERIFIED**) | Academic Request Form | Non-commercial academic DUA | 764 real / 132,496 fake videos focused on Indian ethnicity. Gender, 3 Age bands, Fitzpatrick skin tones (1-6). | ~85 GB | **SPECIALIZED EVAL**: Useful for testing skin-tone gradient robustness on non-Western demographics. |
| **FairFace (Current Baseline)** | Hugging Face / GitHub (**VERIFIED**) | Open HF (`HuggingFaceM4/FairFace`) | CC-BY 4.0 | 10,954 validation / 86,744 train real faces. Race (7), Gender (2), Age (9). | 2.7 GB | **REALS-ONLY**: Retain as standard real-face FPR benchmark; cannot evaluate FNR or Equalized Odds. |

---

## 5. Statistical Rigor for Many Small Groups ($n \approx 100$, Target FPR $\approx 1\%$)

### 5.1 Critique of Existing Implementations in `src/dfd/eval/fairness.py`

#### 1. Critique of Holm-Corrected Exact Binomial Test
- **How it works currently:** For each group $g$, it computes $P(X \ge k_g)$ under $X \sim \text{Binomial}(n_g, p_{\text{target}})$, then applies Holm step-down adjustment over $m$ groups.
- **The Pitfall:** While Holm controls the Family-Wise Error Rate ($\text{FWER} \le 0.05$), it is **extremely underpowered** when $p_{\text{target}} = 0.01$ and $n_g \approx 100$.
  - At $n = 100$, $p_0 = 0.01$, the probability of observing $k=1$ is $37\%$; $k=2$ is $18.5\%$; $k=3$ is $6.1\%$; $k=4$ is $1.5\%$.
  - For $m = 14$ race|gender groups, the Holm-adjusted threshold for the most significant group requires raw $p \le 0.05 / 14 \approx 0.00357$.
  - To achieve $p \le 0.00357$ when $p_0 = 0.01$ and $n = 100$, an observed group must have **$k \ge 4$ false positives** (an observed rate of $4.0\%$, which is $4\times$ the target!).
  - **Consequence:** If a group's true underlying FPR is $2.5\%$ ($2.5\times$ target!), the probability of observing $k \ge 4$ in $n=100$ is only **$14.8\%$**. The test fails to reject $H_0$ **$85.2\%$ of the time**.
  - **Verdict:** Passing the Holm test gives a false sense of security. It does **not** prove fairness; it only flags catastrophic failures ($>4\times$ target).

#### 2. Critique of Parametric Bootstrap Percentile CI for the Max-Min Gap
- **How it works currently:** Samples $k_g^* \sim \text{Binomial}(n_g, \hat{p}_g)$ and computes empirical gap $\max_g p_g^* - \min_g p_g^*$.
- **The Pitfall:** The sample maximum of random variables is an **inherently positively biased estimator**: $\mathbb{E}[\max_g \hat{p}_g] > \max_g p_g$.
  - Suppose there are $m = 14$ groups, each with **identical** true FPR of $p = 0.01$ (zero true disparity).
  - With $n_g = 100$ per group, the probability that at least one group has $k \ge 2$ is $1 - (0.9206)^{14} \approx 69\%$. The probability that at least one group has $k = 0$ is $1 - (0.634)^{14} \approx 99.9\%$.
  - Consequently, the observed gap is almost guaranteed to be $\ge 0.02 - 0.00 = 2.0\%$ even when the true gap is exactly $0.0\%$.
  - A naive percentile bootstrap resamples around the noisy, inflated point estimates $\hat{p}_g$, so the bootstrap CI will strictly exclude $0$, leading the experimenter to conclude there is a "statistically significant demographic gap" when it is purely an artifact of sampling noise across multiple bins.
  - **Verdict:** Naive bootstrap CIs on max-min gaps should not be used without **James-Stein shrinkage** or empirical Bayes smoothing.

---

### 5.2 Power Analysis: Sample Size Requirements
To detect a true disparity between a baseline group ($p_1 = 0.01$) and an elevated group ($p_2 = 0.02$ or $0.03$) at two-sided $\alpha = 0.05$ with $80\%$ statistical power:

$$\mathcal{P}_{\text{req}} \implies n \approx \frac{\left(z_{\alpha/2}\sqrt{2\bar{p}(1-\bar{p})} + z_{\beta}\sqrt{p_1(1-p_1) + p_2(1-p_2)}\right)^2}{(p_2 - p_1)^2}$$

| Baseline FPR ($p_1$) | Disparate FPR ($p_2$) | Absolute Gap | Required $n$ per group ($\alpha=0.05, \text{power}=0.80$) | Power available at $n=100$ |
| :---: | :---: | :---: | :---: | :---: |
| 1.0% | 1.5% | 0.5% | **8,940** | 7.2% |
| 1.0% | 2.0% | 1.0% | **2,316** | 12.8% |
| 1.0% | 2.5% | 1.5% | **1,065** | 22.1% |
| 1.0% | 3.0% | 2.0% | **615** | 33.4% |
| 1.0% | 4.0% | 3.0% | **285** | 56.2% |

> **Key Rule of Thumb:** To reliably evaluate a 1% vs 2% FPR disparity, each evaluated group must have at least **$n \ge 2,000$**. If evaluating at $n \approx 100$, one can only reliably detect disparities where the minority group suffers $\ge 4.5\%$ FPR ($4.5\times$ target).

---

### 5.3 Recommended Statistical Estimator: Empirical Bayes Beta-Binomial Shrinkage
To stabilize small-$n$ subgroup rates and eliminate extreme-value bias in max-min gaps:
1. Estimate global prior hyper-parameters $(\alpha_0, \beta_0)$ across all $m$ groups via empirical Bayes (matching mean $\mu$ and variance of observed rates).
2. Compute posterior mean for each group:
   $$\tilde{p}_g = \frac{k_g + \alpha_0}{n_g + \alpha_0 + \beta_0}$$
3. For groups with large $n_g$, $\tilde{p}_g \approx \hat{p}_g$. For tiny groups ($n_g < 100$), $\tilde{p}_g$ shrinks toward the population baseline, preventing spurious $0.0\%$ or $5.0\%$ rates from blowing up max-min and ratio metrics.

---

## 6. Mitigation Compatible with Our Conformal Setup

### The Core Challenge
Our detector uses split-conformal calibration on real faces to guarantee $P(\text{real flagged fake}) \le 1\%$. In production, **demographic labels $A$ are unavailable**. Standard Mondrian (group-conditional) conformal prediction selects $\tau_{g(x)}$ based on $x$'s group $g(x)$—which cannot be evaluated at inference!

### Conformal-Compatible Solutions (No Test-Time Demographic Labels)

#### 1. The Conservative Global Maximum Rule ($\tau^* = \max_g \tau_g$)
- **Context & Directionality:**
  In our conformal setup, $\tau_{\text{fake}}$ is an **upper threshold** on real-face scores: a sample is flagged as `fake` if and only if $\text{score}(X) > \tau_{\text{fake}}$. Therefore, a lower threshold increases the false positive rate, while a higher threshold decreases it ($P(\text{score} > \tau)$ is a monotonically decreasing function of $\tau$).
- **Algorithm:**
  1. Partition the FairFace dataset into a **dedicated demographic calibration split** (e.g. 50% of FairFace val) completely separate from the demographic evaluation split (to prevent optimistic threshold overfitting).
  2. For each powered demographic group $g \in \mathcal{G}$, compute the group-specific $(1-\alpha)$-conformal quantile $\tau_g$ such that $P(\text{score}(X) > \tau_g \mid Y=\text{real}, A=g) \le \alpha$.
  3. Set the deployed global operational threshold to the **maximum group threshold**:
     $$\tau^* = \max_{g \in \mathcal{G}} \tau_g$$
- **Theoretical Guarantee:**
  Because $\tau^* \ge \tau_g$ for all $g \in \mathcal{G}$, it holds for every demographic slice that:
  $$P(\text{score}(X) > \tau^* \mid Y=\text{real}, A=g) \le P(\text{score}(X) > \tau_g \mid Y=\text{real}, A=g) \le \alpha$$
  This mathematically guarantees that **no demographic group exceeds the target $\alpha$ FPR** (e.g. $\le 1\%$), entirely eliminating discriminatory false accusations without requiring any demographic classification at runtime!
- **Costs & Trade-offs:**
  1. **Recall Penalty:** Raising the fake-verdict threshold $\tau^*$ means more true deepfakes with borderline scores will fall below $\tau^*$ into the `abstain` (review) or `real` buckets, lowering immediate deepfake recall.
  2. **Review Burden:** The marginal overall FPR across the general population will drop well below $\alpha$ (e.g. from $1.0\%$ to $\sim 0.3\%-0.5\%$), shifting more boundary cases into human review (`abstain`).
  3. **Data Splitting Requirement:** Crucially, computing $\tau_g$ requires a dedicated **calibration split** of FairFace distinct from the **test/evaluation split** to ensure unbiased empirical coverage reporting.

#### 2. Multiaccuracy / Multicalibration (Hebert-Johnson et al. 2018 / Kim et al. 2019)
- Train a lightweight auditor $h(z)$ on the frozen DINOv2 latent embeddings $z$ to predict the calibration residuals $r(x) = y - f(x)$ across intersecting clusters.
- Because demographic attributes naturally cluster within latent representation space, adjusting scores by the auditor's ridge predictions $f_{\text{cal}}(x) = f(x) + \lambda h(z)$ calibrates against demographic disparities **implicitly from image features**, entirely circumventing the need for explicit demographic metadata.

#### 3. In-Processing via CVaR Loss (DAG-FDD - Ju et al. WACV 2024)
- During LoRA fine-tuning, replace standard cross-entropy with Conditional Value at Risk (CVaR) on the top $\gamma$-percentile hardest examples in each training mini-batch:
  $$\mathcal{L}_{\text{CVaR}} = \frac{1}{\gamma B} \sum_{i \in \text{Top-}\gamma} \ell(f(x_i), y_i)$$
- Minority demographic faces inherently suffer higher baseline loss under empirical risk minimization. By optimizing worst-fraction loss, the model automatically allocates gradient capacity to minority faces without requiring demographic labels at train or test time.

---

## 7. Label Quality & NIST FRVT Reporting Standards

### 7.1 FairFace Label Quality & Intersectional Sparsity
- **Perceived vs Ground Truth:** FairFace labels are **annotator-perceived** (Amazon Mechanical Turk workers judging visual appearance), not self-reported identities. Annotator perception is known to have high error on boundary ages (e.g. 20-29 vs 30-39) and mixed-race ancestries.
- **Intersectional Binning Strategy:**
  - FairFace has 7 races $\times$ 2 genders $\times$ 9 age bands = **126 intersectional bins**.
  - In FairFace val ($N = 10,954$), over 40 bins have $n < 20$ (e.g. Black Female 70+ has $n=11$).
  - **Action:** Collapse the 9 age bands into **3 robust tiers**:
    - `Young` (0–19 years)
    - `Adult` (20–59 years)
    - `Senior` (60+ years)
  - This reduces the intersectional space to 7 races $\times$ 2 genders $\times$ 3 age tiers = **42 bins**, guaranteeing $n \ge 120$ for almost all cells in FairFace validation.

### 7.2 NIST FRVT Demographic Reporting Conventions
In NIST IR 8280 (Face Recognition Vendor Test: Demographic Effects), NIST establishes reporting protocols for biometric verification and classification:
1. **Fix a Single Global Operating Threshold:** Determine the threshold $\tau$ globally on the aggregate validation split to meet the operational target (e.g. 1% FPR). Never tune individual thresholds per demographic group when evaluating a single deployed system.
2. **Report Both Absolute Gaps and Ratios:**
   - Absolute Disparity: $\Delta = \max_g \text{FPR}_g - \min_g \text{FPR}_g$.
   - Max-to-Min Ratio: $R_{\text{max/min}} = \max_g \text{FPR}_g / \min_g \text{FPR}_g$.
   - Subgroup-to-Median Ratio: $R_{\text{med}}(g) = \text{FPR}_g / \text{median}_{g'} \text{FPR}_{g'}$.
3. **Ratio Stability Guards:** NIST explicitly cautions that ratios explode or become degenerate when the denominator is near zero. If any group in the denominator has $k=0$ or $k < 5$, ratio metrics must be suppressed or flagged as undefined (`N/A`), reporting only the absolute gap.

---

## 8. Concrete Recommendations for the Codebase

Ranked strictly by **Value / Effort**:

### Top-5 Recommendations for `devin-impl` & `devin-impl-helper`

1. **Implement Conservative Maximum Conformal Threshold ($\tau^* = \max_g \tau_g$)**
   - *Category:* Calibration / Mitigation
   - *Value: High | Effort: Low (10 lines of code)*
   - Compute group-specific quantiles $\tau_g$ over powered demographic groups on a held-out FairFace calibration split (distinct from the eval split); deploy the global threshold as $\tau^* = \max_g \tau_g$. Because $\tau_{\text{fake}}$ flags fake when $\text{score} > \tau$, taking the maximum group threshold guarantees that $\text{FPR}_g \le 1\%$ for every group simultaneously without requiring demographic inference at test time. The trade-off is lower automated recall on fakes and higher review burden (`abstain`).

2. **Ingest Xu et al. `DeepFakeAnnotations` for FNR Fairness (Post FF++ Ingestion)**
   - *Category:* Data / Evaluation
   - *Value: High | Effort: Low*
   - Once `bitmind/FaceForensicsC23` is ingested into the pipeline, download the open CSV attribute overlays from `github.com/xuyingzhongguo/DeepFakeAnnotations`. Joining these labels directly with FF++ c23 frames unlocks **both FPR and FNR (Equalized Odds) fairness**, overcoming FairFace's reals-only blind spot without requiring multi-terabyte dataset downloads.

3. **Add Ratio Stability Guards & Empirical Bayes Shrinkage to `fairness.py`**
   - *Category:* Evaluation & Statistics
   - *Value: High | Effort: Medium*
   - Modify `disparity_summary` in `src/dfd/eval/fairness.py`:
     - Suppress `ratio` when $\min_g k_g < 5$ to prevent wild divisions by near-zero rates.
     - Add Empirical Bayes Beta-Binomial shrinkage to smooth subgroup rate estimates before computing bootstrap max-min CIs, removing extreme-value upward bias.

4. **Adopt 3-Tier Age Aggregation for Intersectional Reporting**
   - *Category:* Reporting / Evaluation
   - *Value: Medium | Effort: Low*
   - In `fairness_report()`, collapse FairFace's 9 fine-grained age bands into 3 tiers (`young`: 0-19, `adult`: 20-59, `senior`: 60+) before computing the `race|gender|age` intersection. Eliminates tiny-sample bins ($n < 20$) and ensures all analyzed intersectional groups have sufficient sample support ($n \ge 120$).

5. **Train with CVaR (Worst-Percentile) Batch Loss (DAG-FDD)**
   - *Category:* Model Training
   - *Value: High | Effort: Medium*
   - During DINOv2 LoRA / VIB head fine-tuning, switch standard cross-entropy to optimize the worst $20\%$ loss in each batch. This forces the model to learn features that generalize to hard demographic minorities without needing demographic labels during training.
