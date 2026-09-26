#!/usr/bin/env bash
# Reproduce every table, figure and quoted number of the paper from the shipped LLM response
# caches. No API key is needed and no API call is made.
#
#   ./reproduce.sh all            # the three experiments, then collect figures into figures/
#   ./reproduce.sh exp1|exp2|exp3 # one experiment
#   ./reproduce.sh figures        # only collect the figure PDFs into figures/
#
# Set PYTHON to choose the interpreter (default: python).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
export MPLBACKEND=Agg

run() { echo "+ $*"; "$PY" "$@"; }

exp1() {
  cd "$ROOT/experiments/01_synthetic_cover_letters"
  local C=(--config config.yaml)
  run scripts/00_generate_dataset.py "${C[@]}"
  run scripts/01_build_pairs.py "${C[@]}"
  run scripts/04_fit_comparative.py "${C[@]}"
  run scripts/10a_sandwich_ci.py "${C[@]}"
  run scripts/06d_cost_match.py "${C[@]}"
  run scripts/06e_calibrate_theta.py "${C[@]}"
  run scripts/06f_tau_robustness.py "${C[@]}"
  run scripts/08_bootstrap.py "${C[@]}"
  run scripts/08b_bootstrap_taxonomy.py "${C[@]}"
  run scripts/08c_bootstrap_taxo_arm.py "${C[@]}"
  run scripts/08d_bootstrap_ours_letters.py "${C[@]}"
  run scripts/08d_bootstrap_ours_letters.py "${C[@]}" --frozen-taxonomy
  run scripts/08e_bootstrap_1b_letters.py "${C[@]}"
  run scripts/08e_bootstrap_1b_letters.py "${C[@]}" --arm 2b
  run scripts/08f_bootstrap_2a_letters.py "${C[@]}"
  run scripts/07d_score_pointwise_canonical.py collect "${C[@]}"
  run scripts/09b_ablation_table.py "${C[@]}"
  run scripts/09_build_tables_figures.py "${C[@]}"
  run scripts/10_paper_tables.py "${C[@]}"
  run scripts/11_beta_per_criterion.py "${C[@]}"
  run scripts/11b_beta_k_metrics.py "${C[@]}"
  run scripts/12_calibration.py "${C[@]}"
  run scripts/13_claim_ledger.py "${C[@]}"
  run figures/make_recovery_heatmap_comparative.py
  run figures/make_recovery_heatmap_pointwise.py
  run figures/make_threshold_pair.py
  run figures/make_paired_panels.py
  run figures/make_calibration_figures.py "${C[@]}"
}

exp2() {
  cd "$ROOT/experiments/02_changemyview"
  run scripts/09_paired_lr.py
  run scripts/10_dimensionality.py
  run scripts/10b_matched_k.py
  run scripts/11_win_rates.py
  run scripts/12_contamination.py
  run scripts/13_paper_tables.py
  run scripts/14_paper_numbers.py
  run figures/make_cmv_dimension_accuracy.py
  run figures/make_cmv_dimension_curve.py
  run figures/make_cmv_diff_grid.py
  run figures/make_cmv_win_rates.py
  run figures/make_cmv_win_rates.py --full
}

exp3() {
  cd "$ROOT/experiments/03_interpretable_scoring"
  run scripts/00_build_pairs.py --run all
  run scripts/01_judge.py collect --run all
  run scripts/02_taxonomy.py --replay --run all
  run scripts/03_build_tensors.py --run all
  run scripts/04_fit.py --run all
  run scripts/05_diagnostics.py --run all
  run scripts/06_fit_reliable.py --run all
  run scripts/07_cross_judge_cluster.py --corpus all
  run scripts/08_calibration.py
  run figures/make_cross_judge_radar.py
  run figures/make_radar_panels.py
  run figures/make_total_score_distributions.py
  run figures/make_calibration_figures.py
  run scripts/09_paper_numbers.py
}

# Every figure of the paper, under the file name the paper includes.
figures() {
  local out="$ROOT/figures"
  mkdir -p "$out"
  local e1="$ROOT/experiments/01_synthetic_cover_letters/figures"
  local e2="$ROOT/experiments/02_changemyview/figures"
  local e3="$ROOT/experiments/03_interpretable_scoring/figures"
  local f
  for f in paired_a paired_b recovery_heatmap_comparative recovery_heatmap_m1 threshold_pair \
           calibration_reliability calibration_skill; do
    cp "$e1/$f.pdf" "$out/"
  done
  for f in cmv_dimension_accuracy cmv_dimension_curve cmv_diff_grid cmv_win_rates \
           cmv_win_rates_full; do
    cp "$e2/$f.pdf" "$out/"
  done
  for f in cross_judge_denoised_sigma total_score_distributions \
           radar_std_s_denoised_reliable_4judge_llm radar_std_s_denoised_reliable_asap_4judge_llm \
           calibration_reliability_4judge_llm calibration_reliability_asap_4judge_llm \
           calibration_brier_skill_4judge_llm; do
    cp "$e3/$f.pdf" "$out/"
  done
  echo "collected $(ls "$out"/*.pdf | wc -l) figures in $out"
}

case "${1:-all}" in
  exp1) exp1 ;;
  exp2) exp2 ;;
  exp3) exp3 ;;
  figures) figures ;;
  all) exp1; exp2; exp3; figures ;;
  *) echo "usage: $0 [all|exp1|exp2|exp3|figures]" >&2; exit 2 ;;
esac
