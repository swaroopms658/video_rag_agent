#!/usr/bin/env bash
# Post-eval pipeline: run after gen eval for all 4 systems completes.
set -e
cd "$(dirname "$0")/.."

echo "=== Step 1: Verify all 4 CSVs complete (59 rows each) ==="
for sys in bm25 cross_encoder cfrag_lite itma; do
    n=$(tail -n +2 "analysis/results_test/${sys}.csv" | wc -l)
    echo "  ${sys}: ${n}/59 rows"
done

echo ""
echo "=== Step 2: Add BERTScore (roberta-large, ~1.5GB RAM) ==="
python scripts/add_bertscore.py

echo ""
echo "=== Step 3: Extract generation metrics ==="
python scripts/extract_gen_metrics.py

echo ""
echo "=== Step 4: Regenerate LaTeX tables ==="
python analysis/make_paper_tables.py --out paper/tables

echo ""
echo "=== Done. Update results.md with the Table X above, then commit. ==="
