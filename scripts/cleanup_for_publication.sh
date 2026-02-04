#!/bin/bash
# =============================================================================
# Cleanup Script for Publication
# =============================================================================
# This script removes sensitive files, logs, artifacts, and obsolete documents
# before publishing the repository.
#
# Usage:
#   ./scripts/cleanup_for_publication.sh [--dry-run]
#
# Options:
#   --dry-run    Show what would be deleted without actually deleting
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "=== DRY RUN MODE - No files will be deleted ==="
    echo ""
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

delete_file() {
    local file="$1"
    local reason="$2"
    if [[ -f "$file" ]]; then
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY-RUN]${NC} Would delete: $file ($reason)"
        else
            rm -f "$file"
            echo -e "${GREEN}[DELETED]${NC} $file ($reason)"
        fi
    fi
}

delete_dir() {
    local dir="$1"
    local reason="$2"
    if [[ -d "$dir" ]]; then
        if $DRY_RUN; then
            echo -e "${YELLOW}[DRY-RUN]${NC} Would delete directory: $dir ($reason)"
        else
            rm -rf "$dir"
            echo -e "${GREEN}[DELETED]${NC} $dir/ ($reason)"
        fi
    fi
}

echo "=============================================="
echo "  Deposium Embeddings - Publication Cleanup"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# TIER 1: CRITICAL - Sensitive files with credentials
# -----------------------------------------------------------------------------
echo -e "${RED}=== TIER 1: CRITICAL - Credentials & Secrets ===${NC}"
echo ""

delete_file ".env" "Contains real HF_TOKEN and API keys"
delete_file "test_classifier_api.py" "Contains hardcoded API key"
delete_file "test_graph_api.py" "Contains hardcoded API key"
delete_file ".coverage" "Test coverage artifact"
delete_file "=0.6.0" "Pip installation artifact"

echo ""

# -----------------------------------------------------------------------------
# TIER 2: HIGH - Logs and build artifacts
# -----------------------------------------------------------------------------
echo -e "${YELLOW}=== TIER 2: Logs & Build Artifacts ===${NC}"
echo ""

# Root level logs
delete_file "docker_build.log" "Build log"
delete_file "test_auth.log" "Test log"
delete_file "test_auth_dual.log" "Test log"
delete_file "test_server.log" "Test log"
delete_file "training_distillation.log" "Training log"
delete_file "uvicorn.log" "Server log"
delete_file "uvicorn_new.log" "Server log"
delete_file "uvicorn_optimized.log" "Server log"
delete_file "mteb_quick_output.log" "MTEB evaluation log"

# JSON result artifacts (not needed for publication)
delete_file "mteb_full_comparison.json" "MTEB comparison artifact"
delete_file "mteb_full_results.json" "MTEB results artifact"
delete_file "sts_comparison_results.json" "STS comparison artifact"

# Research archives logs
delete_file "research-archives/granite-4.0-micro/granite_full_comparison.log" "Research log"
delete_file "research-archives/granite-4.0-micro/granite_multilingual_results.log" "Research log"
delete_file "research-archives/huggingface_publication_qwen25/examples/monolingual_test_output.log" "Test output log"
delete_file "research-archives/huggingface_publication_qwen25/examples/advanced_test_output.log" "Test output log"

echo ""

# -----------------------------------------------------------------------------
# TIER 3: MEDIUM - Test scripts and unknown files
# -----------------------------------------------------------------------------
echo -e "${YELLOW}=== TIER 3: Test Scripts & Unknown Files ===${NC}"
echo ""

delete_file "test_real_image.py" "Manual test script"
delete_file "benchmark_embeddings.py" "Benchmark script (not essential)"
delete_file "compare_models.py" "Model comparison utility"
delete_file "contrat.png" "Unknown image file"
delete_file "graph.png" "Unknown image file"

echo ""

# -----------------------------------------------------------------------------
# TIER 4: Obsolete documentation
# -----------------------------------------------------------------------------
echo -e "${YELLOW}=== TIER 4: Obsolete Documentation ===${NC}"
echo ""

delete_file "DEPLOYMENT_SUCCESS.md" "Dated deployment report"
delete_file "HUGGINGFACE_PUBLICATION_SUMMARY.md" "Internal publication notes"
delete_file "research-archives/daddy_project.md" "Template/placeholder file"
delete_file "research-archives/STORAGE_STRATEGY.md" "Internal storage strategy"

# Old/redundant docs
delete_file "docs/analysis/reports/SESSION_SUMMARY.md" "Session notes"
delete_file "docs/analysis/reports/DADDY_INIT_REPORT.md" "Internal report"
delete_file "docs/analysis/reports/STATUS.md" "Dated status"
delete_file "docs/NUMPY_2_MIGRATION_TRACKING.md" "Migration tracking (completed)"

# Integration troubleshooting docs (internal)
delete_file "docs/integrations/FIX_HUGGINGFACE_SPACE.md" "Troubleshooting doc"
delete_file "docs/integrations/QUICK_FIX_BOTH_SPACES.md" "Temporary fix doc"
delete_file "docs/integrations/SOLUTION_GPU_5GB.md" "Workaround doc"

# Deployment status (dated)
delete_file "docs/guides/deployment/DEPLOYMENT_STATUS.md" "Dated status"

echo ""

# -----------------------------------------------------------------------------
# TIER 5: Directories to consider removing (manual review)
# -----------------------------------------------------------------------------
echo -e "${YELLOW}=== TIER 5: Directories for Manual Review ===${NC}"
echo ""

if $DRY_RUN; then
    echo "The following directories may be candidates for removal:"
    echo "  - research-archives/       (historical research, 3+ subdirs)"
    echo "  - docs/analysis/           (internal analysis reports)"
    echo "  - mteb_results_qwen25/     (MTEB evaluation results)"
    echo ""
    echo "Review these manually before publishing."
else
    echo "Review these directories manually before publishing:"
    echo "  - research-archives/"
    echo "  - docs/analysis/"
    echo "  - mteb_results_qwen25/"
fi

echo ""

# -----------------------------------------------------------------------------
# Verification
# -----------------------------------------------------------------------------
echo "=============================================="
echo "  Cleanup Summary"
echo "=============================================="
echo ""

# Check for remaining .env files
if [[ -f ".env" ]]; then
    echo -e "${RED}WARNING: .env file still exists!${NC}"
else
    echo -e "${GREEN}OK: No .env file present${NC}"
fi

# Check for remaining log files at root
LOG_COUNT=$(find . -maxdepth 1 -name "*.log" 2>/dev/null | wc -l)
if [[ $LOG_COUNT -gt 0 ]]; then
    echo -e "${YELLOW}WARNING: $LOG_COUNT .log files still at root level${NC}"
else
    echo -e "${GREEN}OK: No .log files at root level${NC}"
fi

# Check .env.example exists
if [[ -f ".env.example" ]]; then
    echo -e "${GREEN}OK: .env.example present for users${NC}"
else
    echo -e "${RED}WARNING: .env.example missing!${NC}"
fi

echo ""

if $DRY_RUN; then
    echo "=== DRY RUN COMPLETE - No files were deleted ==="
    echo "Run without --dry-run to perform actual cleanup."
else
    echo "=== CLEANUP COMPLETE ==="
    echo "Remember to:"
    echo "  1. Run 'git status' to review changes"
    echo "  2. Review remaining docs in research-archives/ and docs/analysis/"
    echo "  3. Test the application before publishing"
fi
