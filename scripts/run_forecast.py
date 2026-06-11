#!/bin/bash
# run_ensemble_clean.sh — ENSEMBLE 변수를 인자로 받도록 수정된 버전

LOG_DIR="./ensemble_logs"
mkdir -p "$LOG_DIR"

SUCCESS=0
FAIL=0
FAIL_LIST=()

for ENS in $(seq 1 30); do
    echo "[$(date '+%H:%M:%S')] ▶ Ensemble $ENS/30"
    
    ENSEMBLE=$ENS python forecast.py 2>&1 | tee "$LOG_DIR/ens${ENS}.log"
    
    EXIT=${PIPESTATUS[0]}
    if [ $EXIT -ne 0 ]; then
        echo "  ✗ 실패"
        FAIL=$((FAIL + 1))
        FAIL_LIST+=($ENS)
    else
        echo "  ✓ 완료"
        SUCCESS=$((SUCCESS + 1))
    fi
done

echo ""
echo "===== 결과 요약 ====="
echo "성공: $SUCCESS / 30"
echo "실패: $FAIL / 30"
[ $FAIL -gt 0 ] && echo "실패 앙상블: ${FAIL_LIST[*]}"
