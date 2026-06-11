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
        echo "  Fail"
        FAIL=$((FAIL + 1))
        FAIL_LIST+=($ENS)
    else
        echo "  Success"
        SUCCESS=$((SUCCESS + 1))
    fi
done

[ $FAIL -gt 0 ] && echo "Fail ensemble: ${FAIL_LIST[*]}"
