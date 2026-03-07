#!/bin/bash

VISUALIZE=${VISUALIZE:-True}
SEED=42

NITERS_LIST=(3000, 5000, 10000)
KL_ANNEAL_LIST=(True, False)
LR_LIST=(0.01, 0.005, 0.001)

TOTAL=$(( ${#NITERS_LIST[@]} * ${#KL_ANNEAL_LIST[@]} * ${LR_LIST[@]}))
COUNT=0

for NITERS in "${NITERS_LIST[@]}"; do
    for KL_ANNEAL in "${KL_ANNEAL_LIST[@]}"; do
        for LR in "${LR_LIST[@]}"; do
            COUNT=$(( COUNT + 1 ))

            if [ "$KL_ANNEAL" = "True" ]; then
                LABEL="kl_anneal"
            else 
                LABEL="no_kl_anneal"
            fi

            echo "============================================================"
            echo "  Run ${COUNT}/${TOTAL}: niters=${NITERS}  lr=${LR}  ${LABEL}"
            echo "============================================================"
            
            python train.py \
                --niters    "$NITERS" \
                --lr        "$LR" \
                --seed      "$SEED" \
                --kl_anneal "$KL_ANNEAL" \
                --visualize "$VISUALIZE"

            echo ""
        done
    done
done

echo "All ${TOTAL} experiments complete."
