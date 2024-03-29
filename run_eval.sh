MODEL_PATH=./weights/vitb_sa2_cocolvis_hq44k_epoch_0.pth
python ./segnext/scripts/evaluate_model.py \
    --gpus=0 \
    --checkpoint=${MODEL_PATH} \
    --datasets=DAVIS,HQSeg44K \
    --print-ious 