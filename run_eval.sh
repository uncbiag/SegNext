
# MODEL_PATH=./weights/vitb_sa2_cocolvis_hq44k_epoch_0.pth
# python ./segnext/scripts/evaluate_model.py \
#     --gpus=0 \
#     --checkpoint=${MODEL_PATH} \
#     --datasets=DAVIS,HQSeg44K \
#     --print-ious 

MODEL_PATH=/playpen-raid2/qinliu/models/model_0325_2024/default/plainvit_base1024_hqseg44k_sax2/001/checkpoints/000.pth
python ./segnext/scripts/evaluate_model.py \
    --gpus=0 \
    --checkpoint=${MODEL_PATH} \
    --datasets=DAVIS,HQSeg44K    