
MODEL_ROOT=./weights
MODEL_PATH=${MODEL_ROOT}/vitb_sa2_cocolvis_hq44k_epoch_0.pth

python3 ./segnext/demo.py \
--checkpoint=${MODEL_PATH} \
--gpu 0