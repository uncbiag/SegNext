
MODEL_CONFIG=./segnext/models/default/plainvit_base1024_hqseg44k_sax2.py

torchrun --nproc-per-node=4 \
	     --master-port 29504 \
	     ./segnext/train.py ${MODEL_CONFIG} \
	     --weights ./weights/vitb_sa2_cocolvis_epoch_90.pth \
	     --batch-size=8 \
	     --gpus=0,1,2,3
