import os
import gdown

# download models
save_folder ='weights'
os.makedirs(save_folder, exist_ok=True)

# weights for vitb_sax1 (trained on cocolvis)
weights_sa1_url = 'https://drive.google.com/uc?id=1G_m9_sVYVD2MBDf5ulrAUz-iSvi8bQfZ'
output = f'{save_folder}/vitb_sa1_cocolvis_epoch_90.pth'
gdown.download(weights_sa1_url, output, quiet=False)

# weights for vitb_sax2 (trained on cocolvis)
weights_sa2_url = 'https://drive.google.com/uc?id=1uHEcvwA2spsLQqDJiYKqv1pf3vgLAG7-'
output = f'{save_folder}/vitb_sa2_cocolvis_epoch_90.pth'
gdown.download(weights_sa2_url, output, quiet=False)

# weights for vitb_sax2_ft (trained on cocolvis and finetuned on hqseg-44k)
weights_sa2_ft_url = 'https://drive.google.com/uc?id=1qkXIvxnlY3Z4xLYGWJcoAHBUNyDmvwPS'
output = f'{save_folder}/vitb_sa2_cocolvis_hq44k_epoch_0.pth'
gdown.download(weights_sa2_ft_url, output, quiet=False)

# weights for mae-pretrained vitb (only required for training)
weights_mae_vitb_url = 'https://drive.google.com/uc?id=1abaho6lal3Yxthbtw2U-wEshnoPZpR-K'
output = f'{save_folder}/mae_pretrain_vit_base.pth'
gdown.download(weights_mae_vitb_url, output, quiet=False)
