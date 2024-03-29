import os
import gdown

# download models
save_folder ='weights'
os.makedirs(save_folder, exist_ok=True)

# weights for vitb_sax1 (trained on cocolvis)
weights_sa1_url = 'https://drive.google.com/uc?id=1eqkd5-J9MELGIw2WRcT5hejsnGc5oO30'
output = f'{save_folder}/vitb_sa1_cocolvis_epoch_90.pth'
gdown.download(weights_sa1_url, output, quiet=False)

# weights for vitb_sax2 (trained on cocolvis)
weights_sa2_url = 'https://drive.google.com/uc?id=1oxwCm4bFby6RgltO_tl54BqRN9tojylT'
output = f'{save_folder}/vitb_sa2_cocolvis_epoch_90.pth'
gdown.download(weights_sa2_url, output, quiet=False)

# weights for vitb_sax2_ft (trained on cocolvis and finetuned on hqseg-44k)
weights_sa2_ft_url = 'https://drive.google.com/uc?id=1yDN3mwBBO33TlA0KRdO2s07Q5HWXR6nt'
output = f'{save_folder}/vitb_sa2_cocolvis_hq44k_epoch_0.pth'
gdown.download(weights_sa2_ft_url, output, quiet=False)

# weights for mae-pretrained vitb (only required for training)
weights_mae_vitb_url = 'https://drive.google.com/uc?id=1UFa-YHKSZQBrLla0za1fqfoK3QmCee3O'
output = f'{save_folder}/mae_pretrain_vit_base.pth'
gdown.download(weights_mae_vitb_url, output, quiet=False)
