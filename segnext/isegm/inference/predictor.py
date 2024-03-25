import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F

from isegm.model.is_plainvit_model import PlainVitModel
from isegm.inference.clicker import Clicker, Click


class BasePredictor(object):
    def __init__(self, model: PlainVitModel) -> None:
        """ 
        Arguments:
            model: the model for mask prediction.
        """
        super().__init__()                
        self.model = model
        self.to_tensor = transforms.ToTensor()

    def set_image(self, image: np.ndarray) -> None:
        """
        Set an image and obtain its embedding to allow repeated, 
        efficient mask prediction given diverse prompts.

        Arguments:
            image: the image to be segmented.
        """
        image = self.to_tensor(image).to(self.device)
        if len(image.shape) == 3:
            image = image.unsqueeze(0) # CHW -> BCHW
        self.orig_img_shape = image.shape
        self.prev_mask = torch.zeros_like(image, dtype=torch.float32)[:, :1, :, :]
        self.image_feats = self.model.get_image_feats(image)

    def predict_sat(self, points_per_side: int = 16) -> np.ndarray:
        """Segment Anything Task (SAT)
        """
        # resize the previous mask
        target_length = self.model.target_length
        prev_mask = F.interpolate(
            self.prev_mask, target_length, mode='bilinear', align_corners=False)

        # convert points to mask
        clicks = []
        patch_size = target_length // points_per_side
        for i in range(points_per_side):
            for j in range(points_per_side):
                h = patch_size//2 + patch_size*i
                w = patch_size//2 + patch_size*j
                clicks.append([Click(is_positive=True, coords=(h, w), indx=0)])

        points_nd = self.get_points_nd(clicks)            
        point_mask = self.model.dist_maps(prev_mask.shape, points_nd)

        prev_mask_tiled = torch.tile(prev_mask, (points_nd.shape[0], 1, 1, 1))
        prompt_mask = torch.cat((prev_mask_tiled, point_mask), dim=1)
        prompt_feats = self.model.visual_prompts_encoder(prompt_mask)

        # keep shape
        B, N, C = prompt_feats.shape
        H = target_length // self.model.visual_prompts_encoder.patch_size[0]
        W = target_length // self.model.visual_prompts_encoder.patch_size[1]
        prompt_feats = prompt_feats.transpose(1,2).contiguous()
        prompt_feats = prompt_feats.reshape(B, 1, C, H, W)

        preds = []
        for idx in range(B):
            pred_logits = self.model(None, self.image_feats, prompt_feats[idx])
            pred = torch.sigmoid(pred_logits['instances'])
            pred = pred.cpu().numpy()[0, 0]
            preds.append(pred)
        return preds

    def predict(self, clicker: Clicker) -> np.ndarray:
        """
        TBD
        """
        clicks_list = clicker.get_clicks()
        points_nd = self.get_points_nd([clicks_list])        

        prompts = {'points': points_nd, 'prev_mask': self.prev_mask}
        prompt_feats = self.model.get_prompt_feats(self.orig_img_shape, prompts)

        pred_logits = self.model(self.orig_img_shape, self.image_feats, prompt_feats)
        prediction = torch.sigmoid(pred_logits['instances'])
        self.prev_mask = prediction

        return prediction.cpu().numpy()[0, 0]

    def get_points_nd(self, clicks_lists) -> torch.Tensor:
        """
        Arguments:
            clicks_lists: a list containing clicks list for a batch
            
        Returns:
            torch.Tensor: a tensor of points with shape B x 2N x 3 
        """
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)
    
    @property
    def device(self) -> torch.device:
        return self.model.device
    
    def get_states(self):
        return {'prev_prediction': self.prev_mask.clone()}

    def set_states(self, states=None):
        self.prev_mask = states['prev_prediction']