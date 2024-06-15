import torch
from torch import nn

class NeuconWLoss(nn.Module):
    """
    Equation 13 in the NeRF-W paper.
    Name abbreviations:
        c_l: coarse color loss
        f_l: fine color loss (1st term in equation 13)
        s_l: sigma loss (3rd term in equation 13)
    """
    def __init__(self, coef=1, igr_weight=0.1, mask_weight=0.1, depth_weight=0.1, floor_weight=0.01, config=None):
        super().__init__()
        self.coef = coef
        self.igr_weight = igr_weight
        self.mask_weight = mask_weight
        self.depth_weight = depth_weight
        self.floor_weight = depth_weight
        self.config = config
        self.loss = nn.MSELoss()

    def forward(self, inputs, targets, masks=None):
        ret = {}
        if masks is None:
                masks = torch.ones((targets.shape[0], 1)).to(targets.device)
        mask_sum = masks.sum() + 1e-5
        color_error = (inputs['color'] - targets) * masks
        ret['color_loss'] = torch.nn.functional.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum

        ret['normal_loss'] = self.igr_weight * inputs['gradient_error'].mean()

        if self.config.NEUCONW.MESH_MASK_LIST is not None:
            ret['mask_error'] = self.mask_weight * inputs['mask_error'].mean()

        if self.config.NEUCONW.DEPTH_LOSS:
            #ret['sfm_depth_loss'] = self.depth_weight * inputs['sfm_depth_loss'].mean()
            ret['sfm_sdf_loss'] =  self.depth_weight * inputs['sfm_sdf_loss'].mean()

            #depth_reproj_mask = inputs["depth_reproj_error"] <= 0.1
            pixel_dist_mask = inputs["pixel_dist_error"] < 1
            edge_mask = inputs["edge_prior"] <= 0.1
            final_mask = torch.logical_and(pixel_dist_mask, edge_mask)
            final_mask_sum = final_mask.sum() + 1e-5

            #normal_prior_error = (inputs["normals"] - inputs["normals_prior"]) * final_mask
            #ret["normals_prior_loss"] = torch.nn.functional.l1_loss(normal_prior_error, torch.zeros_like(normal_prior_error), reduction='sum') / final_mask_sum
            #ret["normals_prior_loss"] = 0.01 * ret["normals_prior_loss"]
 
            normal_cos_error = ( 1 - (inputs["normals"] * inputs["normals_prior"]).sum(1).unsqueeze(1) ) * normal_mask
            ret["normals_cos_loss"] = torch.nn.functional.l1_loss(normal_cos_error, torch.zeros_like(normal_cos_error), reduction='sum') / final_mask_sum
            ret["normals_cos_loss"] = 0.01 * ret["normals_cos_loss"]


        if self.config.NEUCONW.FLOOR_NORMAL:
            ret['floor_normal_error'] = self.floor_weight * inputs['floor_normal_error'].mean()

        for k, v in ret.items():
            ret[k] = self.coef * v

        return ret

loss_dict = {'neuconw': NeuconWLoss}