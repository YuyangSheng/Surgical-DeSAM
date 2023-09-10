# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss, dice_coefficient, focal_loss_masks)
from .transformer import build_transformer
from models.sam.segment_anything.modeling import ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer
from segment_anything import sam_model_registry, SamPredictor

from torch.nn import functional as FN
from torch import nn

import torch.distributions as dist
import numpy as np
import monai

def add_noise_to_bbox(bbox, max_noise=20):
    '''
    args: bbox (N, 4)
    '''
    # Calculate standard deviation as 10% of the box sidelength
    box_width = bbox[:, 2] - bbox[:, 0]
    box_height = bbox[:, 3] - bbox[:, 1]
    std_dev = 0.1 * torch.max(box_width, box_height)
    
    # Create normal distribution for generating noise
    noise_dist = dist.Normal(0, std_dev) # (num_boxes, )
    num_boxes = bbox.shape[0]

    # Generate random noise for each coordinate
    x1_noise = noise_dist.sample()
    y1_noise = noise_dist.sample()
    x2_noise = noise_dist.sample()
    y2_noise = noise_dist.sample()
    
    # Clip noise to a maximum of 20 pixels
    x1_noise = torch.clamp(x1_noise, -max_noise, max_noise)
    y1_noise = torch.clamp(y1_noise, -max_noise, max_noise)
    x2_noise = torch.clamp(x2_noise, -max_noise, max_noise)
    y2_noise = torch.clamp(y2_noise, -max_noise, max_noise)
    noise = torch.stack([x1_noise, y1_noise, x2_noise, y2_noise], dim=1)

    # Add noise to the original coordinates
    noisy_bbox = bbox + noise
    
    return noisy_bbox

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox.cpu())
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def postprocess_masks(masks, input_size, original_size,) -> torch.Tensor:
        masks = FN.interpolate(
            masks,
            input_size,
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = FN.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

class SAMModel(nn.Module):
    def __init__(self, device, model_type='vit_b', ckpt_path='sam_vit_b_01ec64.pth'):
        super().__init__()
        
        sam = sam_model_registry[model_type](checkpoint=ckpt_path).to(device)
        self.predictor = SamPredictor(sam)

        self.sam_image_encoder = sam.image_encoder
        self.sam_prompt_encoder = sam.prompt_encoder
        self.sam_mask_decoder = sam.mask_decoder
        self.device = device
        self.upsample_layer = nn.ConvTranspose2d(
                                                in_channels=256,      # Number of input channels (should match your input image)
                                                out_channels=256,     # Number of output channels (same as input channels for no change)
                                                kernel_size=4,      # Kernel size for the convolution
                                                stride=2,           # Upsampling factor (doubles the spatial dimensions)
                                                padding=1,          # Padding to maintain spatial dimensions
                                                output_padding=0,   # Additional padding to adjust the output size
                                            )
        
    def forward(self, batched_img, boxes, image_embeddings=None, sizes=(1024, 1024), add_noise=True):
        if image_embeddings is None:
            image_embeddings = self.sam_image_encoder(batched_img)[0].unsqueeze(0)
        else:
            image_embeddings = self.upsample_layer(image_embeddings) # (3, 32, 32)-> (3, 64, 64)
            # print(image_embeddings.shape)
            
        gt_boxes = rescale_bboxes(boxes, sizes) # xyxy-format with shape (N, 4)
        if add_noise:
            noisy_boxes = add_noise_to_bbox(gt_boxes)
        else:
            noisy_boxes = gt_boxes
        
        transformed_boxes = self.predictor.transform.apply_boxes_torch(noisy_boxes, sizes).to(self.device)
        if gt_boxes.shape[0] == 0:
            transformed_boxes = None
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points=None,
                                                         boxes=transformed_boxes,
                                                         masks=None)
        # print(gt_boxes.shape, transformed_boxes)
        # print(image_embeddings.shape, self.sam_prompt_encoder.get_dense_pe().shape)
        # print(sparse_embeddings.shape, dense_embeddings.shape)
        low_res_masks, iou_predictions = self.sam_mask_decoder(
                                                                image_embeddings=image_embeddings,
                                                                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                                                                sparse_prompt_embeddings=sparse_embeddings,
                                                                dense_prompt_embeddings=dense_embeddings,
                                                                multimask_output=False,
                                                                # hq_token_only=False,
                                                                # interm_embeddings=False
                                                            )
        pred_masks = postprocess_masks(low_res_masks, input_size=sizes, original_size=sizes)
        return low_res_masks.reshape(-1, 256, 256), pred_masks.reshape(-1, sizes[0], sizes[1]), iou_predictions

class CustomSAMModel(nn.Module):
    def __init__(self, device, img_size=224, prompt_embed_dim=256, image_embedding_size=(14, 14), input_image_size=(224, 224)):
        super().__init__()
        self.device = device
        self.sam_image_encoder = ImageEncoderViT(img_size=img_size)
        self.sam_prompt_encoder = PromptEncoder(embed_dim=prompt_embed_dim, 
                                                image_embedding_size=image_embedding_size, 
                                                input_image_size=input_image_size,
                                                mask_in_chans=16
                                                )
        self.sam_mask_decoder = MaskDecoder(transformer_dim=256,
                                            transformer=TwoWayTransformer(depth=2,
                                                                    embedding_dim=prompt_embed_dim,
                                                                    mlp_dim=2048,
                                                                    num_heads=8))
        
    def forward(self, batched_img, image_embeddings, boxes, sizes, predictor, add_noise=True):
        image_embeddings = self.sam_image_encoder(batched_img)
        
        gt_boxes = rescale_bboxes(boxes, sizes) # xyxy-format
        if add_noise:
            noisy_boxes = add_noise_to_bbox(gt_boxes)
        else:
            noisy_boxes = gt_boxes
        
        transformed_boxes = predictor.transform.apply_boxes_torch(noisy_boxes, (224, 224)).to(self.device)
        if gt_boxes.shape[0] == 0:
            transformed_boxes = None
        sparse_embeddings, dense_embeddings = self.sam_prompt_encoder(points=None,
                                                         boxes=transformed_boxes,
                                                         masks=None)
        # print(gt_boxes.shape, transformed_boxes)
        # print(image_embeddings.shape, self.sam_prompt_encoder.get_dense_pe().shape)
        # print(sparse_embeddings.shape, dense_embeddings.shape)
        low_res_masks, iou_predictions = self.sam_mask_decoder(
                                                                image_embeddings=image_embeddings,
                                                                image_pe=self.sam_prompt_encoder.get_dense_pe(),
                                                                sparse_prompt_embeddings=sparse_embeddings,
                                                                dense_prompt_embeddings=dense_embeddings,
                                                                multimask_output=False,
                                                            )
        pred_masks = postprocess_masks(low_res_masks, input_size=(224, 224), original_size=(224, 224))
        return low_res_masks.reshape(-1, 56, 56), pred_masks.reshape(-1, 224, 224), iou_predictions
    
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model # =args.hidden_dim 256
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.input_proj = nn.ModuleList([
        #         nn.Sequential(
        #             nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
        #             nn.GroupNorm(32, hidden_dim),
        #         )])
        self.input_proj = nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1)
        # self.input_proj = nn.Conv2d(256, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)
        # print('features:', features[0].tensors.shape)
        # print('pos:', pos[0].shape)
        src, mask = features[-1].decompose()
        # print('src shape:', src.shape, mask.shape)
        assert mask is not None

        hs, image_embeddings = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out, image_embeddings

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, use_matcher=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)
        self.use_matcher = use_matcher

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=0.25, gamma=2) * src_logits.shape[1]

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx] # (N, 4)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        # src_idx = self._get_src_permutation_idx(indices)
        # tgt_idx = self._get_tgt_permutation_idx(indices)
  
        src_masks = outputs["pred_masks"].unsqueeze(0) # (bs, N, H, W)
        # src_masks = src_masks[src_idx]
        # masks = [t["masks"] for t in targets]
        # # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        # target_masks = target_masks.to(src_masks)
        target_masks = targets[0]['masks'].unsqueeze(0)
        
        # target_masks = target_masks[tgt_idx]

        # # upsample predictions to the target size
        # src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        # src_masks = src_masks[:, 0].flatten(1)

        # target_masks = target_masks.flatten(1)
        # target_masks = target_masks.view(src_masks.shape)

        # ---------sam--------
        # src_masks = outputs['pred_masks']
        # target_masks = targets[0]['masks']
        dice_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        losses = {
            "loss_mask": focal_loss_masks(src_masks.cpu(), target_masks.cpu(), num_boxes),
            # "loss_dice": dice_loss(src_masks, target_masks.cpu(), num_boxes),
            "loss_dice": dice_loss(src_masks.cpu(), target_masks.cpu()),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
        else:
            indices = None

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # print('Originial output:')
        # print('Labels:', labels, 'bbox:', boxes)

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    num_classes = 8 if args.dataset_file == 'endovis17' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    if args.model:
        # pretrained_model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
        # # save model weights
        # torch.save(pretrained_model.state_dict(), 'detr_weights.pth')

        # initialize model weights
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )

        weights = torch.load('detr_weights.pth')
        # checkpoint = torch.load('outputs/ckpt_best.pth')
        # weights = checkpoint['model']
        # delete specific layers in weights
        exclude_keys = ['class_embed.weight', 'class_embed.bias', 'input_proj.weight']
        for key in exclude_keys:
            del weights[key]

        model.load_state_dict(weights, strict=False)

    else:
        model = DETR(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=args.num_queries,
            aux_loss=args.aux_loss,
        )
   

    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    
    seg_losses = ['masks']
    seg_losses += losses
    seg_weight_dict = {'loss_mask': args.mask_loss_coef, 'loss_dice': args.dice_loss_coef, 'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef,
                       'loss_giou': args.giou_loss_coef}
    #TODO if use_matcher == True: use Hungarian matching for predicted boxes and GT boxes 
    seg_criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=seg_weight_dict,
                             eos_coef=args.eos_coef, losses=seg_losses, use_matcher=True)
    seg_criterion.to(device)
    
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, seg_criterion, postprocessors

