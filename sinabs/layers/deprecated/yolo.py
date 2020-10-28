import torch
from .layer import Layer
from typing import Tuple


class YOLOLayer(Layer):
    def __init__(self, anchors, num_classes, input_shape, img_dim=416,
                 return_loss=False, compute_rate=False):
        """
        An implementation of the YOLO layer. This is not a spiking layer,
        and the input needs to be converted to rates, unless compute_rate is
        True.

        :param anchors: A list of tuples, containing the anchor locations
        :param num_classes: The number of classes to predict
        :param img_dim: The original size of the network's input image
        :param return_loss: Kept for backward compatibility only
        :param compute_rate: If True, average over the time dimension before \
        applying the YOLO operations.
        """
        super(YOLOLayer, self).__init__(input_shape)
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        # self.ignore_thres = 0.5
        # self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCELoss()
        # self.obj_scale = 1
        # self.noobj_scale = 100
        self.img_dim = img_dim
        self.grid_size = 0  # grid size
        self.return_loss = return_loss
        self.compute_rate = compute_rate

    def get_output_shape(self, input_shape) -> Tuple:
        """
        Returns the shape of output, given an input to this layer

        :param input_shape: (channels, height, width)
        :return: (channelsOut, height_out, width_out)
        """
        return (self.num_classes + 5, self.grid_size, self.grid_size)

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size

        tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(g, 1).view(
            [1, 1, g, g]).type(tensor)
        self.grid_y = torch.arange(g).repeat(g, 1).t().view(
            [1, 1, g, g]).type(tensor)
        self.scaled_anchors = tensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors]
        )
        self.anchor_w = self.scaled_anchors[:, 0:1].view((1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view((1, self.num_anchors, 1, 1))

    def forward(self, x, targets=None, img_dim=None):

        # First of all, convert spiking input to rates
        if self.compute_rate:
            x = x.float().mean(0).unsqueeze(0)  # <-- please note!

        # Tensors for cuda support
        tensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        if img_dim is not None:
            self.img_dim = img_dim
        num_samples = x.size(0)
        grid_size = x.size(2)

        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes + 5,
                   grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = tensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + self.grid_x
        pred_boxes[..., 1] = y.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        if not self.return_loss:
            return output

        return output, 0  # for compatibility with original code
