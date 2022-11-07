import torch
import torch.nn as nn
import torch.nn.functional as F

from data_io.weights import enable_weight_saving
from learning.modules.auxiliary_objective_base import AuxiliaryObjective
from learning.meters_and_metrics.moving_average import MovingAverageMeter
from learning.meters_and_metrics.meter_server import log_value

from visualization import Presenter
from transformations import pos_m_to_px

# Multiply with a hyperparameter and add it into the supervised stage loss
class DistanceLoss(AuxiliaryObjective):
    def __init__(self, name, *inputs):
        super(DistanceLoss, self).__init__(name, *inputs)
        self.MSELoss = nn.MSELoss()
        self.MSELoss_sum = nn.MSELoss(reduction='sum')

    
    def cuda(self, device=None):
        AuxiliaryObjective.cuda(self,device)
        self.MSELoss.cuda(0)
        self.MSELoss_sum.cuda(0)
        return self
    
    
    #TODO: Find out the opts.mse_sum boolean
    def get_value_loss_from_start(self, traj, predicted_value, ended, norm_value=True, threshold=5):
        """
        traj: p_pm_t in the paper
        predicted_value: y_pm_t - the target will be 0 at the beginning and closer to 1 as the agent approaches the goa
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            value_target.append(dist_improved_from_start)

            if dist <= 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            # we will average the loss according to number of not 'ended', and use reduction='sum' for MSELoss
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)

    def forward(self, traj, predicted_value, ended, norm_value=True, threshold=5):
        """
        This loss forces the agent to estimate how good is the current state, i.e. how far away I am from the goal?
        """
        value_target = []
        for i, _traj in enumerate(traj):
            original_dist = _traj['distance'][0]
            dist = _traj['distance'][-1]
            dist_improved_from_start = (original_dist - dist) / original_dist

            dist_improved_from_start = 0 if dist_improved_from_start < 0 else dist_improved_from_start

            value_target.append(dist_improved_from_start)

            if dist < 3.0:  # if we are less than 3m away from the goal
                value_target[-1] = 1

            # if ended, let us set the target to be the value so that MSE loss for that sample with be 0
            if ended[i]:
                value_target[-1] = predicted_value[i].detach()

        value_target = torch.FloatTensor(value_target).to(self.device)

        if self.opts.mse_sum:
            return self.MSELoss_sum(predicted_value.squeeze(), value_target) / sum(1 - ended).item()
        else:
            return self.MSELoss(predicted_value.squeeze(), value_target)
        