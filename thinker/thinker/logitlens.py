import torch
from thinker.actor_net import sample, DRCNet
from torch.nn.functional import relu, softmax, kl_div
from typing import NamedTuple

class DRCTickLogitLens:
    def __init__(self, drc_net: DRCNet):
        self.drc_net = drc_net
    
    def activate(self):
        self.drc_net.record_core_output = True

    def deactivate(self):
        self.drc_net.record_core_output = False

    def get_logits(self, env_out: NamedTuple) -> torch.tensor:
        x = self.drc_net.normalize(env_out.real_states.float())
        x = torch.flatten(x, 0, 1)
        x_enc = self.drc_net.encoder(x)
        core_output = self.drc_net.core.output
        N, B = core_output.shape[:2]
        core_output = torch.flatten(self.drc_net.core.output, 0, 1)
        core_output = torch.cat([torch.cat([x_enc]*N,dim=0), core_output], dim=1)
        core_output = torch.flatten(core_output, 1)
        final_out = relu(self.drc_net.final_layer(core_output))
        pri_logits = self.drc_net.policy(final_out)
        logits = pri_logits.view(N,B,5)
        return logits
    
    def sample_from_tick(self, env_out: NamedTuple, tick: int, greedy: bool =False) -> torch.tensor:
        assert (tick > -1 and tick < self.drc_net.core.tran_t), f"Please enter a valid tick number betwen 0 and {self.drc_net.core.tran_t}"
        logits = self.get_logits(env_out)[tick,:,:]
        sampled_action = sample(logits, greedy)
        return sampled_action
    
    def get_avg_kl_div(self, env_out: NamedTuple, ref_tick: int) -> float:
        logits = self.get_logits(env_out)
        log_probs = softmax(logits, dim=-1).log()
        return kl_div(input=log_probs[ref_tick,:,:], target=log_probs[-1,:,:], log_target=True, reduction="sum").item() / (logits.shape[1]*logits.shape[2])