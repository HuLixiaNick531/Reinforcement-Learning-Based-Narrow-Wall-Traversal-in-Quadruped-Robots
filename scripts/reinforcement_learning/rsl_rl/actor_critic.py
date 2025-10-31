# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_state_dim: int | None = None,  # devide state/sensor, None means all state
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        super().__init__()
        act = resolve_nn_activation(activation)

        if actor_state_dim is None:
            actor_state_dim = num_actor_obs
            actor_scan_dim = 0
        else:
            actor_scan_dim = num_actor_obs - actor_state_dim
            if actor_scan_dim < 0:
                raise ValueError(
                    f"actor_state_dim ({actor_state_dim}) > num_actor_obs ({num_actor_obs}) — check your cfg."
                )

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.actor_state_dim = actor_state_dim
        self.actor_scan_dim = actor_scan_dim

        # divide state/sensor
        first_actor_dim = actor_hidden_dims[0] if len(actor_hidden_dims) > 0 else 256
        self.state_encoder = nn.Sequential(
            nn.Linear(actor_state_dim, first_actor_dim),
            act,
        )
        if actor_scan_dim > 0:
            scan_width = max(first_actor_dim // 2, 64)
            self.scan_encoder = nn.Sequential(
                nn.Linear(actor_scan_dim, scan_width),
                act,
            )
            fused_dim = first_actor_dim + scan_width
        else:
            self.scan_encoder = None
            fused_dim = first_actor_dim

        # add actor layers behind
        actor_layers: list[nn.Module] = []
        in_dim = fused_dim
        for i, h_dim in enumerate(actor_hidden_dims):
            actor_layers.append(nn.Linear(in_dim, h_dim))
            actor_layers.append(act)
            in_dim = h_dim
        
        # output layer
        actor_layers.append(nn.Linear(in_dim, num_actions))
        self.actor = nn.Sequential(*actor_layers)

        # critic head
        critic_layers: list[nn.Module] = []
        in_dim = num_critic_obs
        for i, h_dim in enumerate(critic_hidden_dims):
            critic_layers.append(nn.Linear(in_dim, h_dim))
            critic_layers.append(act)
            in_dim = h_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

        # action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

        # # shape check printout
        # print(
        #     f"[DEBUG] init | obs={num_actor_obs} (state={actor_state_dim}, scan={actor_scan_dim}), "
        #     f"[DEBUG] actor_dims={actor_hidden_dims}, critic_dims={critic_hidden_dims}"
        # )
        # with torch.no_grad():
        #     dummy_actor_in = torch.randn(2, num_actor_obs)
        #     dummy_critic_in = torch.randn(2, num_critic_obs)
        #     actor_feat = self._encode_actor_obs(dummy_actor_in)
        #     actor_out = self.actor(actor_feat)
        #     critic_out = self.critic(dummy_critic_in)
        #     print(f"[DEBUG] actor encoder out shape: {actor_feat.shape}")
        #     print(f"[DEBUG] actor output shape:      {actor_out.shape}")
        #     print(f"[DEBUG] critic output shape:     {critic_out.shape}")

    def _encode_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, num_actor_obs]
        state = obs[..., : self.actor_state_dim]
        state_feat = self.state_encoder(state)
        if self.actor_scan_dim > 0:
            scan = obs[..., self.actor_state_dim : self.actor_state_dim + self.actor_scan_dim]
            scan_feat = self.scan_encoder(scan)
            feat = torch.cat([state_feat, scan_feat], dim=-1)
        else:
            feat = state_feat
        return feat

    def update_distribution(self, observations: torch.Tensor) -> None:
        feat = self._encode_actor_obs(observations)
        mean = self.actor(feat)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        feat = self._encode_actor_obs(observations)
        return self.actor(feat)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(critic_observations)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        return

    def get_hidden_states(self):
        return None, None

    def set_hidden_states(self, *args, **kwargs):
        return

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True