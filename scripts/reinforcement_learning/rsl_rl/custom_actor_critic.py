import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
from gymnasium import spaces


# CNN + MLP
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, input_shape=(1, 45), feature_dim=128, dropout=0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, input_shape[-1]))
            out_dim = self.cnn(dummy).shape[-1]

        self.fc = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        if x.dim() == 2:  # (B, 45)
            x = x.unsqueeze(1)
        x = self.cnn(x)
        return self.fc(x)

class DiagGaussianActor(nn.Module):
    def __init__(self, feature_dim, act_dim, hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(feature_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.LayerNorm(hidden[1]),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden[1], act_dim)
        self.std_head = nn.Linear(hidden[1], act_dim)

        # init params
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.0)

    def dist(self, features):
        h = self.backbone(features)
        mu = self.mu_head(h)
        log_std = torch.clamp(self.std_head(h), -4.0, 1.0)
        std = log_std.exp()
        return D.Independent(D.Normal(mu, std), 1)


# value function network
class Critic(nn.Module):
    def __init__(self, feature_dim, hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.v_net = nn.Sequential(
            nn.Linear(feature_dim, hidden[0]),
            nn.LayerNorm(hidden[0]),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.LayerNorm(hidden[1]),
            nn.SiLU(),
            nn.Linear(hidden[1], 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, features):
        return self.v_net(features).squeeze(-1)


# Actor-Critic
class CustomActorCritic(nn.Module):
    def __init__(self, env, cnn_feat_dim=128, actor_hidden=(256,128), critic_hidden=(256,128)):
        super().__init__()

        if isinstance(env.observation_space, spaces.Dict):
            pol_shape = env.observation_space["policy"].shape 
            cri_shape = env.observation_space["critic"].shape

            def parse_CL(shape):
                if len(shape) == 2:      # (C, L)
                    return shape[0], shape[1]
                elif len(shape) == 1:    # (L,)
                    return 1, shape[0]
                else: 
                    C = int(np.prod(shape[:-1]))
                    L = shape[-1]
                    return C, L

            self.pol_C, self.pol_L = parse_CL(pol_shape)
            self.cri_C, self.cri_L = parse_CL(cri_shape)
            self._is_dict = True
        else:
            raise ValueError("Only continuous action spaces supported.")

        self.policy_encoder = CNNFeatureExtractor(self.pol_C, (self.pol_C, self.pol_L), cnn_feat_dim)
        self.critic_encoder = CNNFeatureExtractor(self.cri_C, (self.cri_C, self.cri_L), cnn_feat_dim)

        act_shape = env.action_space.shape
        self.act_dim = act_shape[-1] if isinstance(act_shape, tuple) else int(np.prod(act_shape))

        self.register_buffer("act_low",  torch.as_tensor(env.action_space.low,  dtype=torch.float32).view(-1))
        self.register_buffer("act_high", torch.as_tensor(env.action_space.high, dtype=torch.float32).view(-1))

        if not torch.isfinite(self.act_low).all() or not torch.isfinite(self.act_high).all():
            print("[WARN] Action bounds ±inf; fallback [-1,1].")
            self.act_low  = torch.full((self.act_dim,), -1.0)
            self.act_high = torch.full((self.act_dim,),  1.0)

        # --- Actor + Critic ---
        self.actor = DiagGaussianActor(cnn_feat_dim, self.act_dim, actor_hidden)
        self.critic = Critic(cnn_feat_dim, critic_hidden)

    # def _flatten(self, x): return x.view(x.shape[0], -1)
    def _as_seq(self, x, C_expected, L_expected):

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        if x.dim() == 2:
            if x.shape[-1] != L_expected:
                raise RuntimeError(f"Got (B,{x.shape[-1]}), expected L={L_expected}.")
            return x
        elif x.dim() >= 3:
            B = x.shape[0]; L = x.shape[-1]
            C = int(np.prod(x.shape[1:-1]))
            x = x.view(B, C, L)
            if L != L_expected:
                raise RuntimeError(f"Got L={L}, expected L={L_expected}.")
            return x
        else:
            raise RuntimeError(f"Unsupported obs dim: {x.shape}")

    def _get_pol_obs(self, obs):
        x = obs["policy"] if self._is_dict else obs
        return self._as_seq(x, self.pol_C, self.pol_L)

    def _get_cri_obs(self, obs):
        x = obs["critic"] if self._is_dict else obs
        return self._as_seq(x, self.cri_C, self.cri_L)
    def _get_pol_obs(self, obs): return self._flatten(obs["policy"]) if self._is_dict else self._flatten(obs)
    def _get_cri_obs(self, obs): return self._flatten(obs["critic"]) if self._is_dict else self._flatten(obs)
    def _squash(self, a_raw):
        a = torch.tanh(a_raw)
        scale = (self.act_high - self.act_low) / 2
        mean = (self.act_high + self.act_low) / 2
        return mean + scale * a

    @torch.no_grad()
    def act(self, obs):
        pol = self._get_pol_obs(obs)
        cri = self._get_cri_obs(obs)
        feat_p = self.policy_encoder(pol)
        feat_c = self.critic_encoder(cri)
        dist = self.actor.dist(feat_p)
        a_raw = dist.rsample()
        a = self._squash(a_raw)
        logp = dist.log_prob(a_raw)
        v = self.critic(feat_c)
        return a, logp, v

    @torch.no_grad()
    def act_inference(self, obs):
        pol = self._get_pol_obs(obs)
        feat_p = self.policy_encoder(pol)
        mu = self.actor.backbone(feat_p)
        mu = self.actor.mu_head(mu)
        return self._squash(mu)

    def evaluate(self, obs, actions):
        pol = self._get_pol_obs(obs)
        cri = self._get_cri_obs(obs)
        feat_p = self.policy_encoder(pol)
        feat_c = self.critic_encoder(cri)
        dist = self.actor.dist(feat_p)
        mid, half = (self.act_high + self.act_low)/2, (self.act_high - self.act_low)/2 + 1e-8
        a_raw_approx = (actions - mid) / half
        logp = dist.log_prob(a_raw_approx)
        ent = dist.base_dist.entropy().sum(-1)
        v = self.critic(feat_c)
        return logp, ent, v