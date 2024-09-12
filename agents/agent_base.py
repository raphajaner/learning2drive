from copy import deepcopy
import torch
from torch import nn
from agents.rl_modules import ActorMeanNet, CriticNet
from torch.distributions.normal import Normal
from torchrl.modules import TanhNormal


class BaseAgent(nn.Module):
    def __init__(self, config, envs):
        """ Base agent class. Based on https://github.com/vwxyzjn/cleanrl.
        Args:
            config: Config
            envs: Environments
        """
        super().__init__()
        self.envs = envs
        # Actor
        self.actor_mean = ActorMeanNet(
            fc_arch=config.rl.actor.fc_arch,
            action_offset=config.rl.actor.action_offset,
            std=config.rl.actor.action_init_std,
            bias_const=config.rl.actor.action_init_bias,
            layer_norm=config.rl.layer_norm
        )
        # Action is a trainable paramter
        if config.rl.anneal_actor_logstd:
            self.actor_logstd = nn.Parameter(torch.tensor([config.rl.init_logstd]), requires_grad=False)
        else:
            self.actor_logstd = nn.Parameter(torch.tensor([config.rl.init_logstd]))

        self.dist_head = TanhNormal if config.rl.distribution == 'TanhNormal' else Normal
        self.register_buffer('upscale_tanh', torch.tensor(config.rl.upscale_tanh))
        self.register_buffer('min_max_dist', torch.tensor(config.rl.min_max_dist))

        # Buffer for RPO
        self.register_buffer('rpo_alpha', torch.tensor(config.rl.rpo_alpha))

        # Critic
        self.critic = CriticNet(
            fc_arch=config.rl.critic.fc_arch,
            std=config.rl.critic.value_init_std,
            bias_const=config.rl.critic.value_init_bias,
            layer_norm=config.rl.layer_norm
        )

        # Perception module must be added by custom agent
        self.modality_encoder = None
        self.modality = None
        self.modality_transforms = nn.Sequential()
        self.feature_size = None

    @property
    def device(self):
        """ Get device of model
        Returns:
            device: Device of model
        """
        return next(self.actor_mean.parameters()).device

    def get_probs(self, mean, logstd):
        """ Sample from distribution.

        Args:
            mean: Mean of distribution
            logstd: Log std of distribution

        Returns:
            dist: Distribution
        """
        std = torch.exp(logstd)
        dist = self.dist_head(mean, std, min=self.min_max_dist[0], max=self.min_max_dist[1], upscale=self.upscale_tanh)
        return dist

    def get_features(self, x, hidden_state=False, done=None):
        """ Get features from perception module and concatenate with state
        Args:
            x: Dict of tensors
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            features: Concatenated features
            hidden_state: Hidden state of LSTM
        """
        x_transformed = self.modality_transforms(x[self.modality])
        modality_encoded = self.modality_encoder.forward(x_transformed)
        # print(f'mean: {modality_encoded.mean()}, std: {modality_encoded.std()}, min: {modality_encoded.min()}, max: {modality_encoded.max()}, mean_abs: {modality_encoded.abs().mean()}')
        features = torch.cat([
            x['state'],
            modality_encoded
        ], dim=1)
        # print(f'state mean : {x["state"].mean()}, state std: {x["state"].std()}, state min: {x["state"].min()}, state max: {x["state"].max()}, state mean_abs: {x["state"].abs().mean()}')
        return self._features(features, hidden_state, done)

    def _features(self, features, hidden_state, done):
        """ Get features from perception module and concatenate with state
        Args:
            features: Features
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            features: Concatenated features
            hidden_state: Hidden state of LSTM
        """
        return features, hidden_state

    def get_value(self, x, hidden_state=False, done=None):
        """ Get value from critic
        Args:
            x: Dict of tensors
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            value: Value
        """
        hidden, _ = self.get_features(x, hidden_state=hidden_state, done=done)
        return self.critic(hidden)

    def get_action_and_value(self, x, action=None, hidden_state=False, done=None, action_offset=None):
        """ Get action and value from actor and critic
        Args:
            x: Dict of tensors
            action: Action to take
            hidden_state: Hidden state of LSTM
            done: Done mask
            action_offset: Action offset
        Returns:
            action: Action
            log_prob: Log probability of action
            entropy: Entropy of action
            value: Value
            hidden_state: Hidden state of LSTM
        """
        features, hidden_state = self.get_features(x, hidden_state, done)

        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        if action is None:
            probs = self.get_probs(action_mean, action_logstd)
            action = probs.rsample()  # or rsample() for reparameterization trick
            if action_offset is not None:
                action_offset = torch.tensor(action_offset).to(self.device)
                action = action + action_offset.expand_as(action)
        else:
            # Robust PO: For the policy update, sample again to add stochasticity
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z.to(self.device)
            probs = self.get_probs(action_mean, action_logstd)

        # The probability of the action taken/sampled
        log_prob = probs.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(1)

        # Value
        value = self.critic(features)
        return action, log_prob, torch.zeros_like(log_prob), value, hidden_state

    def get_action(self, x, hidden_state=False, done=None, mode=True):
        """ Get action from actor
        Args:
            x: Dict of tensors
            hidden_state: Hidden state of LSTM
            done: Done mask
            mode: If true, return mode of distribution
        Returns:
            action: Action, not reparameterized
            hidden_state: Hidden state of LSTM
        """
        hidden, hidden_state = self.get_features(x, hidden_state, done)
        action_mean = self.actor_mean(hidden)
        probs = self.get_probs(action_mean, self.actor_logstd.expand_as(action_mean))
        if mode:
            action = probs.mode
        else:
            action = probs.sample()
        return action, hidden_state

    def get_initial_state(self, batch_size):
        """ Get initial state of LSTM
        Args:
            batch_size: Batch size
        Returns:
            hidden_state: Hidden state of LSTM
        """
        return False


class BaseDoubleAgent(BaseAgent):
    def __init__(self, config, envs):
        """ Base agent with two critics
        Args:
            config: Config
            envs: Environments
        """
        super().__init__(config, envs)
        self.modality_encoder = nn.ModuleDict()

    def get_features(self, x, hidden_state=False, done=None):
        """ Get features from perception module and concatenate with state
        Args:
            x: Dict of tensors
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            features: Concatenated features
            hidden_state: Hidden state of LSTM
        """
        x_transformed = self.modality_transforms(x[self.modality])
        modality_encoded_actor = self.modality_encoder['actor'].forward(x_transformed)
        modality_encoded_critic = self.modality_encoder['critic'].forward(x_transformed)
        features_actor = torch.cat([
            x['state'],
            modality_encoded_actor
        ], dim=1)
        features_critic = torch.cat([
            x['state'],
            modality_encoded_critic
        ], dim=1)
        features = dict(actor=features_actor, critic=features_critic)
        return self._features(features, hidden_state, done)

    def get_value(self, x, hidden_state=False, done=None):
        """ Get value from critic
        Args:
            x: Dict of tensors
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            value: Value
        """
        hidden, _ = self.get_features(x, hidden_state=hidden_state, done=done)
        return self.critic(hidden['critic'])

    def get_action_and_value(self, x, action=None, hidden_state=False, done=None):
        """ Get action and value from actor and critic
        Args:
            x: Dict of tensors
            action: Action to take
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            action: Action
            log_prob: Log probability of action
            entropy: Entropy of action
            value: Value
            hidden_state: Hidden state of LSTM
        """
        features, hidden_state = self.get_features(x, hidden_state, done)

        action_mean = self.actor_mean(features['actor'])
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        if action is None:
            probs = self.dist_head(action_mean, action_std, min=self.min_max_dist[0], max=self.min_max_dist[1],
                                   upscale=self.upscale_tanh)
            action = probs.sample()
        else:
            # Robust PO: For the policy update, sample again to add stochasticity
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean + z.to(self.device)
            probs = self.dist_head(action_mean, action_std, min=self.min_max_dist[0], max=self.min_max_dist[1],
                                   upscale=self.upscale_tanh)

        # The probability of the action taken/sampled
        log_prob = probs.log_prob(action)
        if log_prob.dim() > 1:
            log_prob = log_prob.sum(1)

        # Value
        value = self.critic(features['critic'])

        return action, log_prob, torch.zeros_like(log_prob), value, hidden_state

    def get_action(self, x, hidden_state=False, done=None, mode=True):
        """ Get action from actor
        Args:
            x: Dict of tensors
            hidden_state: Hidden state of LSTM
            done: Done mask
            mean: Return mean action
        Returns:
            action: Action
            hidden_state: Hidden state of LSTM
        """
        hidden, hidden_state = self.get_features(x, hidden_state, done)
        action_mean = self.actor_mean(hidden['actor'])
        probs = self.get_probs(action_mean, self.actor_logstd.expand_as(action_mean))
        if mode:
            action = probs.mode
        else:
            action = probs.sample()
        return action, hidden_state


class BaseAgentLSTM(BaseAgent):
    def __init__(self, config, envs):
        """ Base agent with LSTM
        Args:
            config: Config
            envs: Environments
        """
        super().__init__(config, envs)
        self.lstm_hidden_size = config.rl.lstm.hidden_size
        self.lstm_num_layers = config.rl.lstm.num_layers
        dim_state = envs.observation_space.spaces['state'].shape[1]
        # Memory
        self.lstm = nn.LSTM(config.rl.hidden_out_dim + dim_state, self.lstm_hidden_size, self.lstm_num_layers)

    def _features(self, features, hidden_state, done):
        """ Get features from perception module and concatenate with state
        Args:
            features: Features from perception module
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            features: Concatenated features
            hidden_state: Hidden state of LSTM
        """
        # LSTM logic
        return self._lstm(self.lstm, features, hidden_state, done)

    @staticmethod
    def _lstm(lstm, features, hidden_state, done):
        """ LSTM logic
        Args:
            features: Features from perception module
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            features: Concatenated features
            hidden_state: Hidden state of LSTM
        """
        batch_size = hidden_state[0].shape[1]
        features = features.reshape((-1, batch_size, lstm.input_size))
        done = done.reshape((-1, batch_size))
        hidden = []
        for i, (h, d) in enumerate(zip(features, done)):
            h, hidden_state = lstm(
                h.unsqueeze(0),  # Adds sequence length L=1
                (
                    (1 - d.int()).view(1, -1, 1) * hidden_state[0],
                    (1 - d.int()).view(1, -1, 1) * hidden_state[1],
                ),
            )
            hidden += [h]
        # Outputs each intermediate hidden state when x is a sequence
        # h is the LSTM hidden state; hidden_state state contains also the LSTM cell state
        features = torch.flatten(torch.cat(hidden), 0, 1)
        return features, hidden_state

    def get_initial_state(self, batch_size):
        """ Get initial state of LSTM
        Args:
            batch_size: Batch size
        Returns:
            initial_lstm_state: Initial state of LSTM
        """
        initial_lstm_state = (
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
            torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device)
        )
        return initial_lstm_state


class BaseDoubleAgentLSTM(BaseDoubleAgent):
    def __init__(self, config, envs):
        """ Base agent with LSTM
        Args:
            config: Config
            envs: Environments
        """
        super().__init__(config, envs)
        self.lstm_hidden_size = config.rl.lstm.hidden_size
        self.lstm_num_layers = config.rl.lstm.num_layers
        dim_state = envs.observation_space.spaces['state'].shape[1]
        # Memory
        self.lstm = nn.LSTM(config.rl.hidden_out_dim + dim_state, self.lstm_hidden_size, self.lstm_num_layers)
        # for name, param in self.lstm.named_parameters():
        #     if "bias" in name:
        #         nn.init.constant_(param, 0)
        #     elif "weight" in name:
        #         nn.init.orthogonal_(param, 1.0)
        self.lstm = nn.ModuleDict({
            'actor': self.lstm,
            'critic': deepcopy(self.lstm)
        })

    def _features(self, features, hidden_state, done):
        """ Get features from perception module and concatenate with state
        Args:
            features: Features from perception module
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            features: Concatenated features
            hidden_state: Hidden state of LSTM
        """
        # LSTM logic
        features_actor, hidden_state_actor = self._lstm(self.lstm['actor'], features['actor'],
                                                        (hidden_state[0]['actor'], hidden_state[1]['actor']),
                                                        done)
        features_critic, hidden_state_critic = self._lstm(self.lstm['critic'], features['critic'],
                                                          (hidden_state[0]['critic'], hidden_state[1]['critic']),
                                                          done)
        features = {
            'actor': features_actor,
            'critic': features_critic
        }
        hidden_state = (
            {
                'actor': hidden_state_actor[0],
                'critic': hidden_state_critic[0]
            },
            {
                'actor': hidden_state_actor[1],
                'critic': hidden_state_critic[1]
            }
        )
        return features, hidden_state

    @staticmethod
    def _lstm(lstm, features, hidden_state, done):
        """ LSTM logic
        Args:
            features: Features from perception module
            hidden_state: Hidden state of LSTM
            done: Done mask
        Returns:
            features: Concatenated features
            hidden_state: Hidden state of LSTM
        """
        batch_size = hidden_state[0].shape[1]

        features = features.reshape((-1, batch_size, lstm.input_size))
        done = done.reshape((-1, batch_size))
        hidden = []
        for i, (h, d) in enumerate(zip(features, done)):
            h, hidden_state = lstm(
                h.unsqueeze(0),  # Adds sequence length L=1
                (
                    (1 - d.int()).view(1, -1, 1) * hidden_state[0],
                    (1 - d.int()).view(1, -1, 1) * hidden_state[1],
                ),
            )
            hidden += [h]
        # Outputs each intermediate hidden state when x is a sequence
        # h is the LSTM hidden state; hidden_state state contains also the LSTM cell state
        features = torch.flatten(torch.cat(hidden), 0, 1)
        return features, hidden_state

    def get_initial_state(self, batch_size):
        """ Get initial state of LSTM
        Args:
            batch_size: Batch size
        Returns:
            initial_lstm_state: Initial state of LSTM
        """
        initial_lstm_state = (
            {
                'actor': torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
                'critic': torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
            },
            {
                'actor': torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
                'critic': torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=self.device),
            }
        )
        return initial_lstm_state
