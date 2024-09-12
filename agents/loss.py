import torch
from torch import nn


class PPOLoss(torch.nn.Module):
    """ PPO Loss function. Based on https://github.com/vwxyzjn/cleanrl."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register as buffer
        self.register_buffer('clip_coef', torch.tensor(config.rl.clip_coef))
        self.register_buffer('ent_coef', torch.tensor(config.rl.ent_coef))
        self.register_buffer('vf_coef', torch.tensor(config.rl.vf_coef))

        # Use jump condition with care when compiling the model, probably they will get loaded only once
        self.clip_vloss = config.rl.clip_vloss
        self.norm_adv = config.rl.norm_adv

    def forward(self, b_returns, b_values, b_advantages, b_logprobs, newlogprob, entropy, newvalue):
        logratio = newlogprob - b_logprobs
        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            # approx_kl_batch += approx_kl
            # clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]
            clipfracs = (ratio - 1.0).abs() > self.clip_coef
        mb_advantages = b_advantages

        if self.norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        # Value loss
        newvalue = newvalue.view(-1)

        if self.clip_vloss:
            v_loss_unclipped = (newvalue - b_returns) ** 2
            v_clipped = b_values + torch.clamp(
                newvalue - b_values,
                -self.clip_coef,
                self.clip_coef,
            )
            v_loss_clipped = (v_clipped - b_returns) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
        return loss, pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs

    def grad_step(self, config, agent, optimizer, b_returns, b_values, b_advantages, b_logprobs, newlogprob, entropy,
                  newvalue):
        loss, pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs = self.forward(
            b_returns, b_values, b_advantages, b_logprobs, newlogprob, entropy, newvalue
        )

        optimizer.zero_grad()
        loss.backward()
        grads = nn.utils.clip_grad_norm_(agent.parameters(), config.rl.max_grad_norm)
        optimizer.step()
        return loss.detach(), pg_loss, v_loss, entropy_loss, approx_kl, old_approx_kl, clipfracs, grads
