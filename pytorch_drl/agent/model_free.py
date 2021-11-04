#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random

from .base_agent import BaseAgent


class DQNAgent(BaseAgent):

    def __init__(
        self,
        device,
        buffer_capacity,
        batch_size,
        sync_step,
        discount_factor,
        actor,
        critic,
    ):
        super().__init__(
            device,
            buffer_capacity,
            batch_size,
            sync_step,
            discount_factor,
            actor,
            critic,
        )

    def act(self, state):
        # explore
        if random.random() < self.exploration_rate:
            action_idx = rd.randint(self.action_dim)
        # exploit
        else:
            with torch.no_grad():
                states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
                action = self.actor(states)[0]
                a_int = action.argmax(dim=0).detach().cpu().numpy()
        return a_int

    def learn(self):
        state, action, reward, done, next_state = self.recall()



        # compute Q(s, a)
        Q = self.actor(state).gather(1, action)

        # compute V(s') := max_{a'} Q(s', a')
        with torch.no_grad():
            V = torch.zeros(self.batch_size, device=self.device)
            V[non_final_mask] = self.critic(non_final_next_states).max(1)[0].detach()
        # compute expected Q(s, a) := r(s, a) + gamma * V(s')
        Q_expected = reward + self.gamma * V


        # optimize the model
        self.actor_optim.zero_grad()
        loss = self.criterion(Q, expected_Q.unsqueeze(1))
        loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optim.step()
