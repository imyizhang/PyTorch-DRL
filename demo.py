#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import pytorch_drl as drl


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
            '--device',
            default='cpu',
            type=str,
            help='device',
    )
    parser.add_argument(
            '--env',
            default='CartPole-v0',
            type=str,
            help='environment',
    )
    args = parser.parse_args()
    return args

def main(args):
    # env
    env = drl.env.GymEnv(
        args.device,
        args.env
    )
    # net
    actor = drl.net.QNetActor(
        env.state_dim,
        env.action_dim
    )
    critic = drl.net.QNetCritic(
        env.state_dim,
        env.action_dim
    )
    # agent
    agent = drl.agent.DQNAgent(
        args.device,
        actor,
        critic,
    )
    # trainer
    trainer = drl.trainer.OffPolicyTrainer(env, agent)
    # run
    return trainer()

if __name__ == '__main__':
    args = parse_args()
    main(args)
