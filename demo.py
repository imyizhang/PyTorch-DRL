#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import pytorch_drl as drl


def parse_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
            '--env',
            default='CartPole-v0',
            type=str,
            help='environment',
    )
    args = parser.parse_args()
    return args

def main(args):
    # net
    actor = drl.net.QNet()
    critic = drl.net.QNet()
    # agent
    agent = drl.agent.DQNAgent(
        actor,
        critic,
    )
    # env
    env = drl.env.GymEnv(args.env)
    # trainer
    trainer = drl.trainer.OffPolicyTrainer()
    # run
    trainer(agent, env)

if __name__ == '__main__':
    args = parse_args()
    main(args)
