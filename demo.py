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
    actor = drl.net.QNet()
    critic = drl.net.QNet()
    agent = drl.agent.DQNAgent(
        args.device,
        args.buffer_capacity,
        args.batch_size,
        args.sync_step,
        args.discount_factor,
        actor,
        critic,
    )
    env = drl.env.GymEnv(args.env)
    trainer = drl.trainer.OffPolicyTrainer()
    trainer(agent, env)

if __name__ == '__main__':
    args = parse_args()
    main(args)
