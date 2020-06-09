from __future__ import division
from setproctitle import setproctitle as ptitle
import torch
from environment import atari_env
from utils import setup_logger
from model import A3Cff
from player_util import Agent
from torch.autograd import Variable
import time
import os
import logging


def test(args, shared_model, optimizer, env_conf):
    ptitle('Test Agent')
    gpu_id = args.gpu_ids[-1]
    log = {}
    setup_logger('{}_log'.format(args.env), r'{0}{1}_log'.format(
        args.log_dir, args.env))
    log['{}_log'.format(args.env)] = logging.getLogger('{}_log'.format(
        args.env))
    d_args = vars(args)
    for k in d_args.keys():
        log['{}_log'.format(args.env)].info('{0}: {1}'.format(k, d_args[k]))
    if not os.path.exists(args.save_model_dir):
        os.mkdir(args.save_model_dir)
    if args.seed:
        torch.manual_seed(args.seed)
        if gpu_id >= 0:
            torch.cuda.manual_seed(args.seed)
    env = atari_env(args.env, env_conf, args)
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    player = Agent(None, env, args, None)
    player.gpu_id = gpu_id
    player.model = A3Cff(player.env.observation_space.shape[0],
                           player.env.action_space)

    player.state = player.env.reset()
    player.eps_len += 2
    player.state = torch.from_numpy(player.state).float()
    if gpu_id >= 0:
        with torch.cuda.device(gpu_id):
            player.model = player.model.cuda()
            player.state = player.state.cuda()
    flag = True
    max_score = 0
    
    while True:
        p = optimizer.param_groups[0]['params'][0]
        step = optimizer.state[p]['step']
        
        if flag:
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.model.load_state_dict(shared_model.state_dict())
            else:
                player.model.load_state_dict(shared_model.state_dict())
            player.model.eval()
            flag = False
        
        with torch.no_grad():
            if args.robust:
                #player.action_test_losses(args.epsilon_end)
                lin_coeff = min(1, (1.5*int(step)+1)/(args.total_frames/args.num_steps))
                epsilon = lin_coeff*args.epsilon_end
                player.action_train(epsilon)
            else:
                player.action_train()
                #player.action_test_losses()
            
        reward_sum += player.noclip_reward

        if player.done and not player.info:
            state = player.env.reset()
            player.eps_len += 2
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
        elif player.info:
            # calculate losses for tracking
            R = torch.zeros(1, 1)
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    R = R.cuda()
            player.values.append(R)
            gae = torch.zeros(1, 1)
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    gae = gae.cuda()
            R = Variable(R)
            
            standard_loss = 0
            worst_case_loss = 0
            value_loss = 0
            entropy = 0
            
            for i in reversed(range(len(player.rewards))):
                R = args.gamma * R + player.rewards[i]
                advantage = R - player.values[i]

                value_loss += 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = player.rewards[i] + args.gamma * \
                    player.values[i + 1].data - player.values[i].data
                
                gae = gae * args.gamma * args.tau + delta_t
                if args.robust:
                    if advantage >= 0:
                        worst_case_loss += - player.min_log_probs[i] * Variable(gae)
                    else:
                        worst_case_loss += - player.max_log_probs[i] * Variable(gae)
                        
                standard_loss += -player.log_probs[i] * Variable(gae)
                entropy += player.entropies[i]
            
            standard_loss = standard_loss/len(player.rewards)
            worst_case_loss = worst_case_loss/len(player.rewards)
            value_loss = value_loss/len(player.rewards)
            entropy = entropy/len(player.rewards)
            player.clear_actions()
            
            flag = True
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            log['{}_log'.format(args.env)].info(
                ("Time {0}, steps {1}/{2}, ep reward {3}, ep length {4}, reward mean {5:.3f} \n"+
                "Losses: Policy:{6:.3f}, Worst case: {7:.3f}, Value: {8:.3f}, Entropy: {9:.3f}").
                format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)),
                    int(step), args.total_frames/args.num_steps, reward_sum, player.eps_len, reward_mean,
                      float(standard_loss), float(worst_case_loss), float(value_loss), float(entropy)))

            if args.save_max and reward_sum >= max_score:
                max_score = reward_sum
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_best.pt'.format(
                            args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}_best.pt'.format(
                        args.save_model_dir, args.env))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.eps_len += 2
            
            #stop after total steps gradient updates have passed
            if step >= args.total_frames/args.num_steps:
                if gpu_id >= 0:
                    with torch.cuda.device(gpu_id):
                        state_to_save = player.model.state_dict()
                        torch.save(state_to_save, '{0}{1}_last.pt'.format(
                            args.save_model_dir, args.env))
                else:
                    state_to_save = player.model.state_dict()
                    torch.save(state_to_save, '{0}{1}_last.pt'.format(
                        args.save_model_dir, args.env))
                return
            
            time.sleep(10)
            player.state = torch.from_numpy(state).float()
            if gpu_id >= 0:
                with torch.cuda.device(gpu_id):
                    player.state = player.state.cuda()
