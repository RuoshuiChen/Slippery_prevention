"""
Wrapper around a gym env that provides convenience functions
"""

import gym
from mjrl.policies.ncp_network import NCPNetwork
from mjrl.policies.rnn_network import RNNNetwork
import numpy as np
import pickle
import copy


class EnvSpec(object):
    def __init__(self, obs_dim, act_dim, horizon):
        self.observation_dim = obs_dim
        self.action_dim = act_dim
        self.horizon = horizon


class GymEnv(object):
    def __init__(self, env, env_kwargs=None,
                 obs_mask=None, act_repeat=1, 
                 *args, **kwargs):
    
        # get the correct env behavior
        if type(env) == str:
            env = gym.make(env)
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        self.env = env
        self.env_id = env.spec.id
        self.act_repeat = act_repeat
        try:
            self._horizon = env.spec.max_episode_steps  # max_episode_steps is defnied in the __init__.py file (under )
        except AttributeError:
            self._horizon = env.spec._horizon
        assert self._horizon % act_repeat == 0
        self._horizon = self._horizon // self.act_repeat

        try:
            self._action_dim = self.env.env.action_dim
        except AttributeError:
            self._action_dim = self.env.action_space.shape[0]

        try:
            self._observation_dim = self.env.env.obs_dim
        except AttributeError:
            self._observation_dim = self.env.observation_space.shape[0]

        # Specs
        self.spec = EnvSpec(self._observation_dim, self._action_dim, self._horizon)

        # obs mask
        self.obs_mask = np.ones(self._observation_dim) if obs_mask is None else obs_mask

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self, seed=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model(seed=seed)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset()
    
    def reset4Koopman(self, seed=None, ori=None, init_pos=None, init_vel=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.env.reset_model4Koopman(seed=seed, ori = ori, init_pos = init_pos, init_vel = init_vel)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset_model4Koopman(ori = ori, init_pos = init_pos, init_vel = init_vel)

    def reset_model(self, seed=None):
        # overloading for legacy code
        return self.reset(seed)

    def step(self, action):
        action = action.clip(self.action_space.low, self.action_space.high)
        # type(action_space) -> <class 'gym.spaces.box.Box'>
        # self.action_space.low -> numpy.ndarray(lowest boundary)
        # self.action_space.high -> numpy.ndarray(highest boundary)
        if self.act_repeat == 1: 
            obs, cum_reward, done, ifo = self.env.step(action)  # the system dynamics is defined in each separate env python file
            # if(ifo['goal_achieved']):
            #     print("done: ", ifo)    
            # Run one timestep of the environment’s dynamics.
        else:
            cum_reward = 0.0
            for i in range(self.act_repeat):
                obs, reward, done, ifo = self.env.step(action) # the actual operations can be found in the env files
                # seems done is always set to be False
                cum_reward += reward
                if done: break
        return self.obs_mask * obs, cum_reward, done, ifo

    def render(self):
        try:
            self.env.env.mujoco_render_frames = True
            self.env.env.mj_render()
        except:
            self.env.render()

    def set_seed(self, seed=123):
        try:
            self.env.seed(seed)
        except AttributeError:
            self.env._seed(seed)

    def get_obs(self):
        try:
            return self.obs_mask * self.env.env.get_obs()
        except:
            return self.obs_mask * self.env.env._get_obs()

    def get_env_infos(self):
        try:
            return self.env.env.get_env_infos()
        except:
            return {}

    # ===========================================
    # Trajectory optimization related
    # Envs should support these functions in case of trajopt

    def get_env_state(self):
        try:
            return self.env.env.get_env_state()
        except:
            raise NotImplementedError

    def set_env_state(self, state_dict):
        try:
            self.env.env.set_env_state(state_dict)
        except:
            raise NotImplementedError

    def real_env_step(self, bool_val):
        try:
            self.env.env.real_step = bool_val
        except:
            raise NotImplementedError

    # ===========================================

    def visualize_policy(self, policy, policy_name, ori, init_pos, init_vel, horizon=1000, num_episodes=1, mode='exploration', **kwargs):  #test by generating the new observation data
        task = self.env_id.split('-')[0]
        success_threshold = 20 if task == 'pen' else 25
        episodes = []
        total_score = 0.0
        for ep in range(num_episodes):
            print("Episode %d" % ep)
            o = self.reset4Koopman(seed = None, ori = ori, init_pos = init_pos, init_vel = init_vel)
            # o = self.reset_model()
            # hand_vel = self.env.get_hand_vel()
            d = False
            t = 0
            score = 0.0
            episode_data = {
                'init_state_dict': copy.deepcopy(self.get_env_state()),  # set the initial states
                'actions': [],
                'observations': [],
                # 'handVelocity': [],
                'rewards': [],
                'goal_achieved': []
            }
            if isinstance(policy.model, NCPNetwork):
                hidden_state = np.zeros((1, policy.model.rnn_cell.state_size))
            if isinstance(policy.model, RNNNetwork):
                hidden_state = (np.zeros((1, 1, policy.model.rnn_cell.hidden_size)), np.zeros((1, 1, policy.model.rnn_cell.hidden_size)))
            while t < horizon and d is False:
                # episode_data['handVelocity'].append(hand_vel)
                episode_data['observations'].append(o)
                if isinstance(policy.model, NCPNetwork):
                    a = policy.get_action(o, hidden_state)
                    hidden_state = a[1]['hidden_state']
                elif isinstance(policy.model, RNNNetwork):
                    a = policy.get_action(o, hidden_state)
                    hidden_state = a[1]['hidden_state']
                else:
                    a = policy.get_action(o)
                a = a[0] if mode == 'exploration' else a[1]['evaluation']
                o, r, d, goal_achieved = self.step(a)
                # hand_vel = self.env.get_hand_vel()
                episode_data['actions'].append(a)
                episode_data['rewards'].append(r)
                episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                score = score + r
                if (not kwargs['record']):
                    self.render()
                t = t+1
            episodes.append(copy.deepcopy(episode_data))
            total_score += score
            print("Episode score = %f, Success = %d" % (score, sum(episode_data['goal_achieved']) > success_threshold))
        print("Average score = %f" % (total_score / num_episodes))
        successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
        print("Success rate = %f" % (len(successful_episodes) / len(episodes)))
        if (kwargs['record']):
            pickle.dump(successful_episodes, open('../examples/results_with_noise/' + policy_name, 'wb'))

    def visualize_policy_on_demos(self, policy, demos, horizon=1000, num_episodes=1, mode='exploration', **kwargs):
        task = self.env_id.split('-')[0]
        success_threshold = 20 if task == 'pen' else 25
        episodes = []
        total_score = 0.0
        for idx in range(len(demos)):
            print("Episode %d" % idx)
            self.reset()
            self.set_env_state(demos[idx]['init_state_dict'])
            o = self.get_obs()
            d = False
            t = 0
            score = 0.0
            episode_data = {
                'goal_achieved': []
            }
            isRNN = False
            if isinstance(policy.model, RNNNetwork):
                # generate the hidden states at time 0
                # hidden_state = (np.zeros((1, 1, policy.model.rnn_cell.hidden_size)), np.zeros((1, 1, policy.model.rnn_cell.hidden_size)))
                hidden_state = (  # h_{0} and c_{0}
                    np.zeros((1, 1, policy.model.rnn_cell.hidden_size)),
                    np.zeros((1, 1, policy.model.rnn_cell.hidden_size))
                )
                isRNN = True
            if isinstance(policy.model, NCPNetwork):
                hidden_state = np.zeros((1, policy.model.rnn_cell.state_size))
                isRNN = True
            while t < horizon and d is False:
                if isRNN:
                    a = policy.get_action(o, hidden_state)
                    a, hidden_state = (a[0], a[1]['hidden_state']) if mode == 'exploration' else (a[1]['evaluation'], a[1]['hidden_state'])
                else:
                    a = policy.get_action(o)[0] if mode == 'exploration' else policy.get_action(o)[1]['evaluation']
                o, r, d, goal_achieved = self.step(a)
                episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                score = score + r
                self.render()
                t = t+1
            episodes.append(copy.deepcopy(episode_data))
            total_score += score
            print("Episode score = %f, Success = %d" % (score, sum(episode_data['goal_achieved']) > success_threshold))
        print("Average score = %f" % (total_score / len(demos)))
        successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
        print("Success rate = %f" % (len(successful_episodes) / len(demos)))

    def evaluate_policy(self, policy,
                        num_episodes=5,
                        lstm_layers = 1,   
                        batch_size = 1,
                        horizon=None,
                        gamma=1,
                        visual=False,
                        percentile=[],
                        get_full_dist=False,
                        mean_action=False,
                        init_env_state=None,
                        terminate_at_done=True,
                        seed=123):

        self.set_seed(seed)
        task = self.env_id.split('-')[0]
        success_threshold = 20 if task == 'pen' else 25
        horizon = self._horizon if horizon is None else horizon
        mean_eval, std, min_eval, max_eval = 0.0, 0.0, -1e8, -1e8
        ep_returns = np.zeros(num_episodes)
        episodes = []
        total_score = 0.0
        for ep in range(num_episodes):
            episode_data = {
                'goal_achieved': []
            }
            self.reset()
            if init_env_state is not None:
                self.set_env_state(init_env_state)
            t, done = 0, False
            if isinstance(policy.model, NCPNetwork):
                hidden_state = np.zeros((1, policy.model.rnn_cell.state_size))
                # hidden_state = (  # h_{0} and c_{0}
                #     np.zeros((lstm_layers, batch_size, policy.model.rnn_cell.hidden_size)),
                #     np.zeros((lstm_layers, batch_size, policy.model.rnn_cell.hidden_size))
                # )
            if isinstance(policy.model, RNNNetwork):
                # generate the hidden states at time 0
                # hidden_state = (np.zeros((1, 1, policy.model.rnn_cell.hidden_size)), np.zeros((1, 1, policy.model.rnn_cell.hidden_size)))
                hidden_state = (  # h_{0} and c_{0}
                    np.zeros((lstm_layers, batch_size, policy.model.rnn_cell.hidden_size)),
                    np.zeros((lstm_layers, batch_size, policy.model.rnn_cell.hidden_size))
                )
            while t < horizon and (done == False or terminate_at_done == False):
                self.render() if visual is True else None
                o = self.get_obs()
                if isinstance(policy.model, NCPNetwork):
                    a = policy.get_action(o, hidden_state)
                    # print(a)
                    # input()
                    hidden_state = a[1]['hidden_state']
                elif isinstance(policy.model, RNNNetwork):
                    a = policy.get_action(o, hidden_state)
                    hidden_state = a[1]['hidden_state']  # record the hidden state of the last time step
                else:
                    a = policy.get_action(o)
                a = a[1]['evaluation'] if mean_action is True else a[0] # mean_action is True -> noise-free actions
                # print(a)
                # input()
                o, r, done, goal_achieved = self.step(a)
                episode_data['goal_achieved'].append(goal_achieved['goal_achieved'])
                ep_returns[ep] += (gamma ** t) * r
                t += 1
            print("Episode score = %f, Success = %d" % (ep_returns[ep], sum(episode_data['goal_achieved']) > success_threshold))
            episodes.append(copy.deepcopy(episode_data))
            total_score += ep_returns[ep]
        mean_eval, std = np.mean(ep_returns), np.std(ep_returns)  # mean eval -> rewards
        min_eval, max_eval = np.amin(ep_returns), np.amax(ep_returns)
        base_stats = [mean_eval, std, min_eval, max_eval]

        successful_episodes = list(filter(lambda episode: sum(episode['goal_achieved']) > success_threshold, episodes))
        print("Average score = %f" % (total_score / num_episodes))
        print("Success rate = %f" % (len(successful_episodes) / len(episodes)))
        percentile_stats = []
        for p in percentile:
            percentile_stats.append(np.percentile(ep_returns, p))

        full_dist = ep_returns if get_full_dist is True else None
        return [base_stats, percentile_stats, full_dist]
