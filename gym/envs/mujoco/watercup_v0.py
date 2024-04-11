import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

DEFAULT_CAMERA_CONFIG = {
    "azimuth": -90,
    "distance": 1.5,
}

class WatercupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="watercup.xml",
        in_hand_reward=1,
        ctrl_cost_weight=0.5,
        contact_cost_weight=5e-4,
        contact_force_range=(-1.0, 1.0),
        terminate_when_not_in_hand=True,
        in_hand_range=0.2
    ):
        self.cup_bid = 0
        self.grasp_site_sid = 0
        
        utils.EzPickle.__init__(**locals())

        self._in_hand_reward = in_hand_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_force_range = contact_force_range
        self._terminate_when_not_in_hand = terminate_when_not_in_hand
        self._in_hand_range = in_hand_range
        # self.act_mid = np.mean(self.model.actuator_ctrlrange, axis=1)
        # self.act_rng = 0.5*(self.model.actuator_ctrlrange[:,1]-self.model.actuator_ctrlrange[:,0])
       

        # self.init_dist = abs(self.reset_model()[-1]) # initial distance between cup and grasp_site
        self.init_dist = 0
        self.nstep = 0

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)
        self.cup_bid = self.sim.model.body_name2id('cup')
        self.grasp_site_sid = self.sim.model.site_name2id('grasp_site')

       
        # self.action_space.high = np.ones_like(self.model.actuator_ctrlrange[:,1])
        # self.action_space.low  = -1.0 * np.ones_like(self.model.actuator_ctrlrange[:,0])
        
    @property
    def in_hand_reward(self):
        return (
            float(self.is_in_hand or self._terminate_when_not_in_hand)
            * self._in_hand_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.sensordata # look up touch sensor data
        self.sensor = raw_contact_forces
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_in_hand(self):
        curr_dist = np.linalg.norm(self._get_obs()[-3:])
        self.curr_dist = curr_dist
        is_in_hand = True if curr_dist < self._in_hand_range else False
        return is_in_hand

    @property
    def done(self):
        done = not self.is_in_hand if self._terminate_when_not_in_hand else False
        return done

    def step(self, action):
        
        self.do_simulation(action,self.frame_skip)
        if self.nstep == 1:
            # initialize pose
            # self.reset_model()
            # self.data.qpos[:30] = [0.091343, 0.0342746, -0.233073+0.07, 0.00485932, -0.124965, -0.05135, 0.127681, -0.123797, -0.086891, 0.937618, 0.0101089, 1.27532, -0.149746, 1.10426, -0.000827189, 1.25706, 0.109944, 1.16752, -7.03124e-05, 1.22048, 0.0387147, 0.101557, 1.13742, -0.00127704, 1.19742, -0.931246, 1.24242, 0.22337, 0.545332, 0.722598]
            # action = [0.174, 0.0793, 1.05, 1.22, 0.209, 0.698, 0.81, -0.119, 1.04, 1.29, -0.164, 1.14, 1.29, 0.119, 1.17, 1.23, 0.475, 0.115, 1.2, 1.24, 0.0525, 0, -0.027+0.07, 0, -0.06, 0]
            # action = np.zeros(26)
            # cup_pos  = self.data.body_xpos[self.cup_bid].ravel()
            # palm_pos = self.data.site_xpos[self.grasp_site_sid].ravel()
            # self.init_dist = np.linalg.norm(palm_pos-cup_pos)
            self.init_dist = np.linalg.norm(self._get_obs()[-3:]) # initial distance between cup and grasp_site
        self.nstep += 1

        observation = self._get_obs()

        # cup_pos  = self.data.body_xpos[self.cup_bid].ravel()
        # palm_pos = self.data.site_xpos[self.grasp_site_sid].ravel()

        # reward
        curr_dist = np.linalg.norm(observation[-3:])
        delta_dist = abs(curr_dist - self.init_dist)
        if delta_dist == 0:
            retain_reward = 40
        else:
            retain_reward = np.reciprocal(delta_dist)/25.0
        retain_reward = np.clip(retain_reward, 0, 40)
        
        in_hand_reward = self.in_hand_reward
        rewards = retain_reward + in_hand_reward

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = ctrl_cost + contact_cost

        # reward = rewards - costs
        reward = rewards

        done = self.done
        info = {
            "reward_retain": retain_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_contact": -contact_cost,
            "reward_in_hand": in_hand_reward,
            "in_hand_range": self._in_hand_range,
            "curr_dist1": self.curr_dist,
            "curr_dist2": curr_dist,
            "init_dist": self.init_dist,
            "nstep": self.nstep,
            "action": action,
            "sensor": self.sensor
        }

        if self.nstep == 1:
            done = False
        return observation, reward, done, info

    def _get_obs(self):
        qpos = self.data.qpos[:30].ravel()  # (30,)   qpos details {'total': 30, 'hand_Txyz': 3, 'hand_Rxyz': 3, 'hand_joints': 24}
        cup_pos  = self.data.body_xpos[self.cup_bid].ravel() # (3,) 
        palm_pos = self.data.site_xpos[self.grasp_site_sid].ravel() # (3,) 
        return np.concatenate([qpos, palm_pos-cup_pos])

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()
        self.set_state(qpos, qvel)

        # initial grasping
        self.data.qpos[:30] = [0.091343, 0.0342746, -0.233073+0.07, 0.00485932, -0.124965, -0.05135, 0.127681, -0.123797, -0.086891, 0.937618, 0.0101089, 1.27532, -0.149746, 1.10426, -0.000827189, 1.25706, 0.109944, 1.16752, -7.03124e-05, 1.22048, 0.0387147, 0.101557, 1.13742, -0.00127704, 1.19742, -0.931246, 1.24242, 0.22337, 0.545332, 0.722598]
        self.data.ctrl[:] = [0.174, 0.0793, 1.05, 1.22, 0.209, 0.698, 0.81, -0.119, 1.04, 1.29, -0.164, 1.14, 1.29, 0.119, 1.17, 1.23, 0.475, 0.115, 1.2, 1.24, 0.0525, 0, -0.027+0.07, 0, -0.06, 0]

        observation = self._get_obs()
        return observation

    def viewer_setup(self):
        assert self.viewer is not None
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)