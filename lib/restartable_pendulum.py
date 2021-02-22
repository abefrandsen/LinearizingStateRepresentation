import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
from PIL import Image
from PIL import ImageDraw

class RestartablePendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0,repeats=1,pixels=False,cost="classic",render_style="custom"):
        self.cost = cost
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.viewer = None
        self.pixels = pixels
        self.repeats = repeats
        self.render_style = render_style # specifies whether to use default rendering or custom
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.seed()
        
        if pixels:
            o = self.reset()
            self.observation_space = gym.spaces.Box(low=0.0,high=1.0,shape=o.shape)
            
        else:
            self.reset()
            high = np.array([1., 1., self.max_speed])
            self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
            
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _sub_step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        if self.cost == "classic":
            reward = -(angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2))
        elif self.cost == "dm_control":
            reward = float(np.cos(th)>=np.deg2rad(8))

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        
        
        self.state = np.array([newth, newthdot])
        return self._get_obs(), reward, False, {}

    def reset(self, state=None):
        high = np.array([np.pi, 1])
        if state is None:
            self.state = self.np_random.uniform(low=-high, high=high)
        else:
            self.state = state
        self.last_u = None
        
        ob = self._get_obs()
        obs = [ob]
        for _ in range(self.repeats-1):
            out = self._sub_step([0.])
            obs.append(out[0])
        
        return self._process_multiobs(obs)

    def step(self,a):
        obs = []
        rew = 0
        done = False
        for _ in range(self.repeats):
            out = self._sub_step(2*a) # only take inputs between -1 and 1, so multiply by 2
            obs.append(out[0])
            rew += out[1]
            done = done or out[2]
            info = out[3]
        #return self._process_multiobs(obs), rew/self.repeats, done, info
        return self._process_multiobs(obs), rew, done, info
    
    def _process_multiobs(self,multiobs):
        if len(multiobs)==1:
            return multiobs[0]
        else:
            return np.concatenate(multiobs,axis=1)
    
    def _get_obs(self):
        if self.pixels:
            if self.render_style=="default":
                return self.render(mode="rgb_array")
            else:
                return self._get_pixels()
        else: 
            theta, thetadot = self.state
            return np.array([np.cos(theta), np.sin(theta), thetadot])
                                     
    def _get_state(self):
        return self.state
    
    def _get_pixels(self):
        angle = self.state[0]
        
        width, height = 64, 64
        r = int(.45*width) # radius
        img = Image.new('L', (width, height), 0)  
        draw = ImageDraw.Draw(img)
        m = int(.5*width)
        draw.line([(m,m), (m+r*np.cos(angle+np.pi/2),m-r*np.sin(angle+np.pi/2))], fill=255, width=int(.05*width))
        d = int(.05*width)
        draw.ellipse((m-d, m-d, m+d, m+d),fill=255)
        im = np.asarray(img)/255
        return np.expand_dims(im,0) # put the extra dimension last (previously was first)
    
    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            # I've commented out the lines that plot the angular direction of the control
            # to uncomment these, I need to relocate the file clockwise.png to a relative location
            
            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            #self.img = rendering.Image(fname, 1., 1.)
            #self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        #self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        #if self.last_u:
        #    self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
    

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            
class DreamerPendulumEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self, g=10.0,pixels=True,cost="classic"):
        self.cost = cost
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.g = g
        self.viewer = None
        self.pixels = pixels
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        high = np.array([1., 1., self.max_speed], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )
        self.seed()
        #if pixels:
        #    o = self.reset()['image']
        #    self._observation_space = gym.spaces.Box(low=0.0,high=1.0,shape=o.shape)
            
        #else:
        #    self.reset()
        #    high = np.array([1., 1., self.max_speed])
        #    self._observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
            
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta

        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        if self.cost == "classic":
            reward = -(angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2))
        elif self.cost == "dm_control":
            reward = float(np.cos(th)>=np.deg2rad(8))

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111
        
        
        self.state = np.array([newth, newthdot])
        return self._get_obs(), reward, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        
        return self._get_obs()
        
        
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])
    #def _get_obs(self):
    #    if self.pixels:
    #        return {'image': self._get_pixels()}
    #    else: 
    #        theta, thetadot = self.state
    #        return np.array([np.cos(theta), np.sin(theta), thetadot])
                                     
    def _get_state(self):
        return self.state
    
    def _get_pixels(self):
        angle = self.state[0]
        
        width, height = 64, 64
        r = int(.45*width) # radius
        img = Image.new('L', (width, height), 0)  
        draw = ImageDraw.Draw(img)
        m = int(.5*width)
        draw.line([(m,m), (m+r*np.cos(angle+np.pi/2),m-r*np.sin(angle+np.pi/2))], fill=255, width=int(.05*width))
        d = int(.05*width)
        draw.ellipse((m-d, m-d, m+d, m+d),fill=255)
        im = np.asarray(img) # dreamer needs 3 channels, so just repeat the image in all three channels I guess
        return np.repeat(np.expand_dims(im,2),3,axis=-1)
        
    def render(self, mode='human'):
        """
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            # I've commented out the lines that plot the angular direction of the control
            # to uncomment these, I need to relocate the file clockwise.png to a relative location
            
            #fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            #self.img = rendering.Image(fname, 1., 1.)
            #self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        #self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        #if self.last_u:
        #    self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
        """
        return self._get_pixels()
    
    #@property
    #def observation_space(self):
    #    return gym.spaces.Dict({'image': self._observation_space})

    #@property
    #def action_space(self):
    #    return self._action_space

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)