from collections import namedtuple

Transition = namedtuple("Transition", "observation action reward next_observation")


class Agent:
    def __init__(self, action_space, observation_space, **hyper_params):
        self.action_space = action_space
        self.observation_space = observation_space
        self.hyper_params = hyper_params
        
        if "buffer_size" in hyper_params:
            self.replay_buffer = []

    def act(self, observation):
        """return an action from the action space
        
        Arguments:
            observation {observation} -- env observation
        """
        raise NotImplementedError()

    def learn(self, observation, action, reward, next_observation):
        """learn from action
        
        Arguments:
            observation {observation} -- env observation
            action {action} -- action agent took
            reward {float} -- reward given after action in observation
            new_observation {observation} -- env observation following agent action
        """
        raise NotImplementedError()

    def get_hyper_param(self, param_name, default_value=None):
        if param_name in self.hyper_params:
            setattr(self, param_name, self.hyper_params[param_name])
        elif default_value:
            setattr(self, param_name, default_value)
        else:
            raise ValueError(
                f"param {param_name} not in hyper params, and no default value was provided. "
            )

    def buffer_transition(
        self, observation, action, reward, next_observation
    ) -> Transition:
        """
        add transition to replay buffer, and return the last given arguments as a transition
        """
        if not self.replay_buffer:
            self.replay_buffer = []
        
        # generate transition and insert it into the buffer
        transition = Transition(observation, action, reward, next_observation)
        self.replay_buffer.insert(0, transition)

        # keep buffer at buffer_size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop()

        return transition
