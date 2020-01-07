from collections import namedtuple

Transition = namedtuple("Transition", "observation action reward next_observation")


class Agent:
    def __init__(self, action_space, observation_space, **hyper_params):
        self.action_space = action_space
        self.observation_space = observation_space
        self.hyper_params = hyper_params

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
