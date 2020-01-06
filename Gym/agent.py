class Agent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def act(self, observation):
        """return an action from the action space
        
        Arguments:
            observation {observation} -- env observation
        """
        raise NotImplementedError()

    def learn(self, observation, action, reward, new_observation):
        """learn from action
        
        Arguments:
            observation {observation} -- env observation
            action {action} -- action agent took
            reward {float} -- reward given after action in observation
            new_observation {observation} -- env observation following agent action
        """
        raise NotImplementedError()

