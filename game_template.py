from game import Game


class GameTemplate(Game):
    """
    Template for creating your own agent
    Add any methods you might need in here.
    If you want to store extra information use the `data` method.
    """
    def __init__(self, board, network, render):

        super(GameTemplate, self).__init__(board, network, render)

    def play(self, tickdelay, timeout):
        """
        This method is used as the main game loop. This will be improved :)
        :param tickdelay:
        :param timeout:
        :return: None
        """
        pass

    @property
    def fitness(self):
        """
        Use this to determine the fitness of your agent. If not overridden it will be the score.
        :return:
        :rtype: float
        """
        return 0
