import agent
import gridworld
import math
from gridworld import BoxGame

# def training_cycle(epsilon, num_steps, ag, ag_range):
#     #new_environment(ag, ag_range)
#     agent.e_greedy_episode(epsilon=epsilon, reset=True)
#     for _ in range(num_steps):
#         agent.recall_study(5, backup=1, allow_batch_reduction=True)
#     agent.update_target_net()
#
#
# def epsilon_control(ep):
#
#     #return 1.
#     if ep < 100:
#         return .8
#     elif ep < 200:
#         return .7
#     elif ep < 500:
#         return .5
#     elif ep < 1000:
#         return .1
#     else:
#         return .5
#
# def new_environment(ag, ag_range):
#     game = gridworld.BoxGame(10, block_rate=1., num_exits=1, num_pits=0,
#                              num_empty_targets=0, watch_construction=False,
#                              max_tree_steps=100, bias=.9,
#                              pad_size=ag_range,
#                              instant_death_p=0.)
#     ag.set_environment(game)
#
#
# agent_range = 5
#
# agent = agent.Agent(agent_range, episode_mem_lim=20,discount=.9,short_term_cap=200)
#
#
# print("gather initial data")
# for i in range(30):
#     new_environment(agent, agent_range)
#     agent.e_greedy_episode(epsilon=.99, reset=True)
#
# print("initial training")
# for j in range(130):
#     agent.recall_study(5, backup=1, allow_batch_reduction=True)
#     if j % 20 == 1:
#         agent.update_target_net()
#
# ep_step_list = []
#
# for ep in range(1000):
#     init_steps = agent.step_counter
#     print("episode ", ep, " with epsilon <--", epsilon_control(ep))
#     training_cycle(epsilon_control(ep), 20, agent, agent_range)
#     steps_taken = agent.step_counter - init_steps
#     print("steps taken in episode: ", steps_taken)
#     ep_step_list.append(steps_taken)
#
#




def generate_mazes(num_mazes):
    mazes = []
    for i in range(num_mazes):
        if i % 100 == 0:
            print(i/num_mazes)
        mazes.append(gridworld.BoxGame(10,1.,1,0,0,False,100,.9,1,None))

    return mazes


def probe():
    ag_range = 1
    ag = agent.Agent(ag_range, episode_mem_lim=20, discount=0.,
                     short_term_cap=5)
    game = gridworld.BoxGame(10, block_rate=1., num_exits=1, num_pits=0,
                             num_empty_targets=0, watch_construction=False,
                             max_tree_steps=100, bias=.9,
                             pad_size=ag_range,
                             instant_death_p=0.)
    ag.set_environment(game)
    ag.e_greedy_episode(.9,reset=True)

    while True:
        ag.recall_study(1, backup=1)
        ag.update_target_net()

if __name__ == "__main__":
    import pickle
    import random
    #
    # def starting_epsilon(level):
    #     """Outputs the initial value of epsilon to use at each episode"""
    #     if level == 1:
    #         return 1.
    #     elif level <= 7:
    #         return .5
    #     else:
    #         return .2

    notes = "lr = 2e-12 ADAM, (.5, .999) , 40 field features, no temporal_convolution, batch_size 30, 300 -> 50 hidden\n" \
            "policy gradient on with lr = 2e-8 ADAM, (.5, .999)"

    print("gathering maze data... ", end="")
    mazes = []
    for lev in range(1, 10): #currently we have levels 1-9
        filename = "level_" + str(lev) + ".dat"
        with open(filename, "rb") as f:
            maze_group = pickle.load(f)

        for maze in maze_group:
            maze.exit_reward = 5.
            maze.step_reward = -.1
            maze.crash_reward = -.2
            maze.timeout_reward = maze.crash_reward

        mazes.append(maze_group)
    print("done")

    total_mazes = 10000
    for i in range(1):
        assert len(mazes[i]) == total_mazes
    ag = agent.Agent(5, discount=.95,mem_len=1000)
    episodes_per_group = 10
    num_learning_steps = 1000



    for level in range(1,10):
        print("\n\n*** Starting level ", level, " ***\n")
        maze_pos = 0
        counter = 0
        while counter <= total_mazes:

            epsilon = min(.5, math.exp(-2*counter / total_mazes))

            print(notes)
            print("level, maze number, counter, epsilon:",
                  level, maze_pos, counter, epsilon, sep="   ")
            init_steps = ag.step_counter
            for maze in mazes[level-1][maze_pos: maze_pos + episodes_per_group]:
                ag.set_environment(maze)
                rew = ag.e_greedy_episode(epsilon=epsilon,policy_grad=True)
                normal_steps = rew.count(-.1)
                crashes = rew.count(-.2)
                escape = rew.count(5.)
                print("crash / step / escape\n", crashes, normal_steps, escape)

            steps = ag.step_counter - init_steps

            counter += episodes_per_group
            maze_pos = (maze_pos + episodes_per_group) % total_mazes
            print("learning phase starting")
            #  We base the number of training steps off the number of
            #  steps during that last group of episodes.  This helps to
            #  train to the size of the data set
            for _ in range(steps):
                # if random.random()< .0001:
                #     print(notes)
                ag.recall_study(batch_size=30, allow_batch_reduction=False)

            ag.update_target_net()

        print("\nGraduated!!\n")


