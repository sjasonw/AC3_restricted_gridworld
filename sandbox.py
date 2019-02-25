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

    def starting_epsilon(level):
        """Outputs the initial value of epsilon to use at each episode"""
        if level == 1:
            return 1.
        elif level <= 7:
            return .5
        else:
            return .2

    notes = "self.optimizer = optim.RMSprop(self.value_net.parameters(),\n" \
            " lr=5e-6, momentum=.99, timeout at 75 steps.  9 levels of mazes.\n" \
            "new rewards: exit 1., step -.01, crash -.02, timeout -.02\n" \
            "discount .99\n" \
            "using a 2 layer RNN (60 hidden) but 2 layer fc afterwards \n"

    print("gathering maze data... ", end="")
    mazes = []
    for lev in range(1, 10): #currently we have levels 1-9
        filename = "level_" + str(lev) + ".dat"
        with open(filename, "rb") as f:
            maze_group = pickle.load(f)

        for maze in maze_group:
            maze.exit_reward = 1.
            maze.step_reward = -.1
            maze.crash_reward = -.2
            maze.timeout_reward = maze.crash_reward

        mazes.append(maze_group)
    print("done")

    total_mazes = 10000
    for i in range(9):
        assert len(mazes[i]) == total_mazes
    ep_limit = 20
    short_cap = 100
    num_learning_steps = 100

    ag_1 =agent.Agent(5,1.,ep_limit,short_cap,blur=.05)

    for level in range(1,10):
        print("\n\n*** Starting level ", level, " ***\n")
        maze_pos = 0
        counter = 0
        while counter <= 4 * total_mazes:
            epsilon = min(
                starting_epsilon(level), math.exp(-counter / total_mazes)
            )
            success_count = 0
            print(notes)
            print("level, maze number, counter, epsilon:",
                  level, maze_pos, counter, epsilon, sep="   ")
            for maze in mazes[level-1][maze_pos: maze_pos + ep_limit]:
                ag_1.set_environment(maze)
                s1 = ag_1.step_counter
                ag_1.e_greedy_episode(epsilon=epsilon, reset=True)
                s2 = ag_1.step_counter
                success_count += int(s2 - s1 < maze.timeout_steps - 1)  # TODO the -1 is there for lazy safety...

            # only decrease epsilon if the agent is still winning mazes reasonably well
            if success_count > (ep_limit / 2)*(1-epsilon):
                counter += 2 * ep_limit # 2 increases counter growth rate

            maze_pos = (maze_pos + ep_limit) % total_mazes
            print("learning phase starting")
            for _ in range(num_learning_steps):
                ag_1.recall_study(batch_size=5, backup=1)

            ag_1.update_target_net()

        print("\nGraduated!!\n")


