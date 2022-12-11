# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import time
from statistics import mean

import numpy as np

import distanceCalculator
import game
import util

from captureAgents import CaptureAgent
from distanceCalculator import Distancer
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.grid = None
        self.action_grid = None
        self.width = None
        self.height = None
        self.num_carrying = None
        self.team = None
        self.enemies = []
        self.layout = None
        self.distancer = None
        self.enemy_positions = []
        self.get_dead_ends_iterator = 0
        self.initial_game_state = None
        self.lastLostFood = None
        self.strategic_position = None
        self.enemy_at_gate_way = None
        self.gates = []

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        self.action_grid = self.create_action_grid(game_state)
        self.initial_game_state = game_state
        self.get_dead_ends()
        self.layout = game_state.data.layout
        self.distancer = Distancer(self.layout)
        if game_state.is_on_red_team(self.index):
            self.team = 'red'
            self.enemies = game_state.get_blue_team_indices()
        else:
            self.team = 'blue'
            self.enemies = game_state.get_red_team_indices()

    def create_action_grid(self, game_state):
        action_grid = game.Grid(self.width, self.height)
        all_positions, all_game_states = self.get_all_gameStates(game_state)
        for i in range(len(all_positions)):
            actions = all_game_states[i].get_legal_actions(self.index)
            x_cur, y_cur = all_positions[i]
            action_res_pos = []
            for action in actions:
                successor = all_game_states[i].generate_successor(self.index, action)
                x_next, y_next = successor.get_agent_position(self.index)
                action_res_pos.append([action, (x_next, y_next)])
            action_grid[x_cur][y_cur] = [action_res_pos, 0]
        return action_grid

    def get_dead_ends(self):
        for i in range(23):
            action_grid_tmp = self.create_action_grid(self.initial_game_state)
            for x in range(self.width):
                for y in range(self.height):
                    if self.action_grid[x][y] != False:
                        if self.action_grid[x][y][1] != 0:
                            action_grid_tmp[x][y][1] = self.action_grid[x][y][1]
                        else:
                            amount_of_actions = len(self.action_grid[x][y][0])
                            future_deadends = []
                            for element in self.action_grid[x][y][0]:
                                x_new, y_new = element[1]
                                if self.action_grid[x_new][y_new][1] != 0 and element[0] != "Stop":
                                    future_deadends.append(self.action_grid[x_new][y_new][1])
                                    amount_of_actions -= 1
                            if amount_of_actions <= 2:
                                if len(future_deadends) == 0:
                                    action_grid_tmp[x][y][1] = 1
                                else:
                                    action_grid_tmp[x][y][1] = sum(future_deadends) + 1
            self.action_grid = action_grid_tmp

    def create_mdp_matrix(self, game_state):
        return ...

    # def create_visualization(self):
    #     grid_list = []
    #     for x in range(self.width):
    #         for y in range(self.height):
    #             if self.grid[x][y] is not None:
    #                 grid_list.append(self.grid[x][y][1])
    #             else:
    #                 grid_list.append(0)
    #
    #     np_arr = np.array(grid_list)
    #     arr = np_arr.reshape(self.width, self.height)
    #     np.savetxt('data.txt', arr)
    #     print('done')
    def get_all_gameStates(self, game_state):
        # get all possible gameStates via BFS
        all_game_states = []
        observed_game_states = []
        stack = util.Stack()
        stack.push(game_state)
        all_game_states.append(game_state)
        observed_game_states.append(game_state.get_agent_position(self.index))
        while not stack.isEmpty():
            current_game_state = stack.pop()
            possible_actions = current_game_state.get_legal_actions(self.index)
            for action in possible_actions:
                new_game_state = current_game_state.generate_successor(self.index, action)
                if new_game_state.get_agent_position(self.index) not in observed_game_states:
                    all_game_states.append(new_game_state)
                    observed_game_states.append(new_game_state.get_agent_position(self.index))
                    stack.push(new_game_state)
        return observed_game_states, all_game_states

    def compute_value_iteration(self):
        ...

    def computeQValueFromValues(self, start_position, result_position, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """

        return ...

    def choose_action(self, game_state):
        agent_state = game_state.get_agent_state(self.index)
        self.num_carrying = agent_state.num_carrying
        """
        Picks among the actions with the highest Q(s,a).
        """
        x, y = game_state.get_agent_position(self.index)
        self.grid = self.create_mdp_matrix(game_state)
        self.compute_value_iteration()
        return self.grid[x][y][3]

class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        self.action_grid = self.create_action_grid(game_state)
        self.initial_game_state = game_state
        self.get_dead_ends()
        self.layout = game_state.data.layout
        self.distancer = Distancer(self.layout)
        if game_state.is_on_red_team(self.index):
            self.team = 'red'
            self.enemies = game_state.get_blue_team_indices()
        else:
            self.team = 'blue'
            self.enemies = game_state.get_red_team_indices()
        self.gates = [[1,2,3,4],[7,8],[11,12,13,14]]

    def evaluate_enemy_in_their_half(self, position):
        if self.team == "red":
            return position[0] > 15
        else:
            return position[0] <= 15

    def in_own_half(self, position):
        if self.team == "red":
            return position[0] <= 15
        else:
            return position[0] > 15

    def evaluate_own_food_dropof(self, coordinate, position):
        if self.team == 'red':
            return coordinate == 15 and position > 15
        else:
            return coordinate == 16 and position < 16

    def compute_value_iteration(self):
        blocked = []
        if self.enemy_at_gate_way is not None:
            for gate in self.gates:
                if self.enemy_at_gate_way in gate:
                    blocked = gate
        for i in range(100):
            grid_tmp = game.Grid(self.width, self.height)
            for x in range(self.width):
                for y in range(self.height):
                    if self.grid[x][y] is not None:
                        curr_game_state = self.grid[x][y][0]
                        capsules = []
                        if self.team == 'red' and curr_game_state is not None:
                            capsules = curr_game_state.get_blue_capsules()
                        elif curr_game_state is not None:
                            capsules = curr_game_state.get_red_capsules()
                        if not self.grid[x][y][2]:
                            actions_pos = self.action_grid[x][y][0]
                            values = []
                            action_list = []
                            for element in actions_pos:
                                if self.enemy_at_gate_way is not None:
                                    if y in blocked and ((x==15 and self.team=="red" and element[0] == "East") or (x==16 and self.team=="blue" and element[0]=="West")):
                                        ...
                                    else:
                                        action_list.append(element[0])
                                        value = self.computeQValueFromValues((x, y), element[1], element[0], capsules)
                                        values.append(value)
                                else:
                                    action_list.append(element[0])
                                    value = self.computeQValueFromValues((x, y), element[1], element[0], capsules)
                                    values.append(value)
                            val = max(values)
                            ind = values.index(val)
                            act = action_list[ind]
                            grid_tmp[x][y] = [curr_game_state, val, False, act]
                        else:
                            grid_tmp[x][y] = [curr_game_state, self.grid[x][y][1], True, None]
                    else:
                        grid_tmp[x][y] = None
            self.grid = grid_tmp
        #self.create_visualization()

    def create_mdp_matrix(self, game_state):
        self.enemy_positions = []
        if not self.in_own_half(game_state.get_agent_position(self.index)):
            self.enemy_at_gate_way = None
        for enemy in self.enemies:
            enemy_position = game_state.get_agent_position(enemy)
            if enemy_position is not None and self.evaluate_enemy_in_their_half(enemy_position):
                self.enemy_positions.append(enemy_position)
                if self.in_own_half(game_state.get_agent_position(self.index)):
                    self.enemy_at_gate_way = game_state.get_agent_position(self.index)[1]
                else:
                    self.enemy_at_gate_way = None
        # get all positions with the according game_states
        all_positions, all_game_states = self.get_all_gameStates(game_state)
        # create a new grid
        grid = game.Grid(self.width, self.height)

        # check which team we are on
        red_team = game_state.is_on_red_team(self.index)
        if red_team:
            food = game_state.get_blue_food()
            capsules = game_state.get_blue_capsules()

        else:
            food = game_state.get_red_food()
            capsules = game_state.get_red_capsules()

        # add values to the matrix
        for x in range(self.width):
            for y in range(self.height):
                # wall?
                if game_state.has_wall(x, y):
                    grid[x][y] = None
                # food for us?
                elif (x, y) in capsules and len(self.enemy_positions) > 0 and not self.in_own_half((x,y)):
                    try:
                        grid[x][y] = [all_game_states[all_positions.index((x, y))], 100, True, None, None]
                    except:
                        grid[x][y] = [None, 0, True, None, None]
                elif self.action_grid[x][y][1] != 0 and len(self.enemy_positions) > 0 and not self.in_own_half((x,y)):
                    try:
                        grid[x][y] = [all_game_states[all_positions.index((x, y))], 0, False, None, None]
                    except:
                        grid[x][y] = [None, 0, True, None, None]
                elif food[x][y]:
                    try:
                        grid[x][y] = [all_game_states[all_positions.index((x, y))], 50, True, 'FOOD', None]
                    except:
                        grid[x][y] = [None, 0, True, None, None]
                elif self.evaluate_own_food_dropof(x, game_state.get_agent_position(self.index)[0]):
                    if len(self.enemy_positions) > 0:
                        if(self.num_carrying == 0):
                            reward = 15
                        else:
                            reward = 100
                    else:
                        reward = self.num_carrying*30
                    grid[x][y] = [all_game_states[all_positions.index((x, y))], reward, True, None, None]
                # seems to be simple path
                else:
                    try:
                        grid[x][y] = [all_game_states[all_positions.index((x, y))], 0, False, None, None]
                    except:
                        grid[x][y] = [None, 0, True, None, None]

        return grid

    def computeQValueFromValues(self, start_position, result_position, action, capsules):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        reward = -1

        for enemy in self.enemy_positions:
            distance_old = Distancer.getDistance(self.distancer, start_position, enemy)
            distance_new = Distancer.getDistance(self.distancer, result_position, enemy)
            if not self.in_own_half(start_position):
                if len(capsules) != 0:
                    if distance_old <= 5 and not self.in_own_half(start_position):
                        if distance_new < distance_old:
                            reward -= ((6 - distance_new) * 10)
                    if self.action_grid[result_position[0]][result_position[1]][1] != 0:
                        reward -= 10


        if action == 'Stop':
            reward -= 1000

        value = reward + 0.8 * self.grid[result_position[0]][result_position[1]][1]

        return value


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        self.action_grid = self.create_action_grid(game_state)
        self.initial_game_state = game_state
        self.get_dead_ends()
        self.layout = game_state.data.layout
        self.distancer = Distancer(self.layout)
        self.prev_game_state = None
        self.follow_bec_eaten = []
        if game_state.is_on_red_team(self.index):
            self.team = 'red'
            self.enemies = game_state.get_blue_team_indices()
            self.strategic_position = (12, 8)
        else:
            self.team = 'blue'
            self.enemies = game_state.get_red_team_indices()
            self.strategic_position = (19, 8)

    def evaluate_half_line(self, enemy_position, value):
        if self.team == 'red':
            return enemy_position[0] < value
        else:
            return enemy_position[0] >= value

    def stay_in_own_half(self, result_position, value):
        if self.team == 'red':
            return result_position[0] > value
        else:
            return result_position[0] <= value

    def create_mdp_matrix(self, game_state):
        self.enemy_positions = []
        for enemy in self.enemies:
            enemy_position = game_state.get_agent_position(enemy)
            if enemy_position is not None and self.evaluate_half_line(enemy_position, 16):
                self.enemy_positions.append(enemy_position)
        # get all positions with the according game_states
        all_positions, all_game_states = self.get_all_gameStates(game_state)
        # create a new grid
        grid = game.Grid(self.width, self.height)

        # check which team we are on
        red_team = game_state.is_on_red_team(self.index)
        if not red_team:
            food = game_state.get_blue_food()
        else:
            food = game_state.get_red_food()

        # add values to the matrix
        for x in range(self.width):
            for y in range(self.height):
                # wall?
                if game_state.has_wall(x, y):
                    grid[x][y] = None
                # food has just been eaten
                elif (x,y) in self.follow_bec_eaten:
                    try:
                        grid[x][y] = [all_game_states[all_positions.index((x, y))], 100, True, None, None]
                    except:
                        grid[x][y] = [None, 0, True, None, None]
                # seems to be simple path
                elif x == self.strategic_position[0] and y == self.strategic_position[1]:
                    try:
                        grid[x][y] = [all_game_states[all_positions.index((x, y))], 20, True, None, None]
                    except:
                        grid[x][y] = [None, 0, True, None, None]
                else:
                    try:
                        grid[x][y] = [all_game_states[all_positions.index((x, y))], 0, False, None, None]
                    except:
                        grid[x][y] = [None, 0, True, None, None]
        return grid

    def computeQValueFromValues(self, start_position, result_position, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        reward = -1

        for enemy in self.enemy_positions:
            distance_old = Distancer.getDistance(self.distancer, start_position, enemy)
            distance_new = Distancer.getDistance(self.distancer, result_position, enemy)
            if distance_new < distance_old:
                reward = 30

        if self.stay_in_own_half(result_position, 15):
            reward -= 20
        if action == 'Stop':
            reward -= 1000
        value = reward + 0.8 * self.grid[result_position[0]][result_position[1]][1]

        return value

    def choose_action(self, game_state):

        if game_state.get_agent_position(self.index) in self.follow_bec_eaten:
            self.follow_bec_eaten.remove(game_state.get_agent_position(self.index))

        # calculate food eaten
        if self.prev_game_state is not None:
            food_found = []
            if self.team == 'red':
                prev_food = self.prev_game_state.get_red_food()
                curr_food = game_state.get_red_food()
            else:
                prev_food = self.prev_game_state.get_blue_food()
                curr_food = game_state.get_blue_food()

            # iterate through current food and find where there is a mismatch
            for x in range(self.width):
                for y in range(self.height):
                    # if exist in prev, but not in current
                    if prev_food[x][y] is True and curr_food[x][y] is False:
                        food_found.append((x, y))

            # make sure that we are always running to the newest food
            if len(food_found) != 0:
                self.follow_bec_eaten = food_found

        agent_state = game_state.get_agent_state(self.index)
        self.num_carrying = agent_state.num_carrying
        """
        Picks among the actions with the highest Q(s,a).
        """
        x, y = game_state.get_agent_position(self.index)
        self.grid = self.create_mdp_matrix(game_state)
        self.compute_value_iteration()
        actions = self.grid[x][y][0].get_legal_actions(self.index)
        for action in actions:
            next_state = self.grid[x][y][0].generate_successor(self.index, action)
            value = self.computeQValueFromValues(game_state.get_agent_position(self.index),
                                                 next_state.get_agent_position(self.index), action)
        self.prev_game_state = game_state
        return self.grid[x][y][3]

    def compute_value_iteration(self):
        for i in range(100):
            grid_tmp = game.Grid(self.width, self.height)
            for x in range(self.width):
                for y in range(self.height):
                    if self.grid[x][y] is not None:
                        curr_game_state = self.grid[x][y][0]
                        if not self.grid[x][y][2]:
                            actions_pos = self.action_grid[x][y][0]
                            values = []
                            action_list = []
                            for element in actions_pos:
                                action_list.append(element[0])
                                value = self.computeQValueFromValues((x, y), element[1], element[0])
                                values.append(value)
                            val = max(values)
                            ind = values.index(val)
                            act = action_list[ind]
                            grid_tmp[x][y] = [curr_game_state, val, False, act]
                        # we have reached the point where the food was eaten --> take random action away from there
                        elif (x,y) in self.follow_bec_eaten:
                            actions = curr_game_state.get_legal_actions(self.index)
                            action = random.choice(actions)
                            grid_tmp[x][y] = [curr_game_state, self.grid[x][y][1], True, action]
                        elif x == self.strategic_position[0] and y == self.strategic_position[1]:
                            grid_tmp[x][y] = [curr_game_state, self.grid[x][y][1], True, 'South']
                        else:
                            grid_tmp[x][y] = [curr_game_state, self.grid[x][y][1], True, None]
                    else:
                        grid_tmp[x][y] = None
            self.grid = grid_tmp