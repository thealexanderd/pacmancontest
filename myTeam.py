import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions, Actions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    
    return [eval(first)(first_index), eval(second)(second_index)]

class AStarAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.positions_history = []  # To track the agent's positions
        self.stuck_threshold = 6  # Number of consecutive actions before the agent is considered stuck
        self.last_food_collected_turn = 0

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Main method to choose the agent's action. This should be overridden by subclasses.
        """
        util.raiseNotDefined()

    def is_stuck(self, gameState):
        # Check if the agent is stuck by seeing if its position hasn't changed much
        # print(self.positions_history)
        if len(self.positions_history) == self.stuck_threshold and len(set(self.positions_history)) <= 3:
            print("Stuck!")
            return True

        return False


    def a_star_search(self, start, goal, gameState, offense=False):
        def heuristic(position):
            if offense:
                enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
                num_carrying = gameState.get_agent_state(self.index).num_carrying
                ghost_distances = [self.get_maze_distance(start, a.get_position()) for a in enemies if not a.is_pacman and a.get_position() != None]
                are_scared = [a.scared_timer > 0 for a in enemies if not a.is_pacman and a.get_position() != None]
                min_ghost_distance = 1000
                if len(ghost_distances) > 0:
                    min_ghost_distance = min(ghost_distances)
                    min_ghost_distance_index = ghost_distances.index(min_ghost_distance)
                    is_scared = are_scared[min_ghost_distance_index]
                    if is_scared:
                        min_ghost_distance = 0
                try:
                    return self.get_maze_distance(position, goal) + min_ghost_distance * num_carrying
                except:
                    return 1000
            return abs(position[0] - goal[0]) + abs(position[1] - goal[1])
        
        def alternative_heuristic(position):
        # Define an alternative heuristic that slightly favors different paths
            try:
                return self.get_maze_distance(position, goal) + random.uniform(0, 5)
            except:
                return abs(position[0] - goal[0]) + abs(position[1] - goal[1]) + random.uniform(0, 5)

        open_set = util.PriorityQueue()
        open_set.push(start, 0)
        came_from = {}
        cost_so_far = {}
        came_from[start] = None
        cost_so_far[start] = 0

        # if self.is_stuck(gameState):
        #     # If the agent is stuck, introduce randomness in path selection
        #     actions = gameState.get_legal_actions(self.index)
        #     if actions:
        #         action = random.choice(actions)
        #         dx, dy = Actions.direction_to_vector(action)
        #         nextx, nexty = int(start[0] + dx), int(start[1] + dy)
        #         return [(nextx, nexty)]

        while not open_set.isEmpty():
            current = open_set.pop()

            if current == goal:
                break

            successors = self.get_successors(current, gameState)
            for next in successors:
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    if self.is_stuck(gameState):
                        priority = new_cost + alternative_heuristic(next)
                    else:
                        priority = new_cost + heuristic(next)
                    open_set.push(next, priority)
                    came_from[next] = current

        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    
    def get_action_for_next_step(self, current_position, next_position):
        # Convert the next step (a coordinate) to a direction action
        x1, y1 = current_position
        x2, y2 = next_position
        if x2 > x1: return 'East'
        if x2 < x1: return 'West'
        if y2 > y1: return 'North'
        if y2 < y1: return 'South'
        return 'Stop'
    
    def get_successors(self, position, gameState):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not gameState.has_wall(nextx, nexty):
                successors.append((nextx, nexty))
        return successors


class OffensiveReflexAgent(AStarAgent):
    # def __init__(self, index, time_for_computing=.1):
    #     super().__init__(index, time_for_computing)
    def choose_action(self, gameState):
        current_position = gameState.get_agent_state(self.index).get_position()
        self.positions_history.append(current_position)
        if len(self.positions_history) > self.stuck_threshold:
            self.positions_history.pop(0)

        my_pos = gameState.get_agent_state(self.index).get_position()
        food_list = self.get_food(gameState).as_list()
        capsules = self.get_capsules(gameState)
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        ghost_distances = [self.get_maze_distance(my_pos, a.get_position()) for a in enemies if not a.is_pacman and a.get_position() != None]
        are_scared = [a.scared_timer > 0 for a in enemies if not a.is_pacman and a.get_position() != None]
        
        is_scared = False
        ispac = gameState.get_agent_state(self.index).is_pacman
        # min_ghost_distance = min([self.get_maze_distance(my_pos, a.get_position()) for a in enemies if not a.is_pacman and a.get_position() != None])
        if len(ghost_distances) and capsules:  # Adjust distance threshold as needed
            best_capsule = min(capsules, key=lambda cap: self.get_maze_distance(my_pos, cap))
            path_to_food = self.a_star_search(my_pos, best_capsule, gameState, offense=True)
            return self.get_action_for_next_step(my_pos, path_to_food[0])
        if len(food_list) > 0:
            nearest_food = min(food_list, key=lambda x: self.get_maze_distance(my_pos, x))
            neares_food_distance = self.get_maze_distance(my_pos, nearest_food)
        if len(ghost_distances) > 0:
            min_ghost_distance = min(ghost_distances)
            min_ghost_distance_index = ghost_distances.index(min_ghost_distance)
            is_scared = are_scared[min_ghost_distance_index]
        else:
            min_ghost_distance = 1000
        carrying_food = gameState.get_agent_state(self.index).num_carrying
        food_limit = 5
        
        if is_scared: 
            food_limit = 10
        if gameState.data.timeleft > 50 and carrying_food <= food_limit:
            if len(food_list) > 2 and (not ispac or is_scared or neares_food_distance < min_ghost_distance):
                path_to_food = self.a_star_search(my_pos, nearest_food, gameState, offense=True)
                if path_to_food:
                    return self.get_action_for_next_step(my_pos, path_to_food[0])
            else:
                path_to_home = self.a_star_search(my_pos, self.start, gameState, offense=True)
                if path_to_home:
                    return self.get_action_for_next_step(my_pos, path_to_home[0])
        else:       
            home_pos = self.start
            path_to_home = self.a_star_search(my_pos, home_pos, gameState, offense=True)
            return self.get_action_for_next_step(my_pos, path_to_home[0]) if path_to_home else 'Stop'
        return 'Stop'
    
    def get_successors(self, position, gameState):
        successors = []
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        ghost_position = [a.get_position() for a in enemies if not a.is_pacman and a.get_position() != None]
        ispac = gameState.get_agent_state(self.index).is_pacman
        are_scared = [a.scared_timer > 0 for a in enemies if not a.is_pacman and a.get_position() != None]

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = position
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            
            if not gameState.has_wall(nextx, nexty):
                if ispac or gameState.is_red((nextx, nexty)):
                    if (nextx, nexty) not in ghost_position or are_scared[ghost_position.index((nextx, nexty))]:
                        successors.append((nextx, nexty))
                else:
                    successors.append((nextx, nexty))

        return successors

    def get_ghost_positions(self, gameState):
        """
        Return a list of positions of all visible ghosts.
        """
        ghost_positions = []
        for opponent in self.get_opponents(gameState):
            opponent_state = gameState.get_agent_state(opponent)
            if not opponent_state.is_pacman and opponent_state.get_position() is not None:
                ghost_positions.append(opponent_state.get_position())
        return ghost_positions
    
class DefensiveReflexAgent(AStarAgent):

    def choose_action(self, gameState):
        my_pos = gameState.get_agent_state(self.index).get_position()

        invaders = self.get_invaders(gameState)

        if invaders:
            # If the invader eats the power_pill we must go to eat food
            # invader_with_power_pill = any(invader.scared_timer > 0 for invader in invaders)
            if_im_scared = gameState.get_agent_state(self.index).scared_timer > 0
            if if_im_scared:
                food_list = self.get_food(gameState).as_list()
                nearest_food = min(food_list, key=lambda x: self.get_maze_distance(my_pos, x))
                path_to_food = self.a_star_search(my_pos, nearest_food, gameState)
                if path_to_food:
                    return self.get_action_for_next_step(my_pos, path_to_food[0])
            # If there are invaders, find the closest one and go after it
            target_invader = min(invaders, key=lambda x: self.get_maze_distance(my_pos, x.get_position()))
            path_to_invader = self.a_star_search(my_pos, target_invader.get_position(), gameState)
            if path_to_invader:
                next_action = self.get_action_for_next_step(my_pos, path_to_invader[0])
            return next_action if path_to_invader else 'Stop'

        else:
            # If no invaders, find the point with food closer to the center
            target_food_point = self.get_point_with_center_food(gameState)
            if my_pos == target_food_point:
                # If already in the target point, choose a random action
                legal_actions = gameState.get_legal_actions(self.index)
                action = random.choice(legal_actions)
                legal_actions.remove(action)
                dx, dy = Actions.direction_to_vector(action)
                nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)
                if not gameState.is_on_red_team(self.index):
                    while gameState.is_red((nextx, nexty)):
                        action = random.choice(legal_actions)
                        dx, dy = Actions.direction_to_vector(action)
                        nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)
                    return action
                else:
                    while not gameState.is_red((nextx, nexty)):
                        action = random.choice(legal_actions)
                        dx, dy = Actions.direction_to_vector(action)
                        nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)
                    return action
            path_to_food_point = self.a_star_search(my_pos, target_food_point, gameState)
            if path_to_food_point:
                distance_to_food = len(path_to_food_point)
                if distance_to_food <= 2:
                    legal_actions = gameState.get_legal_actions(self.index)
                    action = random.choice(legal_actions)
                    legal_actions.remove(action)
                    dx, dy = Actions.direction_to_vector(action)
                    nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)
                    if not gameState.is_on_red_team(self.index):
                        while gameState.is_red((nextx, nexty)):
                            action = random.choice(legal_actions)
                            dx, dy = Actions.direction_to_vector(action)
                            nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)
                        return action
                    else:
                        while not gameState.is_red((nextx, nexty)):
                            action = random.choice(legal_actions)
                            dx, dy = Actions.direction_to_vector(action)
                            nextx, nexty = int(my_pos[0] + dx), int(my_pos[1] + dy)
                        return action
                else:
                    # If more than 2 steps away, continue with the original logic
                    next_action = self.get_action_for_next_step(my_pos, path_to_food_point[0])
                    return next_action


    def get_invaders(self, gameState):
        enemies = [gameState.get_agent_state(i) for i in self.get_opponents(gameState)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() != None]
        return invaders


    def has_enemy_food(self, gameState, x, y):

        if gameState.is_on_red_team(self.index):
            return gameState.get_red_food()[x][y]
        else:
            return gameState.get_blue_food()[x][y]
    
    def get_point_with_center_food(self, gameState):
        """
        Returns the position with food that is closer to the center of the screen.
        """
        food_positions = [(x, y) for x in range(gameState.data.layout.width) for y in range(gameState.data.layout.height)
                          if self.has_enemy_food(gameState, x, y)]
        center = (gameState.data.layout.width // 2, gameState.data.layout.height // 2)
        food_distances = {pos: self.get_maze_distance(pos, center) for pos in food_positions}
        if len(food_distances) == 0:
            return None
        target_food_point = min(food_distances, key=food_distances.get)
        return target_food_point
    
    