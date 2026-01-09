import random
from scipy.spatial import KDTree

class Individual(object):

    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.features == other.features
        return False

    def dominates(self, other_individual):
        and_condition = True
        or_condition = False
        for first, second in zip(self.objectives, other_individual.objectives):
            and_condition = and_condition and first <= second
            or_condition = or_condition or first < second
        return (and_condition and or_condition)

class Problem:
    def __init__(self, objectives, surface_coords, tumor_coords):
        self.num_of_objectives = len(objectives)
        self.objectives = objectives
        self.surface_coords = surface_coords
        self.tumor_coords = tumor_coords
        self.surface_kdtree = KDTree(surface_coords)
        self.tumor_kdtree = KDTree(tumor_coords)

    def generate_individual(self):
        individual = Individual()

        entry_point = random.choice(self.surface_coords)
        target_point = random.choice(self.tumor_coords)

        individual.features = list(entry_point) + list(target_point)
        return individual

    def calculate_objectives(self, individual):
        individual.objectives = [f(*individual.features) for f in self.objectives]