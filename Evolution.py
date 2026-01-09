import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pymoo.indicators.hv import HV
from pymoo.indicators.gd import GD
from tqdm import tqdm
import random
import math
matplotlib.use('TkAgg')

class Population:

    def __init__(self):
        self.population = []
        self.fronts = []

    def __len__(self):
        return len(self.population)

    def __iter__(self):
        return self.population.__iter__()

    def extend(self, new_individuals):
        self.population.extend(new_individuals)

    def append(self, new_individual):
        self.population.append(new_individual)

class NSGA2Utils:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob


    def create_initial_population(self):
        population = Population()
        while len(population) < self.num_of_individuals:
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            if (
                    individual.objectives[0] <= 90 and
                    individual.objectives[1] <= math.cos(math.radians(20))
            ):
                population.append(individual)
        return population

    def fast_nondominated_sort(self, population):
        population.fronts = [[]]
        for individual in population:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in population:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                population.fronts[0].append(individual)
        i = 0
        while len(population.fronts[i]) > 0:
            temp = []
            for individual in population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            population.fronts.append(temp)

    def calculate_crowding_distance(self, front):
        if len(front) > 0:
            solutions_num = len(front)
            for individual in front:
                individual.crowding_distance = 0

            for m in range(len(front[0].objectives)):
                front.sort(key=lambda individual: individual.objectives[m])
                front[0].crowding_distance = 10 ** 9
                front[solutions_num - 1].crowding_distance = 10 ** 9
                m_values = [individual.objectives[m] for individual in front]
                scale = max(m_values) - min(m_values)
                if scale == 0: scale = 1
                for i in range(1, solutions_num - 1):
                    front[i].crowding_distance += (front[i + 1].objectives[m] - front[i - 1].objectives[m]) / scale

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            if (
                    child1.objectives[0] <= 90 and
                    child1.objectives[1] <= math.cos(math.radians(20))
            ):
                children.append(child1)

            if (
                    child2.objectives[0] <= 90 and
                    child2.objectives[1] <= math.cos(math.radians(20))
            ):
                children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()

        child1.features[0:3] = individual1.features[0:3]
        child1.features[3:6] = individual2.features[3:6]

        child2.features[0:3] = individual2.features[0:3]
        child2.features[3:6] = individual1.features[3:6]
        return child1, child2

    def __mutate(self, child):
        mutation_prob = 0.3
        k = 10


        current_entry = np.array(child.features[0:3])
        current_target = np.array(child.features[3:6])


        if random.random() < mutation_prob:
            distances, indices = self.problem.surface_kdtree.query(current_entry, k=k)
            child.features[0:3] = self.problem.surface_coords[random.choice(indices)]


        if random.random() < mutation_prob:
            distances, indices = self.problem.tumor_kdtree.query(current_target, k=k)
            child.features[3:6] = self.problem.tumor_coords[random.choice(indices)]


    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant

        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False

class Evolution:

    def __init__(self, problem, num_of_generations=1000, num_of_individuals=100, num_of_tour_particips=2,
                 tournament_prob=0.9):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob)
        self.population = None
        self.num_of_generations = num_of_generations
        self.num_of_individuals = num_of_individuals


        self.all_pareto_raw = []


        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.cbar = None
        plt.ion()

    def evolve(self):
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None

        for gen_idx in tqdm(range(self.num_of_generations), desc="Evolving generations"):
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)

            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(key=lambda ind: ind.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals - len(new_population)])

            returned_population = self.population
            self.population = new_population

            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)

            pareto_front = self.population.fronts[0]


            self.all_pareto_raw.append([
                (ind.objectives[0],
                 np.degrees(np.arccos(np.clip(ind.objectives[1], 0.0, 1.0))),
                 ind.objectives[2])
                for ind in pareto_front
            ])


            self.visualize_pareto_front(pareto_front, gen_idx)

            plt.pause(0.01)

        plt.ioff()
        plt.show()


        self.compute_final_metrics()
        self.plot_all_indicators()

        return returned_population.fronts[0]

    def compute_final_metrics(self):

        all_f1 = np.array([pt[0] for gen in self.all_pareto_raw for pt in gen])
        all_f2 = np.array([pt[1] for gen in self.all_pareto_raw for pt in gen])
        all_f3 = np.array([pt[2] for gen in self.all_pareto_raw for pt in gen])

        self.f1_min, self.f1_max = all_f1.min(), all_f1.max()
        self.f2_min, self.f2_max = all_f2.min(), all_f2.max()
        self.f3_min, self.f3_max = all_f3.min(), all_f3.max()


        all_points = np.unique(np.array([
            pt for gen in self.all_pareto_raw for pt in gen
        ]), axis=0)

        f1_ref_norm = (all_points[:, 0] - self.f1_min) / (self.f1_max - self.f1_min + 1e-9)
        f2_ref_norm = (all_points[:, 1] - self.f2_min) / (self.f2_max - self.f2_min + 1e-9)
        f3_ref_norm = (all_points[:, 2] - self.f3_min) / (self.f3_max - self.f3_min + 1e-9)
        self.reference_pareto = np.vstack([f1_ref_norm, f2_ref_norm, f3_ref_norm]).T


        self.global_ref_point = np.max(self.reference_pareto, axis=0) * 1.1


        self.hypervolume_list = []
        self.gd_list = []
        self.spread_list = []
        self.pareto_size_list = []

        for gen_pareto in self.all_pareto_raw:
            # 提取当前代目标值
            f1 = np.array([p[0] for p in gen_pareto])
            f2 = np.array([p[1] for p in gen_pareto])
            f3 = np.array([p[2] for p in gen_pareto])

            # 归一化
            f1_norm = (f1 - self.f1_min)/(self.f1_max - self.f1_min + 1e-9)
            f2_norm = (f2 - self.f2_min)/(self.f2_max - self.f2_min + 1e-9)
            f3_norm = (f3 - self.f3_min)/(self.f3_max - self.f3_min + 1e-9)
            objs = np.vstack([f1_norm, f2_norm, f3_norm]).T

            # 计算指标
            self.hypervolume_list.append(HV(ref_point=self.global_ref_point)(objs))
            self.gd_list.append(GD(self.reference_pareto)(objs))
            self.spread_list.append(self.manual_spread(objs))
            self.pareto_size_list.append(len(gen_pareto))

    def manual_spread(self, front):
        if len(front) < 2:
            return 0.0

        # 对前沿和参考前沿按f1排序
        front_sorted = front[np.argsort(front[:, 0])]
        ref_sorted = self.reference_pareto[np.argsort(self.reference_pareto[:, 0])]

        # 计算相邻距离
        distances = np.linalg.norm(front_sorted[1:] - front_sorted[:-1], axis=1)
        d_mean = np.mean(distances) if len(distances) > 0 else 0.0

        # 首尾距离
        df = np.linalg.norm(front_sorted[0] - ref_sorted[0]) if len(ref_sorted) > 0 else 0.0
        dl = np.linalg.norm(front_sorted[-1] - ref_sorted[-1]) if len(ref_sorted) > 0 else 0.0

        # 总Spread
        spread = (df + dl + np.sum(np.abs(distances - d_mean))) / (df + dl + len(distances)*d_mean + 1e-12)
        return spread

    def visualize_pareto_front(self, pareto_front, generation):
        self.ax.clear()

        # 提取原始目标值
        f1 = np.array([ind.objectives[0] for ind in pareto_front])
        f2 = np.array([np.degrees(np.arccos(np.clip(abs(ind.objectives[1]), 0.0, 1.0))) for ind in pareto_front])
        f3 = np.array([ind.objectives[2] for ind in pareto_front])

        # 归一化
        f1_norm = self.normalize(f1)
        f2_norm = self.normalize(f2)
        f3_norm = self.normalize(f3)

        wf1, wf2, wf3 = 0.35, 0.2, 0.45
        weighted_score = [-wf1 * a + wf2 * b - wf3 * c for a, b, c in zip(f1_norm, f2_norm, f3_norm)]

        ws_min, ws_max = min(weighted_score), max(weighted_score)
        ws_norm = [(x - ws_min) / (ws_max - ws_min) if ws_max > ws_min else 0.5 for x in weighted_score]

        cmap = LinearSegmentedColormap.from_list("score_map", ["red", "yellow", "green"])
        scatter = self.ax.scatter(f1, f2, f3_norm, c=ws_norm, cmap=cmap, s=30, vmin=0, vmax=1)

        self.ax.set_xlabel("Path Length (mm)")
        self.ax.set_ylabel("Entry Angle (°)")
        self.ax.set_zlabel("Risk Cost")
        self.ax.set_title(f"Generation {generation + 1} - Pareto Front")

        if self.cbar is None:
            cax = self.fig.add_axes([0.88, 0.15, 0.02, 0.7])
            self.cbar = self.fig.colorbar(scatter, cax=cax)
            self.cbar.set_label("Path Score", rotation=90, labelpad=15)
        else:
            self.cbar.update_normal(scatter)

    def normalize(self, values):
        max_v = max(values)
        min_v = min(values)
        return [(v - min_v) / (max_v - min_v) if (max_v - min_v) > 0 else 0.0 for v in values]

    def plot_all_indicators(self):
        generations = range(1, self.num_of_generations + 1)

        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()

        axs[0].plot(generations, self.hypervolume_list, label='HV', color='darkred')
        axs[0].set_title("Hypervolume (↑)")

        axs[1].plot(generations, self.gd_list, label='GD', color='green')
        axs[1].set_title("Generational Distance (↓)")

        axs[2].plot(generations, self.spread_list, label='Spread', color='orange')
        axs[2].set_title("Spread / Diversity (↓)")

        axs[3].plot(generations, self.pareto_size_list, label='Pareto Size', color='blue')
        axs[3].set_title("Size of Pareto Front (↑)")

        for ax in axs:
            ax.set_xlabel("Generation")
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()