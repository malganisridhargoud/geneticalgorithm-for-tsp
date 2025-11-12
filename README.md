# geneticalgorithm-for-tsp
designing a genetic algorithm for the travellig sales person problem using ml techniques crossover,mutation etc
# 🧬 Traveling Salesman Problem using Genetic Algorithm

> *A heuristic AI approach to find the shortest route that visits every city exactly once.*

---

## 🚀 Project Overview

The **Traveling Salesman Problem (TSP)** is one of the most well-known optimization problems.  
A salesman must visit a set of cities **exactly once** and return to the starting point — while minimizing the total distance traveled.

To solve this, we use a **Genetic Algorithm (GA)** — a biologically inspired optimization technique that mimics natural evolution.  
Through the process of **selection, crossover, and mutation**, the algorithm gradually evolves better solutions over generations.

---

## 🧠 What is a Genetic Algorithm?

A **Genetic Algorithm (GA)** is a heuristic search and optimization technique based on the principles of **natural selection and genetics**.

### ⚙️ GA Phases
1. **Initialization** – Create an initial population of random paths (chromosomes).  
2. **Fitness Calculation** – Evaluate how good each solution is (shorter paths = higher fitness).  
3. **Selection** – Choose the fittest individuals for reproduction.  
4. **Crossover** – Combine genes from two parents to produce offspring.  
5. **Mutation** – Introduce random variations to maintain diversity and avoid local minima.

---

## 🧩 Approach for TSP

- **Cities** → Represented as **genes**  
- **Chromosome** → A complete route visiting all cities once  
- **Fitness Score** → Inverse of the total path length (shorter paths are fitter)

The algorithm iterates through multiple generations, evolving toward the optimal (or near-optimal) path.  
A **cooling variable** controls the number of iterations — gradually decreasing until a termination threshold is reached.

---

## 🧮 Algorithm Steps

```text
1. Initialize the population randomly.
2. Evaluate the fitness of each chromosome.
3. Repeat until termination condition:
   a. Select parent chromosomes.
   b. Perform crossover and mutation.
   c. Evaluate fitness of the new population.
   d. Select survivors for the next generation.
