import streamlit as st
import random
import math
import matplotlib.pyplot as plt

# -----------------------------
# 🎯 Helper Functions
# -----------------------------

def calc_distance(cities):
    """Calculate total distance of a route."""
    total_dist = 0
    for i in range(len(cities) - 1):
        cityA, cityB = cities[i], cities[i + 1]
        total_dist += math.sqrt((cityB[1] - cityA[1]) ** 2 + (cityB[2] - cityA[2]) ** 2)
    # Return to start city
    total_dist += math.sqrt((cities[-1][1] - cities[0][1]) ** 2 + (cities[-1][2] - cities[0][2]) ** 2)
    return total_dist


def get_realistic_cities():
    """
    Simulate real-world city-like coordinates in 2D plane.
    Coordinates mimic a simplified world map.
    """
    return [
        ["Delhi", 90, 80],
        ["Mumbai", 40, 40],
        ["Kolkata", 130, 70],
        ["Chennai", 70, 20],
        ["Bangalore", 60, 10],
        ["Hyderabad", 80, 30],
        ["Ahmedabad", 50, 50],
        ["Pune", 55, 35],
        ["Jaipur", 75, 65],
        ["Lucknow", 100, 75],
    ]


def select_population(cities, size):
    """Create an initial population of random paths."""
    population = []
    for _ in range(size):
        route = cities.copy()
        random.shuffle(route)
        population.append([calc_distance(route), route])
    return population, min(population, key=lambda x: x[0])


def genetic_algorithm(population, num_cities, tournament_size, mutation_rate, crossover_rate, target):
    """Core Genetic Algorithm loop."""
    generations = 0
    while generations < 200:
        new_pop = []
        sorted_pop = sorted(population, key=lambda x: x[0])

        # Keep best 2 (elitism)
        new_pop.extend(sorted_pop[:2])

        # Generate children
        for _ in range((len(population) - 2) // 2):
            parent1 = min(random.choices(population, k=tournament_size), key=lambda x: x[0])
            parent2 = min(random.choices(population, k=tournament_size), key=lambda x: x[0])

            # Crossover
            if random.random() < crossover_rate:
                point = random.randint(0, num_cities - 1)
                child1 = parent1[1][:point] + [c for c in parent2[1] if c not in parent1[1][:point]]
                child2 = parent2[1][:point] + [c for c in parent1[1] if c not in parent2[1][:point]]
            else:
                child1, child2 = parent1[1][:], parent2[1][:]

            # Mutation (swap two cities)
            if random.random() < mutation_rate:
                i, j = random.sample(range(num_cities), 2)
                child1[i], child1[j] = child1[j], child1[i]
            if random.random() < mutation_rate:
                i, j = random.sample(range(num_cities), 2)
                child2[i], child2[j] = child2[j], child2[i]

            new_pop.append([calc_distance(child1), child1])
            new_pop.append([calc_distance(child2), child2])

        population = new_pop
        generations += 1

        # Stop if target achieved
        if sorted_pop[0][0] < target:
            break

    return min(population, key=lambda x: x[0]), generations


# -----------------------------
# 🖼️ Visualization
# -----------------------------

def draw_route(cities, best_path):
    """Draw the cities and route on a 2D plane."""
    plt.figure(figsize=(8, 6))
    route = best_path[1]
    x = [city[1] for city in route]
    y = [city[2] for city in route]

    # Draw route
    plt.plot(x + [x[0]], y + [y[0]], 'b-', linewidth=2, label="Path")
    plt.scatter(x, y, color="red", s=100, zorder=5)

    # Annotate city names
    for city in route:
        plt.text(city[1] + 1, city[2] + 1, city[0], fontsize=10, color="black")

    plt.title("🧭 Traveling Salesman Path (Shortest Route Found)")
    plt.xlabel("X-Coordinate (like longitude)")
    plt.ylabel("Y-Coordinate (like latitude)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)


# -----------------------------
# 🧠 Streamlit Interface
# -----------------------------

st.set_page_config(page_title="TSP - Genetic Algorithm", page_icon="🧬", layout="centered")

st.title("🧬 Traveling Salesman Problem (TSP) — Genetic Algorithm Visualization")
st.markdown("""
This app helps you **visualize how a Genetic Algorithm finds the shortest route**
for a salesman who must visit all cities once and return home.  
Imagine each city as a dot on a map — we want to connect all dots using the shortest line possible!
""")

# Sidebar controls
st.sidebar.header("⚙️ GA Settings")
pop_size = st.sidebar.slider("Population Size", 100, 2000, 500, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.9)
tournament_size = st.sidebar.slider("Tournament Size", 2, 10, 4)
target_distance = st.sidebar.number_input("Target Distance", 100.0, 1000.0, 400.0)

if st.button("🚀 Start Algorithm"):
    with st.spinner("Evolving shortest path... please wait!"):
        cities = get_realistic_cities()
        population, fittest = select_population(cities, pop_size)
        best_solution, generations = genetic_algorithm(
            population,
            len(cities),
            tournament_size,
            mutation_rate,
            crossover_rate,
            target_distance,
        )

    st.success(f"✅ Algorithm completed in {generations} generations!")
    st.metric("Initial Distance", f"{fittest[0]:.2f}")
    st.metric("Optimized Distance", f"{best_solution[0]:.2f}")

    st.markdown("### 🗺️ Visualizing the Shortest Route:")
    draw_route(cities, best_solution)

    st.info("""
**Explanation for Kids 👧🧒:**  
- Each red dot is a *city*.  
- The blue line shows the *salesman's travel route*.  
- The algorithm keeps improving until it finds the *shortest total path* connecting all cities!  
""")

else:
    st.info("👈 Adjust parameters and click **Start Algorithm** to begin!")

st.caption("Created by **Sridhar Goud Malgani** — Simple, visual explanation of Genetic Algorithm for TSP")

