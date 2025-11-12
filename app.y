import streamlit as st
import random
import math
import plotly.graph_objects as go
import pandas as pd

# -----------------------------
# 🎯 Helper Functions
# -----------------------------

def calc_distance(cities):
    """Calculate total distance of a path."""
    total_dist = 0
    for i in range(len(cities) - 1):
        cityA, cityB = cities[i], cities[i + 1]
        total_dist += math.sqrt((cityB[1] - cityA[1]) ** 2 + (cityB[2] - cityA[2]) ** 2)
    # Return to starting city
    total_dist += math.sqrt((cities[-1][1] - cities[0][1]) ** 2 + (cities[-1][2] - cities[0][2]) ** 2)
    return total_dist


def generate_random_cities(num_cities):
    """Generate random cities (latitude, longitude) to simulate map coordinates."""
    cities = []
    for i in range(num_cities):
        city_name = f"City-{i+1}"
        # Simulate coordinates (latitude, longitude)
        x, y = random.uniform(10, 50), random.uniform(60, 120)
        cities.append([city_name, x, y])
    return cities


def select_population(cities, size):
    """Create initial population with random paths."""
    population = []
    for _ in range(size):
        shuffled = cities.copy()
        random.shuffle(shuffled)
        population.append([calc_distance(shuffled), shuffled])
    return population, min(population, key=lambda x: x[0])


def genetic_algorithm(population, len_cities, tournament_size, mutation_rate, crossover_rate, target):
    """Main GA function."""
    generations = 0
    while generations < 200:
        new_population = []
        sorted_pop = sorted(population, key=lambda x: x[0])
        new_population.extend(sorted_pop[:2])  # Elitism

        for _ in range((len(population) - 2) // 2):
            # Parent selection
            parent1 = min(random.choices(population, k=tournament_size), key=lambda x: x[0])
            parent2 = min(random.choices(population, k=tournament_size), key=lambda x: x[0])

            # Crossover
            if random.random() < crossover_rate:
                point = random.randint(0, len_cities - 1)
                child1 = parent1[1][:point] + [c for c in parent2[1] if c not in parent1[1][:point]]
                child2 = parent2[1][:point] + [c for c in parent1[1] if c not in parent2[1][:point]]
            else:
                child1, child2 = parent1[1][:], parent2[1][:]

            # Mutation
            if random.random() < mutation_rate:
                i, j = random.sample(range(len_cities), 2)
                child1[i], child1[j] = child1[j], child1[i]
            if random.random() < mutation_rate:
                i, j = random.sample(range(len_cities), 2)
                child2[i], child2[j] = child2[j], child2[i]

            new_population.append([calc_distance(child1), child1])
            new_population.append([calc_distance(child2), child2])

        population = new_population
        generations += 1

        if sorted(population, key=lambda x: x[0])[0][0] < target:
            break

    return min(population, key=lambda x: x[0]), generations


# -----------------------------
# 🌍 Visualization Function
# -----------------------------

def plot_route(cities, best_path):
    df = pd.DataFrame(best_path[1], columns=["City", "lat", "lon"])

    fig = go.Figure()

    # Draw the route lines
    fig.add_trace(go.Scattergeo(
        lon=df["lon"], lat=df["lat"],
        mode="lines+markers+text",
        line=dict(width=2, color="blue"),
        marker=dict(size=8, color="red"),
        text=df["City"],
        textposition="top center"
    ))

    # Close the loop
    fig.add_trace(go.Scattergeo(
        lon=[df["lon"].iloc[-1], df["lon"].iloc[0]],
        lat=[df["lat"].iloc[-1], df["lat"].iloc[0]],
        mode="lines",
        line=dict(width=2, color="gray")
    ))

    fig.update_layout(
        title="🧭 Optimal Route (Genetic Algorithm for TSP)",
        geo=dict(
            scope="world",
            projection_type="natural earth",
            showland=True,
            landcolor="rgb(217, 217, 217)",
            subunitwidth=1,
            countrywidth=1,
            showlakes=True,
            lakecolor="LightBlue",
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig


# -----------------------------
# 🧠 Streamlit UI
# -----------------------------

st.set_page_config(page_title="TSP Genetic Algorithm", page_icon="🧬", layout="wide")

st.title("🧬 Traveling Salesman Problem using Genetic Algorithm")
st.markdown("### Explore how a Genetic Algorithm optimizes the shortest route between cities on a world map 🌍")

st.sidebar.header("⚙️ Algorithm Parameters")
num_cities = st.sidebar.slider("Number of Cities", 5, 30, 10)
pop_size = st.sidebar.slider("Population Size", 100, 2000, 500, 100)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.0, 1.0, 0.9)
tournament_size = st.sidebar.slider("Tournament Size", 2, 10, 4)
target_distance = st.sidebar.number_input("Target Distance", 100.0, 1000.0, 400.0)

if st.button("🚀 Run Genetic Algorithm"):
    with st.spinner("Evolving the optimal path... Please wait ⏳"):
        cities = generate_random_cities(num_cities)
        population, fittest = select_population(cities, pop_size)
        best_solution, generations = genetic_algorithm(
            population,
            len(cities),
            tournament_size,
            mutation_rate,
            crossover_rate,
            target_distance,
        )

    st.success(f"✅ Completed in {generations} generations!")
    st.metric("Initial Distance", f"{fittest[0]:.2f}")
    st.metric("Optimized Distance", f"{best_solution[0]:.2f}")

    fig = plot_route(cities, best_solution)
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Developed by **Sridhar Goud Malgani** — Interactive AI Visualization using Streamlit + Genetic Algorithm")

else:
    st.info("👈 Adjust parameters and click **Run Genetic Algorithm** to start the optimization!")

