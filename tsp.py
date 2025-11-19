import streamlit as st
import random
import math
import matplotlib.pyplot as plt
import io

# ---------------------------
# UPDATED getCity() TO READ UPLOADED FILE
# ---------------------------

def getCity(uploaded_file):
    cities = []
    content = uploaded_file.read().decode("utf-8").strip().split("\n")

    for line in content:
        node_city_val = line.split()
        cities.append(
            [node_city_val[0], float(node_city_val[1]), float(node_city_val[2])]
        )
    return cities


# ---------------------------
# ORIGINAL FUNCTIONS
# ---------------------------

def calcDistance(cities):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]
        d = math.sqrt((cityB[1] - cityA[1]) ** 2 + (cityB[2] - cityA[2]) ** 2)
        total_sum += d

    cityA = cities[0]
    cityB = cities[-1]
    d = math.sqrt((cityB[1] - cityA[1]) ** 2 + (cityB[2] - cityA[2]) ** 2)
    total_sum += d
    return total_sum


def selectPopulation(cities, size):
    population = []
    for i in range(size):
        c = cities.copy()
        random.shuffle(c)
        distance = calcDistance(c)
        population.append([distance, c])
    fitest = sorted(population)[0]
    return population, fitest


def geneticAlgorithm(
    population, lenCities, TOURNAMENT_SELECTION_SIZE,
    MUTATION_RATE, CROSSOVER_RATE, TARGET, progress_callback=None):

    gen_number = 0
    
    for i in range(200):
        new_population = []

        # Elitism
        new_population.append(sorted(population)[0])
        new_population.append(sorted(population)[1])

        for i in range(int((len(population) - 2) / 2)):

            # CROSSOVER
            if random.random() < CROSSOVER_RATE:
                parent1 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]
                parent2 = sorted(random.choices(population, k=TOURNAMENT_SELECTION_SIZE))[0]

                point = random.randint(0, lenCities - 1)

                child1 = parent1[1][0:point]
                for j in parent2[1]:
                    if j not in child1:
                        child1.append(j)

                child2 = parent2[1][0:point]
                for j in parent1[1]:
                    if j not in child2:
                        child2.append(j)
            else:
                child1 = random.choice(population)[1]
                child2 = random.choice(population)[1]

            # MUTATION
            if random.random() < MUTATION_RATE:
                p1, p2 = random.randint(0, lenCities - 1), random.randint(0, lenCities - 1)
                child1[p1], child1[p2] = child1[p2], child1[p1]

                p1, p2 = random.randint(0, lenCities - 1), random.randint(0, lenCities - 1)
                child2[p1], child2[p2] = child2[p2], child2[p1]

            new_population.append([calcDistance(child1), child1])
            new_population.append([calcDistance(child2), child2])

        population = new_population
        gen_number += 1

        # Update Streamlit progress
        if progress_callback:
            progress_callback(gen_number, sorted(population)[0][0])

        if sorted(population)[0][0] < TARGET:
            break

    return sorted(population)[0], gen_number


def plot_route(cities, answer):
    plt.figure(figsize=(10, 6))

    # Plot nodes
    for j in cities:
        plt.plot(j[1], j[2], "ro")
        plt.annotate(j[0], (j[1], j[2]))

    # Draw path
    for i in range(len(answer[1])):
        try:
            A = answer[1][i]
            B = answer[1][i + 1]
            plt.plot([A[1], B[1]], [A[2], B[2]])
        except:
            continue

    # Closing link
    A = answer[1][0]
    B = answer[1][-1]
    plt.plot([A[1], B[1]], [A[2], B[2]])

    plt.title("Optimized Route (Genetic Algorithm)")
    st.pyplot(plt)



# =====================================================
#                 STREAMLIT UI
# =====================================================

st.set_page_config(page_title="Genetic Algorithm - TSP Solver", layout="wide")
st.title("🧬 Genetic Algorithm — Traveling Salesman Problem")
st.markdown("### Visual, Interactive, Realistic TSP Solver")

# File upload section
uploaded_file = st.file_uploader("📄 Upload your TSP .txt file", type=["txt"])

st.sidebar.header("⚙️ Parameters")

POPULATION_SIZE = st.sidebar.slider("Population Size", 200, 5000, 1500)
TOURNAMENT_SELECTION_SIZE = st.sidebar.slider("Tournament Selection Size", 2, 10, 4)
MUTATION_RATE = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
CROSSOVER_RATE = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.9)
TARGET = st.sidebar.slider("Target Distance", 200.0, 600.0, 450.0)

if st.button("🚀 Run Genetic Algorithm"):

    if uploaded_file is None:
        st.error("❌ Please upload a .txt file before running the algorithm.")
        st.stop()

    # Load uploaded cities
    cities = getCity(uploaded_file)

    # Progress containers
    placeholder = st.empty()
    chart_data = []

    def update_progress(gen, best):
        chart_data.append(best)
        placeholder.line_chart(chart_data)

    st.write("### ⏳ Running TSP Optimization...")

    firstPop, firstFitest = selectPopulation(cities, POPULATION_SIZE)

    answer, genNumber = geneticAlgorithm(
        firstPop,
        len(cities),
        TOURNAMENT_SELECTION_SIZE,
        MUTATION_RATE,
        CROSSOVER_RATE,
        TARGET,
        progress_callback=update_progress
    )

    st.success("🎉 Optimization Completed!")

    st.write(f"**Generations:** {genNumber}")
    st.write(f"**Initial Best Distance:** `{firstFitest[0]}`")
    st.write(f"**Optimized Distance:** `{answer[0]}`")

    st.write("### 🗺️ Final Optimized Route")
    plot_route(cities, answer)

    st.write("---")
    st.write("✔️ Built with Streamlit")
