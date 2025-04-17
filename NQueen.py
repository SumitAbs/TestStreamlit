# Single Board but Nice Representation

# import streamlit as st
# import random
# import math

# # ========== Utility Functions ==========

# def get_conflicts(state):
#     conflicts = 0
#     n = len(state)
#     for i in range(n):
#         for j in range(i + 1, n):
#             if abs(state[i] - state[j]) == abs(i - j):
#                 conflicts += 1
#     return conflicts

# def get_initial_state(n):
#     state = list(range(n))
#     random.shuffle(state)
#     return state

# def get_neighbor(state):
#     neighbor = state[:]
#     i, j = random.sample(range(len(state)), 2)
#     neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
#     return neighbor

# def simulated_annealing(n, initial_temp=100, cooling_rate=0.99, min_temp=0.1):
#     current = get_initial_state(n)
#     current_conflicts = get_conflicts(current)
#     temp = initial_temp

#     while temp > min_temp and current_conflicts > 0:
#         neighbor = get_neighbor(current)
#         neighbor_conflicts = get_conflicts(neighbor)
#         delta = neighbor_conflicts - current_conflicts

#         if delta < 0 or random.random() < math.exp(-delta / temp):
#             current = neighbor
#             current_conflicts = neighbor_conflicts

#         temp *= cooling_rate

#     return current if current_conflicts == 0 else None

# def modified_sa_to_hc(n):
#     current = get_initial_state(n)
#     current_conflicts = get_conflicts(current)

#     while current_conflicts > 0:
#         neighbor = get_neighbor(current)
#         neighbor_conflicts = get_conflicts(neighbor)

#         if neighbor_conflicts < current_conflicts:
#             current = neighbor
#             current_conflicts = neighbor_conflicts

#         if current_conflicts == 0:
#             break

#     return current if current_conflicts == 0 else None

# def draw_board(state):
#     n = len(state)
#     board = ""
#     for row in range(n):
#         line = ["🟦"] * n
#         if state:
#             line[state[row]] = "👑"
#         board += "".join(line) + "\n"
#     return board

# # ========== Streamlit App ==========

# st.title("♟️ N-Queens Solver using Simulated Annealing & Hill Climbing")

# n = st.slider("Select Board Size (N):", 4, 20, 8)
# algorithm = st.radio("Select Algorithm:", ["Simulated Annealing", "Modified SA (Hill Climbing)"])

# if algorithm == "Simulated Annealing":
#     temp = st.number_input("Initial Temperature:", min_value=0.1, value=100.0)
#     cooling = st.slider("Cooling Rate:", 0.80, 0.999, 0.99)
#     min_temp = st.number_input("Minimum Temperature:", min_value=0.001, value=0.1)

# if st.button("Solve"):
#     if algorithm == "Simulated Annealing":
#         solution = simulated_annealing(n, initial_temp=temp, cooling_rate=cooling, min_temp=min_temp)
#     else:
#         solution = modified_sa_to_hc(n)

#     if solution:
#         st.success("✅ Solution Found!")
#         st.code(draw_board(solution))
#     else:
#         st.error("❌ No solution found. Try again or change parameters.")

# ============================= New ==================================

# # New code with improved representation and functionality
# import streamlit as st
# st.set_page_config(page_title="N-Queens Solver", layout="wide")

# import random
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# import io
# import base64
# import time

# # Set the style for seaborn
# sns.set(style="whitegrid")
# sns.set_palette("muted")

# # ========== Utility Functions ==========
# def draw_board(solution):
#     n = len(solution)
#     board = np.zeros((n, n))
#     for row in range(n):
#         board[row, solution[row]] = 1
#     return board

# def simulated_annealing(n, initial_temp, cooling_rate, min_temp):
#     current = get_initial_state(n)
#     current_conflicts = get_conflicts(current)
#     temp = initial_temp

#     while temp > min_temp and current_conflicts > 0:
#         neighbor = get_neighbor(current)
#         neighbor_conflicts = get_conflicts(neighbor)
#         delta = neighbor_conflicts - current_conflicts

#         if delta < 0 or random.random() < math.exp(-delta / temp):
#             current = neighbor
#             current_conflicts = neighbor_conflicts

#         temp *= cooling_rate

#     return current if current_conflicts == 0 else None

# def get_initial_state(n):
#     state = list(range(n))
#     random.shuffle(state)
#     return state

# def get_conflicts(state):
#     conflicts = 0
#     n = len(state)
#     for i in range(n):
#         for j in range(i + 1, n):
#             if abs(state[i] - state[j]) == abs(i - j):
#                 conflicts += 1
#     return conflicts

# def get_neighbor(state):
#     neighbor = state[:]
#     i, j = random.sample(range(len(state)), 2)
#     neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
#     return neighbor

# def modified_sa_to_hc(n):
#     current = get_initial_state(n)
#     current_conflicts = get_conflicts(current)

#     while current_conflicts > 0:
#         neighbor = get_neighbor(current)
#         neighbor_conflicts = get_conflicts(neighbor)

#         if neighbor_conflicts < current_conflicts:
#             current = neighbor
#             current_conflicts = neighbor_conflicts

#         if current_conflicts == 0:
#             break

#     return current if current_conflicts == 0 else None

# def draw_board_image(board):
#     n = board.shape[0]
#     fig, ax = plt.subplots(figsize=(n, n))
#     sns.heatmap(board, annot=False, fmt=".0f", cmap="Blues", cbar=False, linewidths=0.5, linecolor='black', ax=ax)
#     ax.set_xticklabels(range(n), fontsize=12)
#     ax.set_yticklabels(range(n), fontsize=12)
#     ax.set_xlabel("Columns", fontsize=14)
#     ax.set_ylabel("Rows", fontsize=14)
#     ax.set_title("N-Queens Board Representation", fontsize=16)
#     plt.tight_layout()

#     # Save the figure to a BytesIO object
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)

#     # Encode the image to base64 for display in Streamlit
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close(fig)  # Close the figure to free memory
#     return img_str

# def display_image(image_str, col):
#     image_html = f'<img src="data:image/png;base64,{image_str}" alt="N-Queens Board" style="width:100%; height:auto;">'
#     col.markdown(image_html, unsafe_allow_html=True)

# def display_board(board, col):
#     image_str = draw_board_image(board)
#     display_image(image_str, col)

# def display_solution(solution, algorithm_name, col):
#     col.subheader(f"Algorithm: {algorithm_name}")
#     if solution:
#         col.success("✅ Solution Found!")
#         board = draw_board(solution)
#         display_board(board, col)
#     else:
#         col.error("❌ No solution found. Try again or change parameters.")

# def display_progress_bar():
#     progress_bar = st.progress(0)
#     for i in range(100):
#         time.sleep(0.05)  # Simulate some processing time
#         progress_bar.progress(i + 1)
#     progress_bar.empty()  # Clear the progress bar after completion

# def display_sidebar():
#     st.sidebar.title("N-Queens Solver Parameters")
#     n = st.sidebar.slider("Select Board Size (N):", 4, 20, 8)
#     temp = st.sidebar.number_input("Initial Temperature (SA):", min_value=0.1, value=100.0)
#     cooling = st.sidebar.slider("Cooling Rate (SA):", 0.80, 0.999, 0.99)
#     min_temp = st.sidebar.number_input("Minimum Temperature (SA):", min_value=0.001, value=0.1)
#     return n, temp, cooling, min_temp

# def main():
#     st.title("♟️ N-Queens Solver using Multiple Algorithms")
#     n, temp, cooling, min_temp = display_sidebar()

#     if st.button("Solve"):
#         display_progress_bar()  # Show progress bar while solving

#         # Create columns for displaying results side by side
#         col1, col2 = st.columns(2)

#         # Solve using Simulated Annealing
#         sa_solution = simulated_annealing(n, initial_temp=temp, cooling_rate=cooling, min_temp=min_temp)
#         display_solution(sa_solution, "Simulated Annealing", col1)

#         # Solve using Modified SA to Hill Climbing
#         hc_solution = modified_sa_to_hc(n)
#         display_solution(hc_solution, "Modified SA (Hill Climbing)", col2)

# if __name__ == "__main__":
#     main()
#     # st.set_page_config(page_title="N-Queens Solver", layout="wide")
#     st.markdown("""
#         <style>
#             .css-1aumxhk {
#                 background-color: #f0f0f5;
#             }
#         </style>
#     """, unsafe_allow_html=True)



# =========================================================Correct ====================================================
# New code with improved representation and functionality
# import streamlit as st
# st.set_page_config(page_title="N-Queens Solver", layout="wide")

# import random
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from PIL import Image
# import io
# import base64
# import time

# # Set the style for seaborn
# sns.set(style="whitegrid")
# sns.set_palette("muted")

# # ========== Utility Functions ==========
# def draw_board(solution):
#     n = len(solution)
#     board = np.zeros((n, n))
#     for row in range(n):
#         board[row, solution[row]] = 1
#     return board

# def get_initial_state(n):
#     state = list(range(n))
#     random.shuffle(state)
#     return state

# def get_conflicts(state):
#     conflicts = 0
#     n = len(state)
#     for i in range(n):
#         for j in range(i + 1, n):
#             if abs(state[i] - state[j]) == abs(i - j):
#                 conflicts += 1
#     return conflicts

# def get_neighbor(state):
#     neighbor = state[:]
#     i, j = random.sample(range(len(state)), 2)
#     neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
#     return neighbor

# def simulated_annealing(n, initial_temp, cooling_rate, min_temp, max_steps=5000):
#     current = get_initial_state(n)
#     current_conflicts = get_conflicts(current)
#     temp = initial_temp
#     steps = 0

#     while temp > min_temp and current_conflicts > 0 and steps < max_steps:
#         neighbor = get_neighbor(current)
#         neighbor_conflicts = get_conflicts(neighbor)
#         delta = neighbor_conflicts - current_conflicts

#         if delta < 0 or random.random() < math.exp(-delta / temp):
#             current = neighbor
#             current_conflicts = neighbor_conflicts

#         temp *= cooling_rate
#         steps += 1

#     return current if current_conflicts == 0 else None

# def modified_sa_to_hc(n, max_steps=5000):
#     current = get_initial_state(n)
#     current_conflicts = get_conflicts(current)
#     steps = 0

#     while current_conflicts > 0 and steps < max_steps:
#         neighbor = get_neighbor(current)
#         neighbor_conflicts = get_conflicts(neighbor)

#         if neighbor_conflicts < current_conflicts:
#             current = neighbor
#             current_conflicts = neighbor_conflicts

#         steps += 1

#     return current if current_conflicts == 0 else None

# def hill_climbing(n, max_iter=1000):
#     current = get_initial_state(n)
#     current_conflicts = get_conflicts(current)

#     for _ in range(max_iter):
#         neighbor = get_neighbor(current)
#         neighbor_conflicts = get_conflicts(neighbor)

#         if neighbor_conflicts < current_conflicts:
#             current = neighbor
#             current_conflicts = neighbor_conflicts

#         if current_conflicts == 0:
#             break

#     return current if current_conflicts == 0 else None

# def draw_board_image(board):
#     n = board.shape[0]
#     fig, ax = plt.subplots(figsize=(n, n))
#     sns.heatmap(board, annot=False, fmt=".0f", cmap="Blues", cbar=False, linewidths=0.5, linecolor='black', ax=ax)
#     ax.set_xticklabels(range(n), fontsize=12)
#     ax.set_yticklabels(range(n), fontsize=12)
#     ax.set_xlabel("Columns", fontsize=14)
#     ax.set_ylabel("Rows", fontsize=14)
#     ax.set_title("N-Queens Board Representation", fontsize=16)
#     plt.tight_layout()

#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     img_str = base64.b64encode(buf.read()).decode('utf-8')
#     plt.close(fig)
#     return img_str

# def display_image(image_str, col):
#     image_html = f'<img src="data:image/png;base64,{image_str}" alt="N-Queens Board" style="width:100%; height:auto;">'
#     col.markdown(image_html, unsafe_allow_html=True)

# def display_board(board, col):
#     image_str = draw_board_image(board)
#     display_image(image_str, col)

# def display_solution(solution, algorithm_name, col):
#     col.subheader(f"Algorithm: {algorithm_name}")
#     if solution:
#         col.success("✅ Solution Found!")
#         board = draw_board(solution)
#         display_board(board, col)
#     else:
#         col.error("❌ No solution found. Try again or change parameters.")

# def display_progress_bar():
#     progress_bar = st.progress(0)
#     for i in range(100):
#         time.sleep(0.01)
#         progress_bar.progress(i + 1)
#     progress_bar.empty()

# def display_sidebar():
#     st.sidebar.title("N-Queens Solver Parameters")
#     n = st.sidebar.slider("Select Board Size (N):", 4, 20, 8)
#     temp = st.sidebar.number_input("Initial Temperature (SA):", min_value=0.1, value=100.0)
#     cooling = st.sidebar.slider("Cooling Rate (SA):", 0.80, 0.999, 0.99)
#     min_temp = st.sidebar.number_input("Minimum Temperature (SA):", min_value=0.001, value=0.1)
#     return n, temp, cooling, min_temp

# def main():
#     st.title("♟️ N-Queens Solver using Multiple Algorithms")
#     n, temp, cooling, min_temp = display_sidebar()

#     if st.button("Solve"):
#         display_progress_bar()

#         col1, col2, col3 = st.columns(3)

#         sa_solution = simulated_annealing(n, initial_temp=temp, cooling_rate=cooling, min_temp=min_temp, max_steps=5000)
#         display_solution(sa_solution, "Simulated Annealing", col1)

#         modified_hc_solution = modified_sa_to_hc(n, max_steps=5000)
#         display_solution(modified_hc_solution, "Modified SA (HC)", col2)

#         hc_solution = hill_climbing(n, max_iter=1000)
#         display_solution(hc_solution, "Classic Hill Climbing", col3)

# if __name__ == "__main__":
#     main()
#     st.markdown("""
#         <style>
#             .css-1aumxhk {
#                 background-color: #f0f0f5;
#             }
#         </style>
#     """, unsafe_allow_html=True)


# With Graphics
import streamlit as st
st.set_page_config(page_title="N-Queens Solver", layout="wide")

import random
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import time

# Set the style for seaborn
sns.set(style="whitegrid")
sns.set_palette("muted")

# ========== Utility Functions ==========

def get_initial_state(n):
    state = list(range(n))
    random.shuffle(state)
    return state


def get_conflicts(state):
    conflicts = 0
    n = len(state)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(state[i] - state[j]) == abs(i - j):
                conflicts += 1
    return conflicts


def get_neighbor(state):
    neighbor = state[:]
    i, j = random.sample(range(len(state)), 2)
    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
    return neighbor


def simulated_annealing(n, initial_temp, cooling_rate, min_temp, max_steps=5000):
    current = get_initial_state(n)
    current_conflicts = get_conflicts(current)
    temp = initial_temp
    steps = 0
    conflict_history = [current_conflicts]
    start_time = time.time()

    while temp > min_temp and current_conflicts > 0 and steps < max_steps:
        neighbor = get_neighbor(current)
        neighbor_conflicts = get_conflicts(neighbor)
        delta = neighbor_conflicts - current_conflicts
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current = neighbor
            current_conflicts = neighbor_conflicts
        conflict_history.append(current_conflicts)
        temp *= cooling_rate
        steps += 1

    end_time = time.time()
    return current, steps, end_time - start_time, conflict_history


def modified_sa_to_hc(n, max_steps=5000):
    current = get_initial_state(n)
    current_conflicts = get_conflicts(current)
    steps = 0
    conflict_history = [current_conflicts]
    start_time = time.time()

    while current_conflicts > 0 and steps < max_steps:
        neighbor = get_neighbor(current)
        neighbor_conflicts = get_conflicts(neighbor)
        if neighbor_conflicts < current_conflicts:
            current = neighbor
            current_conflicts = neighbor_conflicts
        conflict_history.append(current_conflicts)
        steps += 1

    end_time = time.time()
    return current, steps, end_time - start_time, conflict_history


def hill_climbing(n, max_iter=1000):
    current = get_initial_state(n)
    current_conflicts = get_conflicts(current)
    steps = 0
    conflict_history = [current_conflicts]
    start_time = time.time()

    for _ in range(max_iter):
        neighbor = get_neighbor(current)
        neighbor_conflicts = get_conflicts(neighbor)
        if neighbor_conflicts < current_conflicts:
            current = neighbor
            current_conflicts = neighbor_conflicts
        conflict_history.append(current_conflicts)
        steps += 1
        if current_conflicts == 0:
            break

    end_time = time.time()
    return current, steps, end_time - start_time, conflict_history


def draw_board(solution):
    n = len(solution)
    board = np.zeros((n, n))
    for row in range(n):
        board[row, solution[row]] = 1
    return board


def draw_board_image(board):
    n = board.shape[0]
    fig, ax = plt.subplots(figsize=(n, n))
    sns.heatmap(board, annot=False, fmt=".0f", cmap="Blues", cbar=False,
                linewidths=0.5, linecolor='black', ax=ax)
    ax.set_xticks(np.arange(n)+0.5)
    ax.set_yticks(np.arange(n)+0.5)
    ax.set_xticklabels(range(n), fontsize=12)
    ax.set_yticklabels(range(n), fontsize=12)
    ax.set_xlabel("Columns", fontsize=14)
    ax.set_ylabel("Rows", fontsize=14)
    ax.set_title("N-Queens Board Representation", fontsize=16)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def display_solution(solution_tuple, algorithm_name, col):
    sol, steps, elapsed, conflict_history = solution_tuple
    col.subheader(algorithm_name)
    if sol is not None:
        col.success("✅ Solution Found!")
        board = draw_board(sol)
        img_str = draw_board_image(board)
        col.image(base64.b64decode(img_str), use_container_width=True)
        col.write(f"Steps: {steps}")
        col.write(f"Time: {elapsed:.4f} s")
        col.write(f"Final Conflicts: {conflict_history[-1]}")
        # Conflict history graph
        fig, ax = plt.subplots()
        ax.plot(conflict_history, label="Conflicts")
        ax.set_xlabel("Step")
        ax.set_ylabel("Conflicts")
        ax.set_title(f"Conflict Reduction: {algorithm_name}")
        ax.legend()
        col.pyplot(fig)
    else:
        col.error("❌ No solution found.")


def display_sidebar():
    st.sidebar.title("Parameters")
    n = st.sidebar.slider("Board Size N", 4, 20, 8)
    temp = st.sidebar.number_input("Initial Temperature", value=100.0)
    cooling = st.sidebar.slider("Cooling Rate", 0.80, 0.999, 0.99)
    min_temp = st.sidebar.number_input("Min Temperature", value=0.1)
    return n, temp, cooling, min_temp


def plot_performance(data):
    df = data
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    sns.barplot(x="Algorithm", y="Steps", data=df, ax=ax[0])
    ax[0].set_title("Steps")
    sns.barplot(x="Algorithm", y="Time", data=df, ax=ax[1])
    ax[1].set_title("Time (s)")
    st.pyplot(fig)


def main():
    st.title("N-Queens Solver: SA vs HC vs Mod-SA")
    n, temp, cooling, min_temp = display_sidebar()
    if st.sidebar.button("Solve All"):
        sa = simulated_annealing(n, temp, cooling, min_temp)
        mod = modified_sa_to_hc(n)
        hc = hill_climbing(n)
        cols = st.columns(3)
        display_solution(sa, "Simulated Annealing", cols[0])
        display_solution(mod, "Mod SA→HC", cols[1])
        display_solution(hc, "Hill Climbing", cols[2])
        st.markdown("---")
        st.subheader("Performance Comparison")
        data = {
            "Algorithm": ["SA", "Mod-SA", "HC"],
            "Steps": [sa[1], mod[1], hc[1]],
            "Time": [sa[2], mod[2], hc[2]]
        }
        plot_performance(data)

if __name__ == "__main__":
    main()
