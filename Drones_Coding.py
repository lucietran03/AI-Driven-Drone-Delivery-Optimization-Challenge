import heapq
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.gridspec import GridSpec

# Initialize city map
class CityMap:
    # Initialize the city map with a matrix, starting point, and delivery points
    def __init__(self, map_matrix, start, deliveries):
        self.map_matrix = [[0 if cell == '.' else 1 for cell in row] for row in map_matrix] # Convert map matrix: '.' to 0, others to 1
        self.start = start              # Start node
        self.deliveries = deliveries    # List of delivery points
        self.end = deliveries[-1]       # End node: Final delivery point

    # Provide a string representation of the city map instance for easier debugging
    def __repr__(self):
        return f"CityMap(start={self.start}, end={self.end}, deliveries={self.deliveries})"

# Path-finding Function using A* Algorithm
def astar(city_map, start, end, heuristic_func):
    map_matrix = city_map.map_matrix
    rows, cols = len(map_matrix), len(map_matrix[0])
    
    start_node = start
    end_node = end
    
    open_list = []
    heapq.heappush(open_list, (0, start_node))
    
    came_from = {}
    g_score = { (x, y): float('inf') for x in range(rows) for y in range(cols) }
    g_score[start_node] = 0
    f_score = { (x, y): float('inf') for x in range(rows) for y in range(cols) }
    f_score[start_node] = heuristic_func(start_node, end_node)
    
    while open_list:
        current_node = heapq.heappop(open_list)[1]
        
        if current_node == end_node:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start_node)
            path.reverse()
            return path
        
        des = [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        des_man = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        if heuristic_func == manhattan:
            destination = des_man
        else:
            destination = des 
            
        for x, y in destination:
            child_node = (current_node[0] + x, current_node[1] + y)
            
            if 0 <= child_node[0] < rows and 0 <= child_node[1] < cols and map_matrix[child_node[0]][child_node[1]] == 0:
                
                tentative_g_score = g_score[current_node] + 1
                
                if tentative_g_score < g_score[child_node]:
                    came_from[child_node] = current_node
                    g_score[child_node] = tentative_g_score
                    f_score[child_node] = g_score[child_node] + heuristic_func(child_node, end_node)
                    
                    if child_node not in [i[1] for i in open_list]:
                        heapq.heappush(open_list, (f_score[child_node], child_node))
    
    return None

def Chebyshev(start, end):
    """Calculate the Chebyshev distance between the start and end points"""
    return max(abs(start[0] - end[0]), abs(start[1] - end[1]))

def ChooseMap():
    """Prints three maps and lets the user choose one"""
    # Define maps with names and data
    maps = {
        1: {'name': 'Map 1', 'data': [  # Map 1
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  
            ['.', '.', '.', '.', '#', '#', '#', '#', '.', '.'],  
            ['.', '.', '.', '.', '.', '.', '.', '#', '.', '.'],  
            ['.', '#', '#', '#', '#', '.', '.', '#', '.', '.'],  
            ['.', '.', '.', '.', '.', '.', '.', '#', '.', '.'],  
            ['.', '.', '.', '.', '.', '#', '#', '#', '.', '.'],  
            ['.', '#', '#', '#', '.', '#', '.', '.', '.', '.'],  
            ['.', '.', '.', '#', '.', '.', '.', '.', '.', '.'],  
            ['.', '.', '.', '.', '.', '.', '#', '#', '#', '.'],  
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
        ]},
        2: {'name': 'Map 2', 'data': [  # Map 2
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],  
            ['.', '.', '.', '.', '.', '#', '.', '.', '.', '.'],  
            ['.', '#', '.', '.', '.', '#', '#', '#', '#', '.'],  
            ['.', '#', '.', '.', '.', '#', '.', '.', '.', '.'],  
            ['.', '#', '#', '#', '#', '#', '.', '.', '.', '.'],  
            ['.', '.', '.', '#', '.', '.', '.', '.', '.', '.'],  
            ['.', '.', '.', '#', '.', '.', '.', '.', '.', '.'],  
            ['.', '.', '.', '#', '.', '.', '#', '.', '.', '.'],  
            ['.', '.', '.', '.', '.', '.', '#', '.', '.', '.'],  
            ['.', '.', '.', '.', '.', '.', '#', '.', '.', '.']
        ]},
        3: {'name': 'Map 3', 'data': [  # Map 3
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '#', '.', '.', '.', '.', '.'],
            ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.']
        ]}
    }
    
    print("\nPLEASE OPEN VISUAL STUDIO CODE WINDOW IN FULL SCREEN FOR THE BEST EXPERIENCE")
    for map_number in maps:
        print(maps[map_number]['name'], end=':                  ') # Print map names
    print()

    # Print maps in horizontal layout
    for row_index in range(len(maps[1]['data'])):
        for map_number in maps:
            row = maps[map_number]['data'][row_index]
            print(f"|{' '.join(row)}|", end='  ' if map_number < len(maps) else '\n')
    
    # Get and validate user input
    choice = 0
    while choice not in maps:
        try:
            choice = int(input("Enter the number of the map you want to choose (1, 2, or 3): "))
            if choice not in maps:
                print("Invalid choice. Please choose a valid map number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    return choice, maps[choice]['data']

def manhattan(point1, point2):
    """Returns the Manhattan distance between two points"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def euclidean(point1, point2):
    """Returns the Euclidean distance between two points"""
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def visualize_delivery_path(city_map, paths, name):
    # Hình 1: Hiển thị tất cả các đường đi
    fig, ax = plt.subplots(figsize=(8, 6))
    
    vis_map = np.array([row[:] for row in city_map.map_matrix])
    ax.imshow(vis_map, cmap='gray_r', origin='upper')
    ax.set_facecolor('white')

    lines = []
    for path in paths:
        x_coords = [point[1] for point in path]
        y_coords = [point[0] for point in path]
        line, = ax.plot(x_coords, y_coords, marker='o', linestyle='-', label='Path')
        lines.append(line)

    sy, sx = city_map.start
    ax.scatter(sx, sy, c='green', marker='o', edgecolor='black', s=200, zorder=3, label='Start')

    for idx, path in enumerate(paths):
        ey, ex = path[-1]
        if idx < len(paths) - 1:
            ax.scatter(ex, ey, c='yellow', marker='X', edgecolor='black', s=200, zorder=3, label='Stop')
        else:
            ax.scatter(ex, ey, c='red', marker='X', edgecolor='black', s=200, zorder=3, label='End')

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    
    fig.suptitle(f"Delivery Path Visualization ({name})", fontsize=14, y=0.95)
    message = f"Please spend at least {2 * len(paths)} seconds to look at the figure. \nRoutes will be highlighted in order of delivery."
    plt.text(0.5, -0.1, message, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10, color='red')
    
    plt.pause(1 * len(paths))  # Display the message for a period based on number of paths

    def highlight_path(selected_index):
        for idx, line in enumerate(lines):
            if idx == selected_index:
                line.set_linewidth(4)
                line.set_color('red')
            else:
                line.set_linewidth(2)
                line.set_color('blue')
        fig.canvas.draw_idle()

    for i in range(len(paths)):
        highlight_path(i)
        plt.pause(1)  # Wait for 1 second before highlighting the next path

    plt.show()

def unable_to_visualize(name, failed):
    # Hình 2: Hiển thị thông báo không thể giao hàng
    fig, ax = plt.subplots(figsize=(6, 2))  # Chỉnh size figure nhỏ hơn
    ax.text(0.5, 0.5, f'Unable to visualize delivery path to {failed} using {name}!\nPath is empty or invalid.',
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=14, color='red')  # Fontsize nhỏ hơn cho phù hợp
    ax.axis('off')
    plt.show()
 
def generate_evaluation_report(map_number, city_map, execution_times, path_costs, total):
    
    # Dữ liệu từ các dictionary
    start_point = city_map.start
    delivery_point = city_map.deliveries
    end_point = city_map.deliveries[-1]

    # Chuẩn bị dữ liệu cho bảng
    columns = ['', 'Chebyshev', 'Manhattan', 'Euclidean']
    table_data = [
    ['Map Number', map_number, map_number, map_number],  # Map Number
    ['Starting Point', start_point, start_point, start_point],  # Start Point
    ['Delivery Points', delivery_point[:-1], delivery_point[:-1], delivery_point[:-1]],  # Delivery Point
    ['End Point', end_point, end_point, end_point],  # End Point
    ['Total Path Cost', 
     f"{path_costs.get('Chebyshev', 0):.2f}", 
     f"{path_costs.get('Manhattan', 0):.2f}", 
     f"{path_costs.get('Euclidean', 0):.2f}"],  # Total Path Cost
    ['Execution Times', 
     execution_times.get('Chebyshev', ''), 
     execution_times.get('Manhattan', ''), 
     execution_times.get('Euclidean', '')],  # Execution Times
    ['Total Execution Time', 
     total.get('Chebyshev', ''), 
     total.get('Manhattan', ''), 
     total.get('Euclidean', '')]  # Total Execution Times
    ]

    # Tạo figure và axis
    fig = plt.figure(figsize=(12, 6))  # Điều chỉnh kích thước figure
    gs = GridSpec(2, 1, height_ratios=[4, 1])

    # Vẽ bảng
    ax1 = fig.add_subplot(gs[0])
    ax1.axis('tight')
    ax1.axis('off')

    table = ax1.table(cellText=table_data, 
                    colLabels=columns, 
                    loc='center', cellLoc='center', 
                    colColours=['#6699CC', '#6699CC', '#6699CC', '#6699CC'],  # Màu sắc cho tiêu đề cột
                    cellColours=[['#D9EAF7', '#D9EAF7', '#D9EAF7', '#D9EAF7'],  # Màu sắc cho các ô
                                  ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF'],
                                  ['#D9EAF7', '#D9EAF7', '#D9EAF7', '#D9EAF7'],
                                  ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF'],
                                  ['#D9EAF7', '#D9EAF7', '#D9EAF7', '#D9EAF7'],
                                  ['#FFFFFF', '#FFFFFF', '#FFFFFF', '#FFFFFF'],
                                  ['#D9EAF7', '#D9EAF7', '#D9EAF7', '#D9EAF7']])

    # Tùy chỉnh bảng
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  # Điều chỉnh kích thước cột và hàng

    ax1.set_title("Evaluation Report", fontsize=14, pad=5, color = 'red', weight = 'bold')  # Tiêu đề và khoảng cách tiêu đề

    # Thêm thông điệp dưới bảng
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    message = ("With grid map and cross-movement allowed, Chebyshev is the best choice.\n"
               "This reality is also shown through the data in lines 6-7-8.")
    ax2.text(0.5, 0.7, message, ha='center', va='center', fontsize=12, weight='bold')


    # Hiển thị bảng
    plt.show()
                        
def display_instructions():
    # Define the instructions to be displayed
    instructions = """
        *** AI-Driven Drone Delivery Optimization Challenge - Program Usage Instructions ***    
    This program will find the path from the starting point to the ending point on the city map.

    Symbols:
        # = Obstacle
        . = Path
        S = Starting point
        E = End point
        P = Passed point

    The program will display the map with the found path. Press Enter to continue...
    """
    print(instructions) # Print the instructions
    input() # Wait for user to press Enter

def get_point(prompt, map_limits, map_matrix = None, is_start_point=0):
    while True:
        try:
            point = tuple(map(int, input(prompt).split()))
            if len(point) != 2:
                raise ValueError("Please enter exactly two numbers.")
            x, y = point

            # Kiểm tra xem điểm có nằm trong giới hạn bản đồ không
            if not (0 <= x < map_limits[0]) or not (0 <= y < map_limits[1]):
                raise ValueError(f"Coordinates must be within the map limits (0, 0) to ({map_limits[0]-1}, {map_limits[1]-1}).")

            if map_matrix is not None and is_start_point == 1:
                if map_matrix[x][y] == '#':  # Giả sử '#' đại diện cho vật cản
                    print("The starting point is an obstacle. Please choose a different point.")
                    continue

            return point
        except ValueError as e:
            print(f"Invalid input: {e}. Please try again.")

def process_deliveries(city_map, heuristic, name):
    # Initialize variables for tracking delivery progress
    current_location = city_map.start  # Starting location
    total_path_cost = 0  # Total cost of paths
    execution_times = []  # List of execution times
    paths = []  # List of paths taken
    failed_paths = []
    
    for delivery in city_map.deliveries:
        print()  # Print a blank line for readability
        print(f"Delivering to {delivery}, using {name}...")  # Inform user of current delivery

        start_time = time.time()  # Start timing the delivery process

        path = astar(city_map, current_location, delivery, heuristic)  # Find the best path to the delivery point

        time.sleep(1)  # Pause for clarity

        if path:
            # Calculate and update path cost
            if heuristic == euclidean:
                path_cost = sum(euclidean(path[i], path[i + 1]) for i in range(len(path) - 1))
            else:
                path_cost = len(path) - 1
            total_path_cost += path_cost
            paths.append(path)  # Store the path
            print(f"Delivered to {delivery}. Path: {path} with path cost: {path_cost:.2f}")

            print()
            end_time = time.time()  # End timing the delivery process
            execution_time = round(end_time - start_time - 1, 4)  # Adjust for sleep delay
            execution_times.append(execution_time)  # Store execution time
            print(f"Time taken: {execution_time:.4f} seconds")  # Show delivery time
            print()

            current_location = delivery  # Update current location
        else:
            print(f"Unable to deliver to {delivery}.")  # Notify if delivery fails
            failed_paths.append(delivery)
            break  # Exit loop if delivery fails

    print()  # Print a blank line for readability
    print(f"Total path cost: {total_path_cost:.2f}")  # Display total path cost

    return total_path_cost, paths, execution_times, failed_paths
          
def main():
    display_instructions()  # Show the program instructions

    continue_program = True # Flag to control the loop

    # Loop to handle multiple runs of the program
    while continue_program:
        map_number, selected_map = ChooseMap() # Choose a map and get the map data
        
        # If a map was selected
        if selected_map is not None:
            print("\nYou selected:")  # Inform the user of the selected map
            for row in selected_map:
                print(f"[{', '.join(map(str, row))}]") # Print each row of the selected map

            # Get the size of the map
            map_height = len(selected_map)  # Number of rows
            map_width = len(selected_map[0]) if map_height > 0 else 0  # Number of columns (assuming all rows are of equal length)

            # Define map_limits using the size of the map
            map_limits = (map_width, map_height)

            # Get the starting point from the user
            start = get_point("Enter the starting point (e.g., 0 0): ", map_limits, selected_map, 1)
            
            print(f"Start point: {start}")

            deliveries = []  # Initialize list to collect delivery points

            # Get the number of delivery points from the user
            while True:
                try:
                    num_deliveries = int(input("Enter the number of delivery points: "))
                    if num_deliveries < 1:
                        raise ValueError("The number of delivery points must be at least 1.")
                    break
                except ValueError as e:
                    print(f"Invalid input: {e}. Please try again.")

            # Collect each delivery point from the user
            for i in range(num_deliveries):
                if i != num_deliveries - 1:  # Check the endpoint for user-friendly message
                    delivery_point = get_point("Enter a delivery point (e.g., 3 4): ", map_limits)
                    print(f"Delivery point entered: {delivery_point}")  # Print the entered delivery point
                else:
                    delivery_point = get_point("Enter a end point (e.g., 3 4): ", map_limits)
                    print(f"End point entered: {delivery_point}")  # Print the entered end point
                deliveries.append(delivery_point)  # Add to deliveries list

            
            # Create a CityMap object with the selected map and points
            city_map = CityMap(
                map_matrix=selected_map,  # Use selected map data
                start=start,  # Starting point
                deliveries=deliveries  # List of delivery points
            )

            # Initialize variables for tracking delivery progress
            current_location = city_map.start  # Starting location
            execution_times = {'Chebyshev': [], 'Manhattan': [], 'Euclidean': []}  # List of execution times
            paths = {'Chebyshev': [], 'Manhattan': [], 'Euclidean': []}  # List of paths taken
            path_costs = {'Chebyshev': 0, 'Manhattan': 0, 'Euclidean': 0}
            failed_paths = {'Chebyshev': [], 'Manhattan': [], 'Euclidean': []}
            total = {'Chebyshev': 0, 'Manhattan': 0, 'Euclidean': 0}
            
            heuristics = {
                'Chebyshev': Chebyshev,
                'Manhattan': manhattan,
                'Euclidean': euclidean
            }
            # Process each delivery
            for name, heuristic in heuristics.items():
                path_costs[name], paths[name], execution_times[name], failed_paths[name] = process_deliveries(city_map, heuristic, name) # total_path_cost, paths, execution_times, failed_paths
            
            total_execution_times = {key: sum(values) for key, values in execution_times.items()}

            for name, heuristic in heuristics.items():
                if paths.get(name):
                    visualize_delivery_path(city_map, paths[name], name)
                if failed_paths.get(name):
                    unable_to_visualize(name, failed_paths[name])   
                    
            generate_evaluation_report(map_number, city_map, execution_times, path_costs, total_execution_times)
            # (map_number, city_map, execution_times, paths, path_costs              
            
            # Ask user if they want to continue with another map
            continue_choice = int(input("Press '1' to continue with another map, '0' to stop: "))
            if continue_choice == 0:
                continue_program = False  # Exit loop if user chooses to stop

# Call main function if running this script
if __name__ == "__main__":
    main()