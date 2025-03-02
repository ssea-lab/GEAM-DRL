import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

# Predefined workload patterns
workload_list = [
    np.array([1, 2, 1, 2]),
    np.array([2, 3, 3, 4]),
    np.array([5, 7, 6, 6])
]


def random_user_load():
    """Randomly select a user workload from the predefined workload list."""
    return random.choice(workload_list)


def in_coverage(user, server, max_cov):
    """Check if a user is within the coverage radius of a server."""
    return np.linalg.norm(user[:2] - server[:2]) <= max_cov


def get_within_servers(user_list, server_list, x_start, x_end, y_start, y_end, max_cov):
    """
    Determine which servers cover each user. If a user is not covered, regenerate their location.
    :return: Updated user list with valid locations and a mask indicating server coverage for each user.
    """
    users_masks = np.zeros((len(user_list), len(server_list)), dtype=bool)

    def calc_user_within(calc_user, index):
        flag = False
        for j in range(len(server_list)):
            if in_coverage(calc_user, server_list[j], max_cov):
                users_masks[index, j] = 1
                flag = True
        return flag

    for i in tqdm(range(len(user_list))):
        user = user_list[i]
        user_within = calc_user_within(user, i)
        while not user_within:
            user[0] = random.random() * (x_end - x_start) + x_start
            user[1] = random.random() * (y_end - y_start) + y_start
            user_within = calc_user_within(user, i)
    return user_list, users_masks


def draw_data(server_list, user_list):
    """Visualize user and server positions along with coverage areas."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for user in user_list:
        ax.plot(user[0], user[1], 'ro')  # User locations in red
    for server in server_list:
        circle = Circle((server[0], server[1]), server[2], alpha=0.2)  # Server coverage area
        ax.add_patch(circle)
        ax.plot(server[0], server[1], 'bo')  # Server locations in blue
    plt.axis('scaled')
    plt.axis('equal')
    plt.show()


def cal_props_by_seqs(user_seqs, server_seqs, user_allocated_servers, max_cov):
    """Calculate user allocation and server utilization ratios for multiple batches."""
    batch_size = user_seqs.shape[0]
    user_allocated_props = []
    server_used_props = []
    for i in range(batch_size):
        user_seq = user_seqs[i]
        server_seq = server_seqs[i]
        allocated_seq = user_allocated_servers[i]
        user_allocated_prop, server_used_prop = cal_props(user_seq, server_seq, allocated_seq, max_cov)
        user_allocated_props.append(user_allocated_prop)
        server_used_props.append(server_used_prop)
    return user_allocated_props, server_used_props


def can_allocate(workload, capacity):
    """Check if a server has enough capacity to allocate the workload."""
    for i in range(4):
        if capacity[i] < workload[i]:
            return False
    return True


def cal_props(user_seqs, server_seqs, allocated_seq, max_cov):
    """Calculate the proportion of allocated users and utilized servers."""
    tmp_server_capacity = [server_seq[3:] for server_seq in server_seqs]
    user_num = len(user_seqs)
    server_num = len(server_seqs)
    # Track user allocation (-1 means unallocated)
    user_allocate_list = [-1] * user_num
    server_allocate_num = [0] * server_num

    for i in range(user_num):
        user_seq = user_seqs[i]
        server_id = allocated_seq[i]
        if server_id == -1:
            continue

        if in_coverage(user_seq, server_seqs[server_id], max_cov) and can_allocate(user_seq[2:], tmp_server_capacity[server_id]):
            user_allocate_list[i] = server_id
            server_allocate_num[server_id] += 1
            for j in range(4):
                tmp_server_capacity[server_id][j] -= user_seq[2 + j]

    # Calculate the proportion of allocated users
    allocated_user_num = user_num - user_allocate_list.count(-1)
    user_allocated_prop = allocated_user_num / user_num

    # Calculate the proportion of utilized servers
    used_server_num = server_num - server_allocate_num.count(0)
    server_used_prop = used_server_num / server_num

    return user_allocated_prop, server_used_prop


def get_all_server_xy():
    """Retrieve all server coordinates from a dataset."""
    server_list = []
    file = open("data/site-optus-melbCBD.csv", 'r')
    file.readline().strip()  # Skip the first row (header information)
    lines = file.readlines()
    for i in range(len(lines)):
        result = lines[i].split(',')
        # Extract longitude and latitude
        server_mes = (float(result[2]), float(result[1]))
        x, y = miller_to_xy(*server_mes)
        server_list.append([x, y])
    file.close()

    # Normalize coordinates
    server_list = np.array(server_list)
    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy

    # Rotate by 13 degrees
    angle = 13
    for xy in server_list:
        x = xy[0] * math.cos(math.pi / 180 * angle) - xy[1] * math.sin(math.pi / 180 * angle)
        y = xy[0] * math.sin(math.pi / 180 * angle) + xy[1] * math.cos(math.pi / 180 * angle)
        xy[0] = x
        xy[1] = y

    # Normalize again and apply transformations
    min_xy = np.min(server_list, axis=0)
    server_list -= min_xy
    for xy in server_list:
        xy[0] = xy[0] - xy[1] * math.tan(math.pi / 180 * 15)

    # Convert to 100m scale
    server_list /= 100

    return server_list


def miller_to_xy(lon, lat):
    """Convert latitude and longitude to Miller cylindrical projection coordinates."""
    L = 6381372 * math.pi * 2  # Earth's circumference
    W = L  # Treat circumference as X-axis
    H = L / 2  # Approximate Y-axis as half of the circumference
    mill = 2.3  # Miller projection constant (range: ±2.3)
    x = lon * math.pi / 180  # Convert longitude to radians
    y = lat * math.pi / 180
    y = 1.25 * math.log(math.tan(0.25 * math.pi + 0.4 * y))  # Miller projection transformation
    # Convert radians to actual distances
    x = (W / 2) + (W / (2 * math.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    return x, y


def init_server(x_start_prop, x_end_prop, y_start_prop, y_end_prop, max_cov=1.5, miu=35, sigma=10):
    """Extract a subset of servers from the map based on specified proportions."""
    server_xy_list = get_all_server_xy()
    max_x_y = np.max(server_xy_list, axis=0)
    x_start = max_x_y[0] * x_start_prop
    x_end = max_x_y[0] * x_end_prop
    y_start = max_x_y[1] * y_start_prop
    y_end = max_x_y[1] * y_end_prop

    filter_server = [x_start <= server[0] <= x_end and y_start <= server[1] <= y_end for server in server_xy_list]
    server_xy_list = server_xy_list[filter_server]
    min_xy = np.min(server_xy_list, axis=0)
    server_xy_list = server_xy_list - min_xy + max_cov
    server_capacity_list = np.random.normal(miu, sigma, size=(len(server_xy_list), 4))
    server_list = np.concatenate((server_xy_list, server_capacity_list), axis=1)
    return server_list



def init_user_position_and_mask_by_servers(server_list, user_num, max_cov):
    max_server = np.max(server_list, axis=0)
    max_x = max_server[0] + max_cov
    max_y = max_server[1] + max_cov
    min_server = np.min(server_list, axis=0)
    min_x = min_server[0] - max_cov
    min_y = min_server[1] - max_cov

    user_x_list = np.random.uniform(min_x, max_x, (user_num, 1))
    user_y_list = np.random.uniform(min_y, max_y, (user_num, 1))
    user_list = np.concatenate((user_x_list, user_y_list), axis=1)
    user_list, users_masks = get_within_servers(user_list, server_list, min_x, max_x, min_y, max_y, max_cov)
    return {"users_positions": user_list, "users_masks": users_masks}

def init_users_list_by_positions_and_mask(user_position_and_mask, data_num, total_sec, avg_user_num_per_sec, user_stay_miu, user_stay_sigma):
    """
    Generate user data based on user positions and masks.
    
    :param user_position_and_mask: User positions and masks
    :param data_num: Number of datasets to generate
    :param total_sec: Total number of seconds for data generation
    :param avg_user_num_per_sec: Average number of users generated per second
    :param user_stay_miu: Mean value of user stay duration
    :param user_stay_sigma: Standard deviation of user stay duration
    :return:
    """

    # First, calculate the total number of users to generate
    user_num = total_sec * avg_user_num_per_sec
    user_positions = user_position_and_mask["users_positions"]
    user_masks = user_position_and_mask["users_masks"]
    users_list = []
    users_masks_list = []
    # users_nums_per_sec = []
    
    # Generate each dataset
    for _ in tqdm(range(data_num)):
        # Randomly generate indices for `user_num` users
        user_index = np.random.randint(0, len(user_positions), user_num)
        user_list = user_positions[user_index]
        user_mask_list = user_masks[user_index]
        # Assign workloads to users
        user_workload_list = np.array([random_user_load() for _ in range(user_num)])
        # Assign stay durations using a normal distribution, rounding up to ensure minimum stay is 1
        user_stay_list = np.random.normal(user_stay_miu, user_stay_sigma, user_num)
        user_stay_list = np.ceil(user_stay_list)
        user_stay_list = np.maximum(user_stay_list, 1)
        # Concatenate data into a complete user dataset
        user_list = np.concatenate((user_list, user_workload_list, user_stay_list.reshape(-1, 1)), axis=1)
        users_list.append(user_list)
        users_masks_list.append(user_mask_list)

        # # Additionally, randomly distribute `user_num` users across `total_sec` time periods using a uniform distribution
        # # Generate `num_users` random numbers in the range [0, total_sec-1], representing user time periods
        # user_time_periods = np.random.randint(0, total_sec, user_num)
        # # Use `np.bincount` to count the number of users per time period
        # user_num_per_sec = np.bincount(user_time_periods, minlength=total_sec)
        # users_nums_per_sec.append(user_num_per_sec)

    return {"users_list": users_list, "users_masks_list": users_masks_list}  # "users_nums_per_sec": users_nums_per_sec
