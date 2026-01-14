import cv2
import random
from fish_tracking.pixel_tracking.tracker import Particle
from fish_tracking.common.math_utils import gauss_function, create_rand_decimal
from fish_tracking.common.globals import *


def pf_init(width, height, particle_count, segmentation_maps, multi_class, frame):
    # Convert the segmentation maps to grayscale
    if not multi_class:
        particles=[]
        for i in range(particle_count):
            particles.append(
                Particle(random.randint(0, width - 1), random.randint(0, height - 1), PARTICLE_WEIGHT_MULTIPLIER, frame))

        return [particles], 1


    grayscale_maps = [cv2.cvtColor(map, cv2.COLOR_BGR2GRAY) for map in segmentation_maps]

    # Compute the average grayscale segmentation map
    average_map = np.mean(grayscale_maps, axis=0)

    # Convert the average map to binary
    ret, binary_map = cv2.threshold(average_map.astype(np.uint8), FOUND_THRESHOLD, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw and show contours (optional)

    #average_map = cv2.cvtColor(average_map.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    #bla = cv2.drawContours(average_map, contours, -1, (0, 0, 255), 3)
    #cv2.imshow("contours", bla)
    #cv2.waitKey(0)

    # Step 2-4: Initialize particle filters around each contour
    particle_filters = []
    particle_radius = 10  # Radius within which particles are initialized
    num_filters = len(contours)


    for contour in contours:
        # Step 3: Determine the position for the particle filter
        contour_center = np.mean(contour, axis=0)[0].astype(int)

        # Step 4: Set the initial particles
        particles = []
        for _ in range(particle_count):
            x = np.random.randint(contour_center[0] - particle_radius, contour_center[0] + particle_radius)
            y = np.random.randint(contour_center[1] - particle_radius, contour_center[1] + particle_radius)
            particles.append(Particle(x, y, PARTICLE_WEIGHT_MULTIPLIER))

        particle_filters.append(particles)

    return particle_filters,len(particle_filters)


def pf_reset(particles, width, height, particle_count, frame):
    particles.clear()
    for i in range(particle_count):
        particles.append(Particle(random.randint(0, width-1), random.randint(0, height-1), PARTICLE_WEIGHT_MULTIPLIER, frame))

    return particles


def pf_prediction(particles, current_camera, last_camera, width, height):
    for p in particles:
        # apply gaussian noise
        # this gives a random number between 0 and 3 * STD_DEV
        tmp_rand = create_rand_decimal( 3 * STD_DEV)
        distance = MAX_NOISE * (GAUSS_MAX - gauss_function(tmp_rand))
        tmp_rand = create_rand_decimal(2 * math.pi)
        angle = tmp_rand - math.pi
        p.move(distance, angle, width, height)
        # apply optical flow compensation
        point_on_ground = last_camera.project_from_image_frame_to_ground(p.position)
        # and now back with current camera
        new_image_point = current_camera.project_from_ground_to_image_frame(point_on_ground)
        image_flow = (new_image_point[0] - p.position[0], new_image_point[1] - p.position[1])
        p.move_of(image_flow[0], image_flow[1], width, height)

    return particles



def pf_correction(particles, img):
    found = False
    for p in particles:
        weight = float(img[p.position[1], p.position[0], 0])
        p.weight= weight * PARTICLE_WEIGHT_MULTIPLIER
        if weight > FOUND_THRESHOLD:
            found = True

    return particles,found

def pf_resampling_narrow(particles, width, height):
    particle_count = len(particles)
    acc_weights = np.zeros(particle_count)
    acc_weights[0] = particles[0].get_weight()
    for i in range(1, particle_count):
        acc_weights[i] = acc_weights[i - 1] + particles[i].get_weight()
    total_weight = acc_weights[particle_count - 1]

    for p_i in range(len(particles)):
        index = 0
        if total_weight > 0:
            tmp_rand = create_rand_decimal(acc_weights[particle_count - 1])
            while acc_weights[index] < tmp_rand:
                index += 1
        else:
            index = random.randint(0, particle_count - 1)

        particles[p_i].set_position(particles[index].position[0], particles[index].position[1], width, height)

    return particles

def create_rand_array(max_value, size):
    RAND_ACCURACY = 10000.0
    if max_value == 0:
        return np.zeros(size)
    tmp_rand = np.random.randint(0, int(RAND_ACCURACY * max_value), size=size)
    return tmp_rand / RAND_ACCURACY

def pf_resampling(particles, width, height):
    particle_count = len(particles)
    acc_weights = np.cumsum([particles[i].get_weight() for i in range(particle_count)])
    total_weight = acc_weights[particle_count - 1]

    random_values = create_rand_array(total_weight, particle_count)
    indices = np.searchsorted(acc_weights, random_values)

    particles_positions = np.array([particles[index].position for index in indices])
    if total_weight <=0:
        random.shuffle(particles_positions)  # Randomly shuffle the particle positions array
    for p_i in range(len(particles)):
        particles[p_i].set_position(particles_positions[p_i][0], particles_positions[p_i][1], width, height)

    return particles


def compute_particle_set_screen_radius(particles, width):
    x_min = width
    x_max = 0
    for p in particles:
        x_min = min(x_min, p.get_position().x)
        x_max = max(x_max, p.get_position().x)
    return (x_max - x_min) // (2 * math.sqrt(2))


def create_particle_set(center, radius, particle_count):
    particles = []
    for i in range(particle_count):
        angle = create_rand_decimal(2.0 * math.pi)
        magnitude = create_rand_decimal( 1.0)
        angle -= math.pi
        x = center[0] + radius * magnitude * math.cos(angle)
        y = center[1] + radius * magnitude * math.sin(angle)
        particles.append(Particle(x, y, PARTICLE_WEIGHT_MULTIPLIER))
    return particles


def merge_particle_sets(normal, reverse, merged, confidence_normal, confidence_reverse, merged_confidence):
    if len(normal) != len(reverse) or len(normal) != len(confidence_normal) or len(normal) != len(confidence_reverse):
        print("Error in merge_particle_set: particle sets have different sizes")
        exit(-1)

    vector_size = len(normal)
    merged.clear()

    merged_confidence.clear()

    for i in range(vector_size):
        conf_normal = confidence_normal[i]
        conf_reverse = confidence_reverse[vector_size - i - 1]
        merged.append(normal[i] if conf_normal > conf_reverse else reverse[vector_size - i - 1])
        merged_confidence.append(max(conf_normal, conf_reverse))


    return merged,merged_confidence


def compute_particle_set_com(particle_set):
    particle_count = len(particle_set)
    if particle_count == 0:
        return (0, 0)
    x_sum = 0
    y_sum = 0
    for p in particle_set:
        x_sum += p.position[0]
        y_sum += p.position[1]
    return np.array([x_sum // particle_count, y_sum // particle_count])

def smoothe_particle_set(merged, merged_confidence):
    if len(merged) != len(merged_confidence):
        print("Error in smoothe_particle_set: particle sets have different sizes")
        exit(-1)
    if len(merged) == 0:
        print("Error: particle set is empty")
        exit(-1)
    particle_count = len(merged[0])

    i = 1
    while i < len(merged):

        if merged_confidence[i] > CONFIDENCE_THRESHOLD:
            i += 1
            continue
        first_com = compute_particle_set_com(merged[i - 1])
        ind_1 = i - 1


        while merged_confidence[i] <= CONFIDENCE_THRESHOLD:
            i += 1
            if i >= (len(merged) - 1):
                break

        if i >= (len(merged) - 1):
            break
        ind_2 = i
        second_com = compute_particle_set_com(merged[i])
        back_vector = (first_com[0] - second_com[0], first_com[1] - second_com[1])
        for j in range(ind_2 - 1, ind_1, -1):
            interpolated = (
                int(second_com[0] + back_vector[0] * (ind_2 - j) / (ind_2 - ind_1)),
                int(second_com[1] + back_vector[1] * (ind_2 - j) / (ind_2 - ind_1))
            )
            merged[j] = create_particle_set(interpolated, INTERPOLATION_RADIUS, particle_count)

    return merged

def compute_num_components(particles, width, height):
    ccs = []
    input_image = np.zeros((height, width), dtype=np.uint8)
    for index,p in enumerate(particles):
        cv2.circle(input_image, (int(p.position[0]),int(p.position[1])), PARTICLE_RADIUS_DEFAULT, 255, -1)

    num_labels,labels, stats, centroids = cv2.connectedComponentsWithStats(input_image)
    num_components = 0
    for i in range(stats.shape[0]):
        if stats[i, cv2.CC_STAT_AREA] > MIN_COMPONENT_SIZE and stats[i, cv2.CC_STAT_AREA] < MAX_COMPONENT_SIZE:
            num_components += 1
            ccs.append((int(centroids[i, 0]), int(centroids[i, 1])))
    return num_components, ccs

def write_pixel_statistics(trackers, csv_file, filename):
    with open(csv_file, 'w') as file:
        file.write("filename,frame,id,label,score,x,y,angle\n")

        for tracker in trackers:
            last_avg_position = None  # Store the position from the previous frame
            #print("len memory: ", len(tracker.memory))
            #print("len confidence: ", len(tracker.confidence))

            for i, particles in enumerate(tracker.memory):


                frame_num = tracker.start_frame + i
                avg_position = np.mean([p.position for p in particles], axis=0)

                if last_avg_position is not None:  # If there's a previous position, compute the angle and write to file
                    delta_x = avg_position[0] - last_avg_position[0]
                    delta_y = avg_position[1] - last_avg_position[1]
                    theta = math.atan2(delta_y, delta_x)
                    to_write = "{},{},{},{},{},{},{},{}\n".format(filename, frame_num, tracker.id, tracker.label,
                                                              tracker.confidence[i], avg_position[0], avg_position[1],
                                                              theta)
                    file.write(to_write)


                last_avg_position = avg_position

        file.close()



def write_final_particle_csv(memory, csv_file):

    num_particles = len(memory[0])
    with open(csv_file, 'w') as file:
        for particle_set in memory:
            for j,particle in enumerate(particle_set):
                to_write = str(int(particle.position[0])) + "," + str(int(particle.position[1]))
                file.write(to_write)
                if j != num_particles  - 1:
                    file.write(",")
                else:
                    file.write("\n")
        file.close()
    return 0
