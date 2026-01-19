from typing import List, Tuple
from fish_tracking.common.math_utils import *
from fish_tracking.common.globals import *


class Camera:
    img_normal = np.array([0.0, 0.0, 0.0])
    img_up = np.array([0.0, 0.0, 0.0])
    img_center = np.array([0.0, 0.0, 0.0])
    cam_center = np.array([0.0, 0.0, 0.0])

    top_right_corner = np.array([0.0, 0.0, 0.0])
    top_left_corner = np.array([0.0, 0.0, 0.0])
    bottom_left_corner = np.array([0.0, 0.0, 0.0])
    bottom_right_corner = np.array([0.0, 0.0, 0.0])
    left_right_vector = np.array([0.0, 0.0, 0.0])
    top_bottom_vector = np.array([0.0, 0.0, 0.0])

    def __init__(self,
                 focal_length=0.0,
                 img_width=0.0,
                 img_height=0.0,
                 width_pixels=0,
                 height_pixels=0,
                 opening_angle_x=0.0,
                 opening_angle_y=0.0):
        # Validate if non-default values are provided
        if any(v > 0 for v in [focal_length, img_width, img_height, width_pixels, height_pixels]):
            if focal_length <= 0 or img_width <= 0 or img_height <= 0 or width_pixels <= 0 or height_pixels <= 0:
                raise ValueError("Invalid initialization values: all dimensions must be positive")

        self.focal_length = focal_length
        self.img_width = img_width
        self.img_height = img_height
        self.width_pixels = width_pixels
        self.height_pixels = height_pixels
        self.opening_angle_x = opening_angle_x
        self.opening_angle_y = opening_angle_y

    def copy(self):
        new_cam = Camera()
        new_cam.focal_length = self.focal_length
        new_cam.img_width = self.img_width
        new_cam.img_height = self.img_height
        new_cam.width_pixels = self.width_pixels
        new_cam.height_pixels = self.height_pixels
        new_cam.opening_angle_x = self.opening_angle_x
        new_cam.opening_angle_y = self.opening_angle_y
        new_cam.img_normal = self.img_normal
        new_cam.img_up = self.img_up
        new_cam.img_center = self.img_center
        new_cam.cam_center = self.cam_center
        new_cam.top_right_corner = self.top_right_corner
        new_cam.top_left_corner = self.top_left_corner
        new_cam.bottom_left_corner = self.bottom_left_corner
        new_cam.bottom_right_corner = self.bottom_right_corner
        new_cam.left_right_vector = self.left_right_vector
        new_cam.top_bottom_vector = self.top_bottom_vector
        return new_cam

    def compute_image_pose(self, q, cam_pos):
        '''if not q[0] and not q.x and not q.y and not q.z:
            print("INVALID QUATERNION")
            return -1  # Return error type INVALID_QUATERNION
        if not cam_pos.x and not cam_pos.y  and not cam_pos:
            print("INVALID POSITION")
            return -1  # Return error type INVALID_VECTOR'''

        rot_mat = create_matrix_from_quaternion(q)  # Implement the create_matrix_from_quaternion() function

        self.img_normal = normalize_vector(
            np.dot(rot_mat, DEFAULT_IMG_NORMAL))  # Implement the normalize_vector() and apply_matrix() functions
        self.img_up = normalize_vector(
            np.dot(rot_mat, DEFAULT_IMG_UP))  # Implement the normalize_vector() and apply_matrix() functions
        self.img_center = cam_pos - (
                    self.focal_length * self.img_normal)  # Implement the add(), reverse(), and scalar_multiply() functions
        self.cam_center = np.copy(cam_pos)

        self.compute_image_plane_specs()

    def compute_image_plane_specs(self):
        # create unit vectors pointing to the right and down

        unit_right = normalize_vector(
            np.cross(self.img_up, self.img_normal))  # Implement the normalize_vector() function
        # unit_down =np.array(-self.img_up.x, -self.img_up.y, -self.img_up.z)  # Implement the normalize_vector()
        # function
        unit_down = normalize_vector(-self.img_up)  # Implement the normalize_vector() function

        if not np.linalg.norm(unit_right) or not np.linalg.norm(unit_down):
            print("Error: INVALID VECTORS")
            exit(-1)  # Return error type INVALID_VECTOR
        self.top_right_corner = self.img_center + (-(self.img_height / 2.0) * unit_down) + (
                    (self.img_width / 2.0) * unit_right)
        rev_unit_right = -unit_right

        self.top_left_corner = self.top_right_corner + self.img_width * rev_unit_right
        # print("top_left_corner: ", self.top_left_corner.x, self.top_left_corner.y, self.top_left_corner.z)

        self.bottom_left_corner = self.top_left_corner + self.img_height * unit_down
        self.bottom_right_corner = self.bottom_left_corner + self.img_width * unit_right
        self.left_right_vector = self.top_right_corner - self.top_left_corner
        self.top_bottom_vector = self.bottom_left_corner - self.top_left_corner
        # print("top_left_corner: ", self.top_left_corner.x, self.top_left_corner.y, self.top_left_corner.z)

    def world_to_image_space(self, point):

        # vector from top left corner to the point
        top_left_corner_point_vector = (point - self.top_left_corner)

        # projection length of the above vector onto the horizontal image plane edge
        x_pos = np.dot(self.left_right_vector, top_left_corner_point_vector) / self.img_width
        # projection length of the above vector onto the vertical image plane edge
        y_pos = np.dot(self.top_bottom_vector, top_left_corner_point_vector) / self.img_height

        # convert relative world space position to pixel position
        r = np.array([int(round((self.width_pixels * x_pos) / self.img_width)),
                      int(round((self.height_pixels * y_pos) / self.img_height))])

        # ensure image space boundaries are not exceeded
        if r[0] < 0:
            r[0] = 0
        if r[0] >= self.width_pixels:
            r[0] = self.width_pixels - 1

        if r[1] < 0:
            r[1] = 0
        if r[1] >= self.height_pixels:
            r[1] = self.height_pixels - 1

        return r

    def image_to_world_space(self, point):

        return self.top_left_corner + (point[0] * self.left_right_vector / self.width_pixels) + (
                    point[1] * self.top_bottom_vector / self.height_pixels)

    def get_cam_center(self):
        return self.cam_center

    def get_img_center(self):
        return self.img_center

    def get_img_normal(self):
        return self.img_normal

    def get_width_pixels(self):
        return self.width_pixels

    def get_height_pixels(self):
        return self.height_pixels

    def get_opening_angle_x(self):
        return self.opening_angle_x

    def get_opening_angle_y(self):
        return self.opening_angle_y

    def project_from_image_frame_to_ground(self, point):
        image_world_p = self.image_to_world_space(point)

        line_to_ground = image_world_p - self.cam_center
        # from image plane to ground
        r = line_plane_intersection(OCEAN_POINT, OCEAN_NORMAL, self.cam_center, line_to_ground)
        return r

    def project_from_ground_to_image_frame(self, point):
        line_to_image_plane = self.cam_center - point
        new_point_image_plane = line_plane_intersection(self.img_center, self.img_normal, self.cam_center,
                                                        line_to_image_plane)
        r = self.world_to_image_space(new_point_image_plane)
        return r

    '''
    def compute_projection_set(self, grid_resolution: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        r = []

        # corners of projected image frame on ocean surface
        corners = [np.zeros(3)] * 4
        # rays from camera origin through corners of image frame
        frustum_lines = [None] * 4
        # lines connecting the projected corners
        image_boundaries = [None] * 4

        frustum_lines[0] = create_line(self.cam_center, self.top_left_corner)
        frustum_lines[1] = create_line(self.cam_center, self.top_right_corner)
        frustum_lines[2] = create_line(self.cam_center, self.bottom_right_corner)
        frustum_lines[3] = create_line(self.cam_center, self.bottom_left_corner)

        min_x = float('inf')
        max_x = -float('inf')
        min_y = float('inf')
        max_y = -float('inf')

        for i in range(4):
            corners[i] = line_plane_intersection(OCEAN_POINT, OCEAN_NORMAL, frustum_lines[i].loc, frustum_lines[i].dir)
            if corners[i][0] > max_x:
                max_x = corners[i][0]
            if corners[i][0] < min_x:
                min_x = corners[i][0]
            if corners[i][1] > max_y:
                max_y = corners[i][1]
            if corners[i][1] < min_y:
                min_y = corners[i][1]

            corners[i][2] = 0.0

            # draw edges after corners are projected onto the ground
            if i == 3:
                image_boundaries[i] = self.create_line(corners[0], corners[i])
            if i > 0:
                image_boundaries[i - 1] = self.create_line(corners[i], corners[i - 1])

        # compute boundaries for grid lines
        min_x_int = int(min_x / grid_resolution)
        max_x_int = int(max_x / grid_resolution)
        min_y_int = int(min_y / grid_resolution)
        max_y_int = int(max_y / grid_resolution)

        # check vertical grid lines
        for i in range(min_x_int, max_x_int + 1):
            new_pair = (np.zeros(3), np.zeros(3))
            candidates = []
            vertical_line = self.create_line(self.new_point(i * grid_resolution, 0.0, 0.0), self.new_point(i * grid_resolution, 1.0, 0.0))
            # check all four walls
            for j in range(4):
                ni = self.line_line_intersection(vertical_line.loc, vertical_line.loc + vertical_line.dir,
                                                 image_boundaries[j].loc, image_boundaries[j].loc + image_boundaries[j].dir)
                # check if intersection is inside the boundaries, any intersection outside is irrelevant
                if self.inside_four_walls_2d(corners, ni, NUM_TOL_POS):
                    candidates.append(ni)
            # any line passing through a convex polygon can have two intersections at most
            if len(candidates) != 2:
                continue
            # save the grid line as a pair of points
            new_pair[0] = candidates[0]
            new_pair[1] = candidates[1]
            r.append(new_pair)

        # check horizontal grid
        # check horizontal grid lines
        for i in range(min_y_int, max_y_int + 1):
            new_pair = (np.zeros(3), np.zeros(3))
            candidates = []
            horizontal_line = self.create_line(self.new_point(0.0, i * grid_resolution, 0.0),
                                               self.new_point(1.0, i * grid_resolution, 0.0))
            # check all four walls
            for j in range(4):
                ni = self.line_line_intersection(horizontal_line.loc, horizontal_line.loc + horizontal_line.dir,
                                                 image_boundaries[j].loc,
                                                 image_boundaries[j].loc + image_boundaries[j].dir)
                # check if intersection is inside the boundaries, any intersection outside is irrelevant
                if self.inside_four_walls_2d(corners, ni, NUM_TOL_POS):
                    candidates.append(ni)
            # any line passing through a convex polygon can have two intersections at most
            if len(candidates) != 2:
                continue
            # save the grid line as a pair of points
            new_pair[0] = candidates[0]
            new_pair[1] = candidates[1]
            r.append(new_pair)

        retur
  '''





